"""Preprocessing steps for a single T1 MRI series (DICOM -> NIfTI -> N4 ->
SynthStrip -> ANTs registration -> SynthSeg -> WhiteStripe -> resample).

Each step shells out to a CLI tool (expected on PATH), raises StepError on a
non-zero exit, and skips itself if its output already exists (so a
crashed/interrupted run can be re-invoked). Outputs are written to a temp
name and renamed only on success, so a kill mid-write never leaves a
truncated file under the final name for the existence check to trust.
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

REQUIRED_TOOLS = [
    "dcm2niix",
    "N4BiasFieldCorrection",
    "antsRegistration",
    "mri_synthstrip",
    "mri_synthseg",
    "intensity-normalize",
]


class StepError(RuntimeError):
    def __init__(self, step: str, cmd: list, returncode: int, stderr: str):
        super().__init__(f"{step} failed (exit {returncode}): {' '.join(cmd)}\n{stderr[-2000:]}")
        self.step = step
        self.cmd = cmd
        self.returncode = returncode


def require_tools(names: list = REQUIRED_TOOLS) -> None:
    missing = [n for n in names if shutil.which(n) is None]
    if missing:
        raise RuntimeError(
            f"Missing required tools on PATH: {', '.join(missing)}. "
            "Activate the `longi` conda env and source SetUpFreeSurfer.sh first."
        )


def _run(step: str, cmd: list, env: dict | None = None) -> None:
    logger.debug("[%s] %s", step, " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if result.returncode != 0:
        raise StepError(step, cmd, result.returncode, result.stderr)


def _tmp_path(dst: Path) -> Path:
    # Tools pick their writer by extension, so the temp name must still end in .nii.gz.
    base = dst.name[: -len(".nii.gz")] if dst.name.endswith(".nii.gz") else dst.name
    return dst.with_name(base + ".tmp.nii.gz")


def _prepare(dst: Path) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp_path(dst)
    tmp.unlink(missing_ok=True)
    return tmp


def synthseg_gpu_env(tfcuda_site_packages: Path) -> dict:
    """Environment that lets FreeSurfer's bundled TensorFlow load CUDA 11.8 /
    cuDNN 8.6 from the pip `nvidia-*` wheels in a separate conda env, without
    modifying the FreeSurfer install. Without it, mri_synthseg silently runs
    on CPU (~30x slower here)."""
    nvidia = tfcuda_site_packages / "nvidia"
    lib_dirs = sorted(str(p) for p in nvidia.glob("*/lib") if p.is_dir())
    if not lib_dirs:
        raise RuntimeError(f"no CUDA libraries found under {nvidia}")
    env = os.environ.copy()
    existing = [p for p in env.get("LD_LIBRARY_PATH", "").split(":") if p]
    env["LD_LIBRARY_PATH"] = ":".join(lib_dirs + ["/usr/lib/wsl/lib"] + existing)
    env["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    env["TF_CPP_MIN_LOG_LEVEL"] = "1"
    nvcc = nvidia / "cuda_nvcc"
    env["PATH"] = f"{nvcc / 'bin'}:{env.get('PATH', '')}"
    env["XLA_FLAGS"] = f"--xla_gpu_cuda_data_dir={nvcc}"
    return env


def check_synthseg_gpu(env: dict) -> str:
    """Fail fast if FreeSurfer's TensorFlow can't see a GPU under `env`."""
    code = ("import tensorflow as tf; g = tf.config.list_physical_devices('GPU'); "
            "print(tf.config.experimental.get_device_details(g[0])['device_name'] if g else '')")
    result = subprocess.run(["fspython", "-c", code], capture_output=True, text=True, env=env)
    name = result.stdout.strip().splitlines()[-1] if result.stdout.strip() else ""
    if result.returncode != 0 or not name:
        raise RuntimeError("FreeSurfer TensorFlow does not see a GPU; mri_synthseg would fall back "
                           f"to CPU. stderr tail:\n{result.stderr[-2000:]}")
    return name


def dicom_to_nifti(dicom_dir: Path, out_dir: Path, stem: str) -> Path:
    """Convert one DICOM series to a compressed NIfTI, picking the largest
    file if dcm2niix splits the series into more than one output."""
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(p for p in out_dir.glob(f"{stem}*.nii.gz") if ".tmp." not in p.name)
    if existing:
        return existing[0]
    work = out_dir / f".{stem}_dcm2niix"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir()
    _run("dicom_to_nifti", [
        "dcm2niix", "-z", "y", "-b", "n", "-f", stem, "-o", str(work), str(dicom_dir),
    ])
    produced = sorted(work.glob(f"{stem}*.nii.gz"), key=lambda p: p.stat().st_size, reverse=True)
    if not produced:
        shutil.rmtree(work, ignore_errors=True)
        raise StepError("dicom_to_nifti", ["dcm2niix", str(dicom_dir)], -1, "no .nii.gz produced")
    if len(produced) > 1:
        logger.warning("dcm2niix produced %d files for %s, using the largest: %s",
                        len(produced), dicom_dir, produced[0].name)
    dst = out_dir / f"{stem}.nii.gz"
    produced[0].rename(dst)
    shutil.rmtree(work, ignore_errors=True)
    return dst


def n4_bias_correction(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    tmp = _prepare(dst)
    _run("n4_bias_correction", [
        "N4BiasFieldCorrection", "-d", "3", "-s", "4", "-i", str(src), "-o", str(tmp),
    ])
    tmp.rename(dst)


def skull_strip(src: Path, dst: Path, gpu: bool, threads: int | None = None) -> None:
    if dst.exists():
        return
    tmp = _prepare(dst)
    cmd = ["mri_synthstrip", "-i", str(src), "-o", str(tmp)]
    if gpu:
        cmd.append("-g")
    elif threads:
        cmd += ["-t", str(threads)]
    _run("skull_strip", cmd)
    tmp.rename(dst)


def register_to_reference(src: Path, dst: Path, reference: Path) -> None:
    if dst.exists():
        return
    tmp = _prepare(dst)
    transform_prefix = str(dst) + "_transform_"
    _run("register", [
        "antsRegistration", "-d", "3",
        "-n", "BSpline",
        "--shrink-factors", "8x4x2x1",
        "--convergence", "[1000x500x250x100,1e-6,10]",
        "--transform", "Affine[0.1]",
        "--smoothing-sigmas", "3x2x1x0vox",
        "-m", f"MI[{reference},{src},1,32,Regular,0.1]",
        "-o", f"[{transform_prefix},{tmp}]",
    ])
    tmp.rename(dst)


def segment(src: Path, dst: Path, gpu: bool, env: dict | None = None) -> None:
    """GPU is mri_synthseg's default (there is no --gpu flag), but it only
    takes effect if TensorFlow can load CUDA -- pass synthseg_gpu_env()."""
    if dst.exists():
        return
    tmp = _prepare(dst)
    cmd = ["mri_synthseg", "--i", str(src), "--o", str(tmp)]
    if not gpu:
        cmd.append("--cpu")
    _run("segment", cmd, env=env if gpu else None)
    tmp.rename(dst)


def normalize_intensity(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    tmp = _prepare(dst)
    _run("normalize", [
        "intensity-normalize", "whitestripe", str(src), "-o", str(tmp), "--modality", "t1",
    ])
    tmp.rename(dst)


def resample(src: Path, dst: Path, resolution: float, is_segmentation: bool) -> None:
    if dst.exists():
        return
    import nibabel as nib
    import numpy as np
    import scipy.ndimage

    tmp = _prepare(dst)
    img = nib.load(str(src))
    data = img.get_fdata()
    affine = img.affine
    spacing = img.header.get_zooms()[:3]

    zoom_factors = [s / resolution for s in spacing]
    order = 0 if is_segmentation else 1
    resampled = scipy.ndimage.zoom(data, zoom_factors, order=order)
    resampled = resampled.astype(np.uint8 if is_segmentation else np.float32)

    new_affine = affine.copy()
    scale = np.array(spacing) / resolution
    new_affine[:3, :3] = affine[:3, :3] @ np.diag(1 / scale)

    nib.save(nib.Nifti1Image(resampled, new_affine), str(tmp))
    tmp.rename(dst)


def cleanup(paths: list) -> None:
    for p in paths:
        if p.exists():
            p.unlink()
