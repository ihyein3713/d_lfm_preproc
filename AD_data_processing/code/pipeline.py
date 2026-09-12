"""Preprocessing steps for a single T1 MRI series (DICOM -> NIfTI -> N4 ->
SynthStrip -> ANTs registration -> SynthSeg -> WhiteStripe -> resample).

Each step shells out to a CLI tool (expected on PATH), raises StepError on a
non-zero exit, and skips itself if its output already exists (so a
crashed/interrupted run can be re-invoked).
"""
from __future__ import annotations

import logging
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
            "Activate the preprocessing conda env first (e.g. `mamba activate mri`)."
        )


def _run(step: str, cmd: list) -> None:
    logger.debug("[%s] %s", step, " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise StepError(step, cmd, result.returncode, result.stderr)


def dicom_to_nifti(dicom_dir: Path, out_dir: Path, stem: str) -> Path:
    """Convert one DICOM series to a compressed NIfTI, picking the largest
    file if dcm2niix splits the series into more than one output."""
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(out_dir.glob(f"{stem}*.nii.gz"))
    if existing:
        return existing[0]
    _run("dicom_to_nifti", [
        "dcm2niix", "-z", "y", "-b", "n", "-f", stem, "-o", str(out_dir), str(dicom_dir),
    ])
    produced = sorted(out_dir.glob(f"{stem}*.nii.gz"), key=lambda p: p.stat().st_size, reverse=True)
    if not produced:
        raise StepError("dicom_to_nifti", ["dcm2niix", str(dicom_dir)], -1, "no .nii.gz produced")
    if len(produced) > 1:
        logger.warning("dcm2niix produced %d files for %s, using the largest: %s",
                        len(produced), dicom_dir, produced[0].name)
    return produced[0]


def n4_bias_correction(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    # ITK picks the writer by extension, so the temp name must still end in .nii.gz.
    base = dst.name[: -len(".nii.gz")] if dst.name.endswith(".nii.gz") else dst.name
    tmp = dst.with_name(base + ".tmp.nii.gz")
    _run("n4_bias_correction", [
        "N4BiasFieldCorrection", "-d", "3", "-s", "4", "-i", str(src), "-o", str(tmp),
    ])
    tmp.rename(dst)


def skull_strip(src: Path, dst: Path, gpu: bool) -> None:
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["mri_synthstrip", "-i", str(src), "-o", str(dst)]
    if gpu:
        cmd.append("-g")
    _run("skull_strip", cmd)


def register_to_reference(src: Path, dst: Path, reference: Path) -> None:
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    transform_prefix = str(dst) + "_transform_"
    _run("register", [
        "antsRegistration", "-d", "3",
        "-n", "BSpline",
        "--shrink-factors", "8x4x2x1",
        "--convergence", "[1000x500x250x100,1e-6,10]",
        "--transform", "Affine[0.1]",
        "--smoothing-sigmas", "3x2x1x0vox",
        "-m", f"MI[{reference},{src},1,32,Regular,0.1]",
        "-o", f"[{transform_prefix},{dst}]",
    ])


def segment(src: Path, dst: Path, gpu: bool) -> None:
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["mri_synthseg", "--i", str(src), "--o", str(dst)]
    # No --gpu flag exists; GPU is the default, --cpu is the only override.
    if not gpu:
        cmd.append("--cpu")
    _run("segment", cmd)


def normalize_intensity(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    _run("normalize", [
        "intensity-normalize", "whitestripe", str(src), "-o", str(dst), "--modality", "t1",
    ])


def resample(src: Path, dst: Path, resolution: float, is_segmentation: bool) -> None:
    if dst.exists():
        return
    import nibabel as nib
    import numpy as np
    import scipy.ndimage

    dst.parent.mkdir(parents=True, exist_ok=True)
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

    nib.save(nib.Nifti1Image(resampled, new_affine), str(dst))


def cleanup(paths: list) -> None:
    for p in paths:
        if p.exists():
            p.unlink()
