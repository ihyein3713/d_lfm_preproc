#!/usr/bin/env python
"""ADNI T1 series preprocessing (no cohort/pair CSVs -- that's s1/s2/s3).

Raw tree is DICOM: <raw_dir>/<subject_id>/<sequence_folder>/<date_time>/<image_id>/*.dcm
Run with --list-only first to check which series get picked up.

Usage:
    python preprocess_adni.py \
        --raw-dir /mnt/d/d_lfm_preproc/ADNI_raw/ADNI_complete \
        --out-dir /mnt/d/d_lfm_preproc/adni/processed \
        --list-only

    python preprocess_adni.py \
        --raw-dir /mnt/d/d_lfm_preproc/ADNI_raw/ADNI_complete \
        --out-dir /mnt/d/d_lfm_preproc/adni/processed \
        --reference-brain /path/to/MNI152_T1_1mm_brain.nii.gz \
        --limit 5   # sanity-check a handful before the full batch

Env var fallbacks: RAW_DATA_DIR, OUT_DIR, REFERENCE_BRAIN, FS_LICENSE.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pipeline

logger = logging.getLogger("preprocess_adni")

# Substring match, not an exact-name allowlist: Siemens/Philips use MPRAGE/
# MP-RAGE, GE uses (IR-)SPGR/(IR-)FSPGR. Excludes FLAIR/DTI/rsfMRI/ASL/
# field-map/localizer/HighResHippocampus, which this T1-tuned pipeline isn't for.
DEFAULT_T1_PATTERNS = ["MPRAGE", "MP-RAGE", "SPGR"]

# mri_synthstrip and mri_synthseg both run on the GPU and share this lock:
# SynthSeg alone peaks at ~10.7 of the 11 GB, so any overlap risks CUDA OOM.
gpu_lock = threading.Lock()

# CPU SynthStrip peaks at ~8-10 GB RSS per process on 16-25 MB inputs, so
# several in parallel would exhaust RAM; run one at a time with more threads.
strip_cpu_lock = threading.Lock()


def discover_series(raw_dir: Path, patterns: list) -> list:
    """Return (subject_id, dicom_series_dir) pairs: one per leaf directory
    containing .dcm files, under any <subject>/<sequence>/ folder whose name
    contains one of `patterns` (case-sensitive substring match). Visit date
    and image ID depth aren't assumed fixed -- we just walk from the
    sequence folder down to wherever the .dcm files are.
    """
    series = []
    for subject_dir in sorted(p for p in raw_dir.iterdir() if p.is_dir()):
        subject_id = subject_dir.name
        for seq_dir in sorted(p for p in subject_dir.iterdir() if p.is_dir()):
            if not any(pattern in seq_dir.name for pattern in patterns):
                continue
            for dirpath, _dirnames, filenames in os.walk(seq_dir):
                if any(f.lower().endswith(".dcm") for f in filenames):
                    series.append((subject_id, Path(dirpath)))
    return series


def output_paths(dicom_dir: Path, raw_dir: Path, out_dir: Path, resolutions: list) -> dict:
    # Flattened to <subject>/<image_id>/: the sequence and visit-date levels are
    # dropped since image_id is already unique within a subject.
    subject_id = dicom_dir.relative_to(raw_dir).parts[0]
    image_id = dicom_dir.name
    base_dir = out_dir / subject_id / image_id
    paths = {
        "raw_nifti_dir": base_dir,
        "raw_nifti_stem": f"{image_id}_step0_raw",
        "biasfield": base_dir / f"{image_id}_step1_biasfield.nii.gz",
        "stripped": base_dir / f"{image_id}_step2_stripped.nii.gz",
        "registered": base_dir / f"{image_id}_step3_registered.nii.gz",
        "synthseg": base_dir / f"{image_id}_synthseg_native.nii.gz",
        "final": base_dir / f"{image_id}_final_native.nii.gz",
        "synthseg_resampled": {},
        "final_resampled": {},
    }
    for res in resolutions:
        tag = f"{res:g}mm"
        paths["synthseg_resampled"][res] = base_dir / f"{image_id}_synthseg_{tag}.nii.gz"
        paths["final_resampled"][res] = base_dir / f"{image_id}_final_{tag}.nii.gz"
    return paths


def is_complete(paths: dict) -> bool:
    # Intermediates are deleted once a series finishes, so the per-step existence
    # checks alone would redo the whole chain; the resampled outputs are the marker.
    finals = list(paths["synthseg_resampled"].values()) + list(paths["final_resampled"].values())
    return all(p.exists() for p in finals)


def process_series(subject_id, dicom_dir, raw_dir, out_dir, reference_brain, resolutions, gpu,
                    keep_intermediates, keep_native_res, seg_env, strip_device, strip_threads) -> dict:
    paths = output_paths(dicom_dir, raw_dir, out_dir, resolutions)
    start = time.time()
    try:
        raw_nifti = pipeline.dicom_to_nifti(dicom_dir, paths["raw_nifti_dir"], paths["raw_nifti_stem"])

        pipeline.n4_bias_correction(raw_nifti, paths["biasfield"])

        if gpu and strip_device == "gpu":
            with gpu_lock:
                pipeline.skull_strip(paths["biasfield"], paths["stripped"], gpu=True)
        else:
            with strip_cpu_lock:
                pipeline.skull_strip(paths["biasfield"], paths["stripped"], gpu=False, threads=strip_threads)

        pipeline.register_to_reference(paths["stripped"], paths["registered"], reference_brain)

        if gpu:
            with gpu_lock:
                pipeline.segment(paths["registered"], paths["synthseg"], gpu=True, env=seg_env)
        else:
            pipeline.segment(paths["registered"], paths["synthseg"], gpu=False)

        pipeline.normalize_intensity(paths["registered"], paths["final"])

        for res in resolutions:
            pipeline.resample(paths["synthseg"], paths["synthseg_resampled"][res], res, is_segmentation=True)
            pipeline.resample(paths["final"], paths["final_resampled"][res], res, is_segmentation=False)

        if not keep_intermediates:
            transforms = list(paths["raw_nifti_dir"].glob(f"{paths['registered'].name}_transform_*"))
            pipeline.cleanup([raw_nifti, paths["biasfield"], paths["stripped"], paths["registered"]] + transforms)
        if not keep_native_res:
            pipeline.cleanup([paths["synthseg"], paths["final"]])

        return {"subject_id": subject_id, "input": str(dicom_dir), "status": "ok",
                "seconds": round(time.time() - start, 1)}
    except pipeline.StepError as e:
        logger.error("failed %s: %s", dicom_dir, e)
        return {"subject_id": subject_id, "input": str(dicom_dir), "status": "failed", "step": e.step, "error": str(e)}
    except Exception as e:
        logger.exception("unexpected failure on %s", dicom_dir)
        return {"subject_id": subject_id, "input": str(dicom_dir), "status": "failed",
                "step": type(e).__name__, "error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="Preprocess ADNI T1 series (no CSV output).")
    parser.add_argument("--raw-dir", default=os.environ.get("RAW_DATA_DIR"))
    parser.add_argument("--out-dir", default=os.environ.get("OUT_DIR"))
    parser.add_argument("--reference-brain",
                         default=os.environ.get("REFERENCE_BRAIN", str(Path.home() / "tools" / "MNI152_T1_1mm_brain.nii.gz")))
    parser.add_argument("--fs-license", default=os.environ.get("FS_LICENSE", str(Path.home() / "tools" / "license.txt")))
    parser.add_argument("--sequences", default=",".join(DEFAULT_T1_PATTERNS),
                         help="Comma-separated substrings (case-sensitive); a sequence folder matches if its name contains any of them.")
    parser.add_argument("--resolutions", default="1.0,1.5",
                         help="Comma-separated isotropic resolutions (mm) to resample synthseg/final to. All are kept -- disk is not the constraint here.")
    parser.add_argument("--cpu-workers", type=int, default=4,
                         help="Concurrency for the CPU-only steps (dcm2niix, N4, registration). GPU steps are always serialized.")
    parser.add_argument("--gpu", dest="gpu", action="store_true", default=True)
    parser.add_argument("--no-gpu", dest="gpu", action="store_false")
    parser.add_argument("--strip-device", choices=["gpu", "cpu"], default="gpu",
                         help="Where SynthStrip runs. 'cpu' takes it out of the GPU lock so it runs in the "
                              "CPU worker pool; SynthSeg stays on the GPU.")
    parser.add_argument("--strip-threads", type=int, default=None,
                         help="PyTorch CPU threads per SynthStrip process when --strip-device cpu.")
    parser.add_argument("--tfcuda-site-packages",
                         default=os.environ.get("TFCUDA_SITE_PACKAGES",
                                                str(Path.home() / "miniconda3/envs/tfcuda/lib/python3.8/site-packages")),
                         help="site-packages holding the pip nvidia-* CUDA 11.8/cuDNN 8.6 wheels that FreeSurfer's TensorFlow needs for GPU SynthSeg.")
    parser.add_argument("--keep-intermediates", action="store_true",
                         help="Keep step0_raw/step1_biasfield/step2_stripped/step3_registered instead of deleting them once a series finishes.")
    parser.add_argument("--keep-native-res", action="store_true",
                         help="Keep native-resolution synthseg/final instead of deleting them once resampling finishes.")
    parser.add_argument("--list-only", action="store_true", help="Discover series and print them -- no processing.")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N discovered series (debugging).")
    parser.add_argument("--log-file", default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if not args.raw_dir or not args.out_dir:
        parser.error("--raw-dir/--out-dir (or RAW_DATA_DIR/OUT_DIR) are required")

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    if not raw_dir.exists():
        parser.error(f"raw dir not found: {raw_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    patterns = [s.strip() for s in args.sequences.split(",") if s.strip()]
    series = discover_series(raw_dir, patterns)
    subjects = sorted({s for s, _ in series})
    logger.info("discovered %d T1 series across %d subjects under %s (patterns=%s)",
                len(series), len(subjects), raw_dir, patterns)

    if args.list_only:
        for subject_id, dicom_dir in series[:50]:
            n_dcm = sum(1 for _ in dicom_dir.glob("*.dcm"))
            print(subject_id, dicom_dir, f"({n_dcm} dcm files)")
        if len(series) > 50:
            print(f"... and {len(series) - 50} more")
        return

    reference_brain = Path(args.reference_brain)
    if not reference_brain.exists():
        parser.error(f"reference brain not found: {reference_brain} (put MNI152_T1_1mm_brain.nii.gz there, or pass --reference-brain)")

    if args.fs_license:
        if not Path(args.fs_license).exists():
            parser.error(f"FreeSurfer license not found: {args.fs_license}")
        os.environ["FS_LICENSE"] = args.fs_license

    pipeline.require_tools()

    seg_env = None
    if args.gpu:
        seg_env = pipeline.synthseg_gpu_env(Path(args.tfcuda_site_packages))
        logger.info("SynthSeg GPU preflight OK: %s", pipeline.check_synthseg_gpu(seg_env))

    resolutions = [float(r.strip()) for r in args.resolutions.split(",") if r.strip()]

    todo = [(s, d) for s, d in series if not is_complete(output_paths(d, raw_dir, out_dir, resolutions))]
    logger.info("%d already complete, %d to process", len(series) - len(todo), len(todo))
    series = todo

    if args.limit:
        series = series[: args.limit]

    log_path = Path(args.log_file) if args.log_file else out_dir / "preprocess_log.jsonl"
    results = []
    start = time.time()
    with ThreadPoolExecutor(max_workers=args.cpu_workers) as executor, log_path.open("a") as log_f:
        futures = {
            executor.submit(
                process_series, subject_id, dicom_dir, raw_dir, out_dir, reference_brain,
                resolutions, args.gpu, args.keep_intermediates, args.keep_native_res, seg_env,
                args.strip_device, args.strip_threads,
            ): (subject_id, dicom_dir)
            for subject_id, dicom_dir in series
        }
        for i, future in enumerate(as_completed(futures), 1):
            result = future.result()
            results.append(result)
            log_f.write(json.dumps(result) + "\n")
            log_f.flush()
            if i % 20 == 0 or i == len(series):
                logger.info("progress %d/%d (%.0fs elapsed)", i, len(series), time.time() - start)

    n_ok = sum(1 for r in results if r["status"] == "ok")
    n_failed = len(results) - n_ok
    logger.info("done: %d ok, %d failed. log: %s", n_ok, n_failed, log_path)


if __name__ == "__main__":
    main()
