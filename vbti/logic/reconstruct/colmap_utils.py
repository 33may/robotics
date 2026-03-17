"""
COLMAP reconstruction utilities.

Wraps nerfstudio's ns-process-data (COLMAP internally) and COLMAP image_undistorter.
Produces undistorted PINHOLE output ready for MILo training.

Usage:
    python -m vbti.logic.reconstruct.colmap_utils.py reconstruct --frames_dir data/frames --output_dir data/colmap
    python -m vbti.logic.reconstruct.colmap_utils.py run_colmap --frames_dir data/frames --output_dir data/colmap
    python -m vbti.logic.reconstruct.colmap_utils.py validate_models --colmap_dir data/colmap
    python -m vbti.logic.reconstruct.colmap_utils.py undistort --colmap_dir data/colmap --output_dir data/colmap_undistorted
"""

import shutil
import subprocess
from pathlib import Path

from loguru import logger


def run_colmap(
    frames_dir: str,
    output_dir: str,
    matching_method: str = "exhaustive",
):
    """Run COLMAP reconstruction via nerfstudio's ns-process-data.

    Args:
        frames_dir: path to directory with extracted frames (*.png)
        output_dir: where to write COLMAP output
        matching_method: "exhaustive" (recommended <500 imgs) or "vocab_tree"
    """
    frames_dir = str(Path(frames_dir).resolve())
    output_dir = str(Path(output_dir).resolve())

    cmd = [
        "ns-process-data", "images",
        "--data", frames_dir,
        "--output-dir", output_dir,
        "--matching-method", matching_method,
    ]

    logger.info(f"Running COLMAP via nerfstudio: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    sparse_dir = Path(output_dir) / "colmap" / "sparse"
    if not sparse_dir.exists():
        raise RuntimeError(f"COLMAP failed — no sparse directory at {sparse_dir}")

    models = sorted(sparse_dir.iterdir())
    logger.success(f"COLMAP finished. Found {len(models)} model(s) in {sparse_dir}")


def validate_models(colmap_dir: str) -> int:
    """Check COLMAP sparse models and swap the largest into slot 0.

    COLMAP mapper can produce multiple disconnected models. Nerfstudio blindly
    picks sparse/0, which may not be the largest. This function finds the model
    with the most registered images and moves it to sparse/0.

    Returns the number of registered images in the best model.
    """
    sparse_dir = Path(colmap_dir).resolve() / "colmap" / "sparse"
    if not sparse_dir.exists():
        raise FileNotFoundError(f"No sparse directory at {sparse_dir}")

    # Score each model by images.bin file size (correlates with registered image count)
    models = {}
    for model_dir in sorted(sparse_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        images_bin = model_dir / "images.bin"
        if images_bin.exists():
            models[model_dir.name] = images_bin.stat().st_size
            logger.debug(f"Model {model_dir.name}: images.bin = {images_bin.stat().st_size:,} bytes")

    if not models:
        raise RuntimeError(f"No valid models found in {sparse_dir}")

    best_model = max(models, key=lambda k: models[k])
    logger.info(f"Best model: {best_model} ({models[best_model]:,} bytes)")

    if best_model != "0":
        logger.warning(f"Swapping model {best_model} into slot 0...")
        slot_0 = sparse_dir / "0"
        backup = sparse_dir / "0_original"

        if slot_0.exists():
            shutil.move(str(slot_0), str(backup))
        shutil.move(str(sparse_dir / best_model), str(slot_0))

        logger.info(f"Old model 0 backed up to {backup}")

    return len(models)


def undistort(
    colmap_dir: str,
    output_dir: str,
):
    """Run COLMAP image_undistorter to convert OPENCV → PINHOLE camera model.

    MILo requires PINHOLE cameras (no lens distortion). This step removes
    distortion from images and rewrites cameras.bin with PINHOLE model.

    Args:
        colmap_dir: nerfstudio output dir (contains colmap/sparse/0 and images/)
        output_dir: where to write undistorted output
    """
    colmap_path = Path(colmap_dir).resolve()
    output_dir = str(Path(output_dir).resolve())

    image_path = colmap_path / "images"
    input_path = colmap_path / "colmap" / "sparse" / "0"

    if not image_path.exists():
        raise FileNotFoundError(f"No images directory at {image_path}")
    if not input_path.exists():
        raise FileNotFoundError(f"No sparse model at {input_path}")

    cmd = [
        "colmap", "image_undistorter",
        "--image_path", str(image_path),
        "--input_path", str(input_path),
        "--output_path", output_dir,
        "--output_type", "COLMAP",
    ]

    logger.info(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    # Verify output
    out_sparse = Path(output_dir) / "sparse"
    out_images = Path(output_dir) / "images"
    if not out_sparse.exists() or not out_images.exists():
        raise RuntimeError(f"Undistorter failed — missing output at {output_dir}")

    # COLMAP undistorter puts bins in sparse/, but MILo expects sparse/0/
    sparse_0 = out_sparse / "0"
    if not sparse_0.exists() and (out_sparse / "cameras.bin").exists():
        sparse_0.mkdir()
        for f in out_sparse.glob("*.bin"):
            f.rename(sparse_0 / f.name)
        logger.debug("Moved sparse bins into sparse/0/ (MILo convention)")

    n_images = len(list(out_images.glob("*")))
    logger.success(f"Undistorted {n_images} images -> {output_dir} (PINHOLE)")


def process_colmap(
    frames_dir: str,
    output_dir: str,
    matching_method: str = "exhaustive",
):
    """Full reconstruction pipeline: COLMAP → validate → undistort.

    This is the master entry point. Produces undistorted PINHOLE output
    ready to feed directly into MILo training.

    Args:
        frames_dir: path to extracted frames (*.png)
        output_dir: base output directory. Creates:
                    output_dir/colmap/     — raw COLMAP output
                    output_dir/undistorted/ — PINHOLE output for MILo
    """
    output_path = Path(output_dir).resolve()
    colmap_dir = str(output_path / "colmap")
    undistorted_dir = str(output_path / "undistorted")

    logger.info("Running COLMAP")
    run_colmap(frames_dir, colmap_dir, matching_method)

    logger.info("Validating models")
    validate_models(colmap_dir)

    logger.info("Undistorting images (OPENCV -> PINHOLE)")
    undistort(colmap_dir, undistorted_dir)

    logger.success(f"MILo-ready output at: {undistorted_dir}")


if __name__ == "__main__":
    import fire
    fire.core.Display = lambda lines, out: print(*lines, file=out)
    fire.Fire({
        "run_colmap": run_colmap,
        "validate_models": validate_models,
        "undistort": undistort,
        "reconstruct": process_colmap,
    })
