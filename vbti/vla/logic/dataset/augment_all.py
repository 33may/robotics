"""Augment all 3 black gripper datasets with detection + phase features.

Run after detection processing is complete:
    conda run -n lerobot python -m vbti.logic.dataset.augment_all

Checks that both detection_results.parquet and phase_labels.parquet exist
for each dataset before proceeding.
"""

from vbti.logic.dataset import resolve_dataset_path
from vbti.logic.dataset.augment import augment_dataset

DATASETS = [
    "eternalmay33/01_black_gripper_front",
    "eternalmay33/02_black_full_center",
    "eternalmay33/03_black_full_center_cups",
]


def check_ready(dataset: str) -> tuple[bool, str]:
    """Check if a dataset has both detection and phase parquets."""
    ds_path = resolve_dataset_path(dataset)
    det = ds_path / "detection_results.parquet"
    phase = ds_path / "phase_labels.parquet"
    if not det.exists():
        return False, f"missing {det.name}"
    if not phase.exists():
        return False, f"missing {phase.name}"
    return True, "ready"


def main():
    print("Checking datasets...\n")
    all_ready = True
    for ds in DATASETS:
        ready, msg = check_ready(ds)
        status = "OK" if ready else f"NOT READY ({msg})"
        print(f"  {ds}: {status}")
        if not ready:
            all_ready = False

    if not all_ready:
        print("\nSome datasets not ready. Run detection first:")
        print("  conda run -n lerobot python /tmp/run_all_detections.py")
        print("  conda run -n lerobot python -m vbti.logic.detection.phases <dataset>")
        return

    print("\nAll datasets ready. Starting augmentation...\n")

    for ds in DATASETS:
        short = ds.split("/")[-1]
        output = f"eternalmay33/{short}_aug"
        print(f"\n{'='*60}")
        print(f"Augmenting: {ds} -> {output}")
        print(f"{'='*60}")
        augment_dataset(
            source_dataset=ds,
            output_name=output,
            cameras=["left", "right", "top"],
            include_detection=True,
            include_phase=True,
            include_confidence=False,
        )

    print("\n\nAll done! Augmented datasets:")
    for ds in DATASETS:
        short = ds.split("/")[-1]
        output = f"eternalmay33/{short}_aug"
        out_path = resolve_dataset_path(output)
        print(f"  {output}: {out_path}")


if __name__ == "__main__":
    main()
