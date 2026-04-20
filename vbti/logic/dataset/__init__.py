"""Dataset utilities — check, convert, load."""
import shutil
from pathlib import Path


LEROBOT_CACHE = Path.home() / ".cache" / "huggingface" / "lerobot"


def delete_dataset(dataset_path: str, root: str = None, force: bool = False) -> bool:
    """Delete a cached LeRobot dataset.

    Args:
        dataset_path: Repo ID or path (e.g. "eternalmay33/02_det_only")
        root: Override root path.
        force: Skip confirmation prompt.

    Returns:
        True if deleted, False if cancelled.
    """
    ds_path = resolve_dataset_path(dataset_path, root=root)

    if not ds_path.exists():
        print(f"Not found: {ds_path}")
        return False

    # Show what will be deleted
    total_size = sum(f.stat().st_size for f in ds_path.rglob("*") if f.is_file() and not f.is_symlink())
    symlinks = sum(1 for f in ds_path.rglob("*") if f.is_symlink())
    print(f"Dataset:  {dataset_path}")
    print(f"Path:     {ds_path}")
    print(f"Size:     {total_size / 1e6:.1f} MB ({symlinks} symlinks)")

    if not force:
        confirm = input("Delete? [y/N] ").strip().lower()
        if confirm != "y":
            print("Cancelled.")
            return False

    shutil.rmtree(ds_path)
    print(f"Deleted: {ds_path}")
    return True


def resolve_dataset_path(dataset_path: str, root: str = None) -> Path:
    """Resolve a dataset path, repo_id, or dataset name to an absolute Path.

    Resolution order:
        1. If `root` is given, use it directly.
        2. If `dataset_path` is an existing filesystem path, use it.
        3. Try as repo_id in ~/.cache/huggingface/lerobot/ (e.g. "enea-c/VBTI-Align-v2A").
        4. Raise with suggestions.
    """
    if root:
        return Path(root).expanduser().resolve()
    p = Path(dataset_path).expanduser()
    if p.exists():
        return p.resolve()
    cache = LEROBOT_CACHE / dataset_path
    if cache.exists():
        return cache.resolve()
    # Suggest similar
    available = []
    if LEROBOT_CACHE.exists():
        for author_dir in LEROBOT_CACHE.iterdir():
            if author_dir.is_dir() and not author_dir.name.startswith("."):
                for ds_dir in author_dir.iterdir():
                    if ds_dir.is_dir():
                        available.append(f"{author_dir.name}/{ds_dir.name}")
    needle = dataset_path.split("/")[-1][:5]
    suggestions = [a for a in available if needle in a]
    hint = f"\n  Did you mean: {', '.join(suggestions)}" if suggestions else f"\n  Available: {', '.join(available)}"
    raise FileNotFoundError(f"Dataset not found: {dataset_path}{hint}")


def __getattr__(name):
    """Lazy imports to avoid circular import with HuggingFace datasets."""
    if name in ("load_and_split_dataset", "create_dataloaders", "aggregate_sources"):
        from .loading_utils import load_and_split_dataset, create_dataloaders, aggregate_sources
        return {"load_and_split_dataset": load_and_split_dataset,
                "create_dataloaders": create_dataloaders,
                "aggregate_sources": aggregate_sources}[name]
    if name in ("convert", "sim_to_normalized", "normalized_to_sim",
                "discover_cameras", "build_features"):
        from . import convert_utils
        return getattr(convert_utils, name)
    if name in ("report_lerobot", "report_hdf5", "compare_actions"):
        from . import check_utils
        return getattr(check_utils, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "load_and_split_dataset",
    "create_dataloaders",
    "convert",
    "sim_to_normalized",
    "normalized_to_sim",
    "discover_cameras",
    "build_features",
    "report_lerobot",
    "report_hdf5",
    "compare_actions",
]
