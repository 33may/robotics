"""Dataset utilities — check, convert, load."""


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
