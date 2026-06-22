from pathlib import Path

from vbti.logic.reconstruct import gs_milo_utils


def test_milo_dir_resolves_from_package_root():
    assert gs_milo_utils.MILO_DIR == Path(gs_milo_utils.__file__).resolve().parents[2] / "libs" / "MILo" / "milo"


def test_default_config_includes_resolution_limit():
    assert gs_milo_utils.DEFAULT_CONFIG["resolution"] == 4


if __name__ == "__main__":
    test_milo_dir_resolves_from_package_root()
    test_default_config_includes_resolution_limit()
