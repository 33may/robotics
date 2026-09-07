#!/usr/bin/env python3
"""Open review workspaces (.rrd) in the native Rerun viewer.

The workspaces are built by `workspace.py` into the durable derived tree,
`inspection/data/derived/<run>/rrd/<run>.rrd` — gitignored, regeneratable,
sitting next to the step clouds and fit JSONs it was built from. This is
just the front door:

    p inspection/record/rr/view.py list
    p inspection/record/rr/view.py show 2408-cup3

/tmp/rrd is still searched, second, so workspaces built before the move
(and any scratch file a test rig drops there) still open by name.
"""
import subprocess
import sys
import time
from pathlib import Path

DERIVED = Path(__file__).resolve().parents[2] / "data" / "derived"
#: Where workspaces lived until 2026-09-03. Kept as a fallback, never as a
#: destination.
RRD_TMP = Path("/tmp/rrd")


def _viewer() -> str:
    """The rerun binary that ships with rerun-sdk in the active env."""
    cand = Path(sys.executable).parent / "rerun"
    if cand.exists():
        return str(cand)
    import shutil
    found = shutil.which("rerun")
    if not found:
        raise SystemExit("no rerun viewer on PATH — pip install rerun-sdk")
    return found


def _hypr() -> bool:
    """Are we on a live Hyprland session that can be told how to place it?"""
    import os
    import shutil
    return bool(os.environ.get("HYPRLAND_INSTANCE_SIGNATURE")
                and shutil.which("hyprctl"))


def _episodes() -> dict[str, Path]:
    """{name: path}, derived tree first, /tmp second — first writer wins.

    Keyed by name rather than returned as a list so a stale /tmp copy of a
    run that has since been rebuilt into the derived tree cannot shadow it;
    `show 2408-cup3` must mean the current workspace, not the older one
    that happens to sort first.
    """
    found: dict[str, Path] = {}
    for p in sorted(DERIVED.glob("*/rrd/*.rrd")) if DERIVED.is_dir() else []:
        found.setdefault(p.stem, p)
    for p in sorted(RRD_TMP.glob("*.rrd")) if RRD_TMP.is_dir() else []:
        # underscore files are scratch from the build/test loops
        if not p.name.startswith("_"):
            found.setdefault(p.stem, p)
    return found


def list() -> None:  # noqa: A001 - fire command name
    eps = _episodes()
    if not eps:
        print(f"no workspaces in {DERIVED}/*/rrd or {RRD_TMP} — "
              f"run a producer first")
        return
    now = time.time()
    for name, p in eps.items():
        st = p.stat()
        age_min = (now - st.st_mtime) / 60
        age = f"{age_min:.0f} min" if age_min < 90 else f"{age_min / 60:.1f} h"
        where = "derived" if p.is_relative_to(DERIVED) else "tmp"
        print(f"{name:20s} {st.st_size / 1e6:7.1f} MB   built {age} ago   "
              f"[{where}] {p}")


def show(episode: str) -> None:
    eps = _episodes()
    name = str(episode).removesuffix(".rrd")
    path = eps.get(name)
    if path is None:
        names = ", ".join(eps) or "none"
        raise SystemExit(f"no workspace named {name!r} — have: {names}")
    print(f"opening {path}")
    if _hypr():
        # TILED, explicitly (Anton 2026-09-03). The viewer was coming up
        # floating and needing a MainMod+O before it was usable — a review
        # tool that costs a keystroke before you can review is the thing
        # this workspace exists to avoid. Hyprland's exec rule sets the
        # state at map time, so there is no float-then-snap flicker either.
        # `hyprctl dispatch exec` also detaches for us.
        subprocess.run(["hyprctl", "dispatch", "exec",
                        f"[tile] {_viewer()} {path}"],
                       stdout=subprocess.DEVNULL, check=False)
        return
    # Detached on purpose: the CLI returns, the viewer window stays yours.
    subprocess.Popen([_viewer(), str(path)], start_new_session=True,
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


if __name__ == "__main__":
    import fire
    fire.core.Display = lambda lines, out: print(*lines, file=out)
    fire.Fire({"list": list, "show": show})
