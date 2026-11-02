"""§7.1 — disposable `bench-<candidate>` conda envs for the localization loop (D8).

One process ⇒ one env: a bench run boots the WHOLE brain inside the candidate's own
env (`conda run -n bench-<name> …`), so the brain's stdlib/numpy base and the candidate's
stack (SLAM libs, torch, …) live together and never contaminate the sacred envs.

`env_create` solves the realization's `environment.yml`, runs its `build.sh` inside the
new env, then freezes the exact solve to `lock.yml` (committed → "what was built" stays
answerable in `report.json` provenance). `env_remove` leaves no trace. A hard guard
refuses to ever name `brain|isaac|limx|hum`.

Conda is reached through an injected `run(argv) -> CondaResult` so the guard / naming /
force-recreate / export logic is pure and unit-tested with no conda installed. No Docker.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Callable, NamedTuple

BENCH_PREFIX = "bench-"
PROTECTED_ENVS = frozenset({"brain", "isaac", "limx", "hum"})
_SLUG_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


class EnvError(RuntimeError):
    """A bench-env operation was refused or a conda command failed."""


class CondaResult(NamedTuple):
    returncode: int
    stdout: str
    stderr: str


CondaRunner = Callable[[list], CondaResult]


def default_conda_runner(argv: list) -> CondaResult:
    """Real conda: capture stdout/stderr, never raise here (callers check returncode)."""
    p = subprocess.run(argv, capture_output=True, text=True)
    return CondaResult(p.returncode, p.stdout, p.stderr)


# ── naming + guard ──────────────────────────────────────────────────────────


def bench_env_name(candidate: str) -> str:
    """`<candidate>` → `bench-<candidate>` (the only env name this tool ever touches)."""
    return f"{BENCH_PREFIX}{candidate}"


def _guard(candidate: str) -> str:
    """Validate the candidate slug and refuse the sacred envs. Returns the bench env name."""
    name = (candidate or "").strip()
    if not _SLUG_RE.match(name):
        raise EnvError(f"invalid candidate name {candidate!r} (need [A-Za-z0-9][A-Za-z0-9_-]*)")
    if name in PROTECTED_ENVS:
        raise EnvError(f"refusing to touch the protected env {name!r} (brain|isaac|limx|hum)")
    env = bench_env_name(name)
    if env in PROTECTED_ENVS:  # defensive — impossible while BENCH_PREFIX is set
        raise EnvError(f"refusing to touch the protected env {env!r}")
    return env


def _ok(res: CondaResult, what: str) -> CondaResult:
    if res.returncode != 0:
        raise EnvError(f"{what} failed (exit {res.returncode}): {res.stderr.strip() or res.stdout.strip()}")
    return res


# ── existence ───────────────────────────────────────────────────────────────


def env_exists(candidate: str, *, run: CondaRunner = default_conda_runner) -> bool:
    env = _guard(candidate)
    out = _ok(run(["conda", "env", "list"]), "conda env list").stdout
    for line in out.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.split()[0] == env:  # first column is the env name
            return True
    return False


# ── create / remove ─────────────────────────────────────────────────────────


def env_create(
    candidate: str,
    realization_dir,
    *,
    run: CondaRunner = default_conda_runner,
    force: bool = False,
    log: Callable[[str], None] = print,
) -> Path:
    """Build `bench-<candidate>` from the realization recipe; return the written lock.yml path."""
    env = _guard(candidate)
    rdir = Path(realization_dir)
    recipe = rdir / "environment.yml"
    if not recipe.is_file():
        raise EnvError(f"no environment.yml in {rdir} — scaffold the realization first (loc-new)")

    if env_exists(candidate, run=run):
        if not force:
            raise EnvError(f"env {env!r} already exists — pass force=True to recreate")
        log(f"[env] {env} exists → removing (force)")
        _ok(run(["conda", "env", "remove", "-n", env, "-y"]), f"remove {env}")

    log(f"[env] creating {env} from {recipe}")
    _ok(run(["conda", "env", "create", "-n", env, "-f", str(recipe)]), f"create {env}")

    build = rdir / "build.sh"
    if build.is_file():
        log(f"[env] running {build.name} inside {env}")
        _ok(run(["conda", "run", "-n", env, "bash", str(build)]), f"build.sh in {env}")

    lock = rdir / "lock.yml"
    export = _ok(run(["conda", "env", "export", "-n", env]), f"export {env}").stdout
    lock.write_text(export)
    log(f"[env] froze solve → {lock}")
    return lock


def env_remove(
    candidate: str,
    *,
    run: CondaRunner = default_conda_runner,
    log: Callable[[str], None] = print,
) -> None:
    """Delete `bench-<candidate>` (no-op if absent). Leaves the committed lock.yml in place."""
    env = _guard(candidate)
    if not env_exists(candidate, run=run):
        log(f"[env] {env} not present — nothing to remove")
        return
    log(f"[env] removing {env}")
    _ok(run(["conda", "env", "remove", "-n", env, "-y"]), f"remove {env}")
