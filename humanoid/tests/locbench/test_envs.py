"""TDD for §7.1 env tooling (logic/locbench/envs.py).

`locbench env create|remove <candidate>` builds the disposable `bench-<candidate>` conda
env from a realization's recipe (D8: the WHOLE brain boots inside it), exports `lock.yml`
(committed provenance), and refuses to touch the sacred envs (brain|isaac|limx|hum).

The conda calls are injected as a runner (`run: CondaRunner`) so the decision logic —
name resolution, guard, force-recreate, export path — is pure and runs under `brain` with
no conda installed. The live build is a smoke step Anton runs, not a unit test.
"""

import pytest

from humanoid.logic.locbench.envs import (
    BENCH_PREFIX,
    PROTECTED_ENVS,
    CondaResult,
    EnvError,
    bench_env_name,
    env_create,
    env_exists,
    env_remove,
)

pytestmark = pytest.mark.brain


class FakeConda:
    """Records argv and emulates `conda env list/create/export/remove`."""

    def __init__(self, existing=(), export="name: bench-x\ndependencies:\n  - numpy\n"):
        self.calls = []
        self._existing = set(existing)
        self._export = export

    def _name_after(self, argv, flag="-n"):
        return argv[argv.index(flag) + 1]

    def __call__(self, argv):
        self.calls.append(list(argv))
        head = argv[:3]
        if head == ["conda", "env", "list"]:
            body = "\n".join(f"{e}    /opt/conda/envs/{e}" for e in sorted(self._existing))
            return CondaResult(0, f"# conda environments:\n#\nbase  /opt/conda\n{body}\n", "")
        if head == ["conda", "env", "export"]:
            return CondaResult(0, self._export, "")
        if head == ["conda", "env", "create"]:
            self._existing.add(self._name_after(argv))
            return CondaResult(0, "", "")
        if head == ["conda", "env", "remove"]:
            self._existing.discard(self._name_after(argv))
            return CondaResult(0, "", "")
        return CondaResult(0, "", "")  # conda run bash build.sh

    def argv_starting(self, prefix):
        return [c for c in self.calls if c[: len(prefix)] == prefix]


def _recipe(tmp_path, *, build=False):
    rdir = tmp_path / "realization"
    rdir.mkdir()
    (rdir / "environment.yml").write_text(
        "name: bench-<name>\nchannels:\n  - conda-forge\ndependencies:\n  - python=3.11\n"
    )
    if build:
        (rdir / "build.sh").write_text("#!/usr/bin/env bash\nset -euo pipefail\necho hi\n")
    return rdir


# ── naming + guard ──────────────────────────────────────────────────────────

def test_bench_env_name_prefixes():
    assert bench_env_name("rtabmap") == f"{BENCH_PREFIX}rtabmap"


# ids are prefixed so a bare env name (e.g. "isaac") never becomes a pytest keyword —
# tests/conftest.py skips anything keyworded `isaac`, which would silently drop these.
@pytest.mark.parametrize("bad", sorted(PROTECTED_ENVS), ids=lambda n: f"prot_{n}")
def test_create_refuses_protected_names(tmp_path, bad):
    fake = FakeConda()
    with pytest.raises(EnvError):
        env_create(bad, _recipe(tmp_path), run=fake)
    assert fake.calls == []  # guarded BEFORE any conda touch


@pytest.mark.parametrize("bad", sorted(PROTECTED_ENVS), ids=lambda n: f"prot_{n}")
def test_remove_refuses_protected_names(bad):
    fake = FakeConda(existing=[bad])
    with pytest.raises(EnvError):
        env_remove(bad, run=fake)
    assert fake.calls == []


def test_create_refuses_blank_or_bad_slug(tmp_path):
    rdir = _recipe(tmp_path)
    for bad in ("", "  ", "has space", "bad/slash"):
        with pytest.raises(EnvError):
            env_create(bad, rdir, run=FakeConda())


# ── existence ───────────────────────────────────────────────────────────────

def test_env_exists_parses_list():
    fake = FakeConda(existing=["bench-foo", "brain"])
    assert env_exists("foo", run=fake) is True
    assert env_exists("bar", run=fake) is False


# ── create ──────────────────────────────────────────────────────────────────

def test_create_builds_and_exports_lock(tmp_path):
    fake = FakeConda(export="name: bench-foo\ndependencies:\n  - python=3.11\n")
    rdir = _recipe(tmp_path, build=True)
    lock = env_create("foo", rdir, run=fake)

    # created with explicit -n (overrides recipe's placeholder name) + recipe file
    creates = fake.argv_starting(["conda", "env", "create"])
    assert creates and "-n" in creates[0] and creates[0][creates[0].index("-n") + 1] == "bench-foo"
    assert str(rdir / "environment.yml") in creates[0]

    # build.sh executed inside the env
    assert fake.argv_starting(["conda", "run", "-n", "bench-foo"])

    # lock.yml written from the export stdout, in the realization dir
    assert lock == rdir / "lock.yml"
    assert lock.read_text() == "name: bench-foo\ndependencies:\n  - python=3.11\n"


def test_create_skips_build_when_no_build_sh(tmp_path):
    fake = FakeConda()
    env_create("foo", _recipe(tmp_path, build=False), run=fake)
    assert fake.argv_starting(["conda", "run"]) == []


def test_create_refuses_missing_recipe(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(EnvError, match="environment.yml"):
        env_create("foo", empty, run=FakeConda())


def test_create_refuses_existing_env_without_force(tmp_path):
    fake = FakeConda(existing=["bench-foo"])
    with pytest.raises(EnvError, match="exists"):
        env_create("foo", _recipe(tmp_path), run=fake)


def test_create_force_recreates(tmp_path):
    fake = FakeConda(existing=["bench-foo"])
    env_create("foo", _recipe(tmp_path), run=fake, force=True)
    assert fake.argv_starting(["conda", "env", "remove"])  # removed first
    assert fake.argv_starting(["conda", "env", "create"])  # then recreated


# ── remove ──────────────────────────────────────────────────────────────────

def test_remove_calls_conda_remove():
    fake = FakeConda(existing=["bench-foo"])
    env_remove("foo", run=fake)
    removes = fake.argv_starting(["conda", "env", "remove"])
    assert removes and removes[0][removes[0].index("-n") + 1] == "bench-foo"
    assert "-y" in removes[0]


def test_remove_noop_when_absent():
    fake = FakeConda(existing=[])
    env_remove("foo", run=fake)  # must not raise
