"""Shared pytest config: gate `isaac`-marked tests on the isaac env.

The brain-side suite is pure (no isaacsim/limxsdk) and runs in the `brain` env.
World-side tests boot Isaac and must run in the `isaac` env. Rather than mock Isaac
(which would test nothing), we mark those `@pytest.mark.isaac` and skip them whenever
`isaacsim` is not importable — so `conda run -n brain pytest` stays green and fast,
and `conda run -n isaac pytest` exercises the real articulation.
"""

import importlib.util

import pytest


def pytest_collection_modifyitems(config, items):
    if importlib.util.find_spec("isaacsim") is not None:
        return  # isaac env — run everything
    skip_isaac = pytest.mark.skip(reason="needs the `isaac` env (isaacsim not importable)")
    for item in items:
        if "isaac" in item.keywords:
            item.add_marker(skip_isaac)
