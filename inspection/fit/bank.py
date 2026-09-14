"""The bank: name -> fitter. Adding a method is registering one function.

`bank.fit(points, methods=[...])` is the primary entry point — the one the
real loop will call. The CLI is a wrapper over exactly this call, so what
research measures and what the loop runs cannot drift apart.

Scoring happens here, once, for every method, using `scores.score`. Fitters
never grade themselves; a fitter's own notion of an inlier is replaced by
the uniform one before the `Decomposition` is handed back. That is what
makes `coverage` comparable between a plane decomposition and a
superquadric.
"""
from __future__ import annotations

import time
from typing import Callable, Dict, Iterable, List, Optional

import numpy as np

from inspection.fit import cyl as _cyl
from inspection.fit import ems as _ems
from inspection.fit import planar as _planar
from inspection.fit.contract import Decomposition
from inspection.fit.scores import score

Fitter = Callable[..., Decomposition]

_REGISTRY: Dict[str, Fitter] = {}


def register(name: str, fn: Fitter) -> None:
    _REGISTRY[name] = fn


def methods() -> List[str]:
    return sorted(_REGISTRY)


def get(name: str) -> Fitter:
    if name not in _REGISTRY:
        raise KeyError(f"unknown method {name!r}; have {methods()}")
    return _REGISTRY[name]


register(_planar.NAME, _planar.fit)
register(_cyl.NAME, _cyl.fit)
register(_ems.NAME, _ems.fit)

DEFAULT_METHODS = ("planar", "cyl", "ems")


def fit(points: np.ndarray, methods: Iterable[str] = DEFAULT_METHODS,
        *, seed: int = 0, table_normal=None,
        warm_start: Optional[Dict[str, Decomposition]] = None,
        **kw) -> Dict[str, Decomposition]:
    """Run each named fitter on `points` and return `{name: Decomposition}`.

    A fitter that raises is not allowed to take the others down with it: the
    exception becomes an abstain naming the exception, which is a truthful
    row in the score log rather than a hole in it.
    """
    P = np.asarray(points, dtype=float)
    out: Dict[str, Decomposition] = {}
    warm_start = warm_start or {}
    for name in methods:
        fn = get(name)
        try:
            dec = fn(P, seed=seed, table_normal=table_normal,
                     warm_start=warm_start.get(name), **kw)
        except Exception as exc:                     # noqa: BLE001
            from inspection.fit.contract import abstained
            dec = abstained(name, f"{name}/unknown",
                            f"fitter raised {type(exc).__name__}: {exc}")
        out[name] = score(P, dec)
    return out


def fit_timed(points: np.ndarray, methods: Iterable[str] = DEFAULT_METHODS,
              **kw):
    """`fit`, plus wall-clock seconds per method — for the score log."""
    P = np.asarray(points, dtype=float)
    decs, secs = {}, {}
    for name in methods:
        t0 = time.perf_counter()
        decs.update(fit(P, [name], **kw))
        secs[name] = round(time.perf_counter() - t0, 3)
    return decs, secs
