#!/usr/bin/env python3
"""Eval session inspection and comparison utilities.

Usage:
    python -m vbti.logic.inference.eval_helpers ls
    python -m vbti.logic.inference.eval_helpers ls --detailed
    python -m vbti.logic.inference.eval_helpers info latest
    python -m vbti.logic.inference.eval_helpers info v18-20k
    python -m vbti.logic.inference.eval_helpers info latest --group_by=scene
    python -m vbti.logic.inference.eval_helpers play latest 3
"""
import json
import re
import subprocess
from collections import defaultdict
from pathlib import Path

from vbti.logic.train.experiment_utils import (
    _resolve_experiment,
    _resolve_version,
    _version_dir,
)


PROTO_DIR = Path(__file__).parent / "protocols"


# ── Path utilities ────────────────────────────────────────────────────────────

def _eval_dir(experiment: str, version: str) -> Path:
    return _version_dir(experiment, version) / "eval_sessions"


def _list_sessions(experiment: str, version: str) -> list[Path]:
    d = _eval_dir(experiment, version)
    if not d.exists():
        return []
    return sorted([p for p in d.iterdir() if p.is_dir() and (p / "session.json").exists()],
                  key=lambda p: p.name)


def _parse_step_token(s: str) -> int:
    """`30k` → 30000, `30000` → 30000, `step_030000` → 30000."""
    s = s.lower().strip()
    m = re.fullmatch(r"step_0*(\d+)", s)
    if m:
        return int(m.group(1))
    if s.endswith("k"):
        return int(s[:-1]) * 1000
    return int(s)


def _step_in_session_id(sid: str) -> int | None:
    """Extract checkpoint step from session id like `chkpt_step_020000_...`."""
    m = re.search(r"chkpt_step_0*(\d+)_", sid)
    if m:
        return int(m.group(1))
    m = re.search(r"chkpt_(?:step_)?0*(\d+)", sid)
    return int(m.group(1)) if m else None


def resolve_session(spec: str,
                    experiment: str | None = None,
                    version: str | None = None) -> Path:
    """Resolve a session spec to its directory.

    Accepted forms:
        latest                                       → most recent in active version
        v13-20k / v013-20000 / v13-step_020000       → latest session for that ckpt
        20k / 30000 / step_030000                    → latest in active version for that ckpt
        chkpt_step_..._YYYYMMDD_HHMMSS               → exact id match
        /absolute/path/to/session_dir                → used directly
    """
    # Direct path
    p = Path(spec).expanduser()
    if p.exists() and (p / "session.json").exists():
        return p

    experiment = _resolve_experiment(experiment)

    # vXX-... shorthand
    m = re.fullmatch(r"v(\d+)(?:-(\S+))?", spec, re.IGNORECASE)
    if m:
        ver_str  = f"v{int(m.group(1)):03d}"
        sessions = _list_sessions(experiment, ver_str)
        if not sessions:
            raise FileNotFoundError(f"No eval sessions in {experiment}/{ver_str}")
        if m.group(2) is None or m.group(2).lower() == "latest":
            return sessions[-1]
        step  = _parse_step_token(m.group(2))
        match = [s for s in sessions if _step_in_session_id(s.name) == step]
        if not match:
            raise FileNotFoundError(
                f"No session for step {step} in {experiment}/{ver_str}")
        return match[-1]

    # `latest` in active version
    version  = _resolve_version(experiment, version)
    sessions = _list_sessions(experiment, version)
    if not sessions:
        raise FileNotFoundError(f"No eval sessions in {experiment}/{version}")
    if spec.lower() == "latest":
        return sessions[-1]

    # Step shorthand in active version
    if re.fullmatch(r"\d+k?|step_\d+", spec, re.IGNORECASE):
        step  = _parse_step_token(spec)
        match = [s for s in sessions if _step_in_session_id(s.name) == step]
        if not match:
            raise FileNotFoundError(f"No session for step {step} in {experiment}/{version}")
        return match[-1]

    # Exact id (or prefix) in active version
    match = [s for s in sessions if s.name == spec or s.name.startswith(spec)]
    if len(match) == 1:
        return match[0]
    if len(match) > 1:
        raise ValueError(f"Ambiguous session spec '{spec}'; matches: "
                         + ", ".join(p.name for p in match))
    raise FileNotFoundError(f"Could not resolve session: {spec}")


# ── Data loading ──────────────────────────────────────────────────────────────

def load_session(spec: str,
                 experiment: str | None = None,
                 version: str | None = None) -> dict:
    """Load session.json from disk."""
    sdir = resolve_session(spec, experiment, version)
    data = json.loads((sdir / "session.json").read_text())
    data["_dir"] = str(sdir)
    return data


# ── Statistics ────────────────────────────────────────────────────────────────

def wilson_ci(s: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson 95% CI for a Bernoulli SR — well-behaved at small n."""
    if n == 0:
        return 0.0, 0.0
    p = s / n
    denom  = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half   = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def fmt_sr(s: int, n: int) -> str:
    """`14/20 (70%) [46–87]` — printable success rate with CI."""
    if n == 0:
        return "0/0 (—)"
    lo, hi = wilson_ci(s, n)
    return f"{s}/{n} ({s/n:.0%}) [{lo*100:.0f}–{hi*100:.0f}]"


def _success_count(trials: list[dict]) -> int:
    return sum(1 for t in trials if t["result"] == "success")


def _bucket(trials: list[dict], key) -> dict:
    """Group trials by `key` (callable trial→hashable). Returns {bucket: [trials]}."""
    out: dict = defaultdict(list)
    for t in trials:
        out[key(t)].append(t)
    return dict(out)


# ── ls ────────────────────────────────────────────────────────────────────────

def ls(experiment: str | None = None,
       version: str | None = None,
       detailed: bool = False):
    """List eval sessions for an experiment/version.

    Compact: id (truncated), checkpoint, protocol, SR, n, date.
    Detailed (--detailed): adds config, per-zone SR, n notes, n videos.
    """
    experiment = _resolve_experiment(experiment)
    version    = _resolve_version(experiment, version)
    sessions   = _list_sessions(experiment, version)

    print(f"\n{experiment}/{version} — {len(sessions)} eval sessions\n")
    if not sessions:
        return

    for sdir in sessions:
        data    = json.loads((sdir / "session.json").read_text())
        trials  = data.get("trials", [])
        cfg     = data.get("config", {})
        n       = len(trials)
        s       = _success_count(trials)
        sr      = fmt_sr(s, n)
        ckpt    = cfg.get("checkpoint_label", "?")
        proto   = data.get("protocol", "?")
        date    = data.get("date", "?")[:16].replace("T", " ")

        if not detailed:
            print(f"  {ckpt:>14}  {proto:<28}  SR {sr:<22}  {date}  ({sdir.name[-15:]})")
        else:
            id_t   = [t for t in trials if t["zone"] == "ID"]
            ood_t  = [t for t in trials if t["zone"] == "OOD"]
            n_note = sum(1 for t in trials if t.get("note"))
            n_vid  = sum(1 for t in trials if t.get("video"))
            print(f"━━ {sdir.name}")
            print(f"   ckpt:   {ckpt}     protocol: {proto}     date: {date}")
            print(f"   SR:     {sr}")
            if id_t:
                print(f"   ID:     {fmt_sr(_success_count(id_t),  len(id_t))}")
            if ood_t:
                print(f"   OOD:    {fmt_sr(_success_count(ood_t), len(ood_t))}")
            print(f"   ah={cfg.get('action_horizon')}  "
                  f"detection={cfg.get('detection')}  "
                  f"delta={cfg.get('delta_actions')}  "
                  f"rtc={cfg.get('enable_rtc')}  "
                  f"record={cfg.get('record', False)}")
            print(f"   notes:  {n_note}/{n}      videos: {n_vid}/{n}")
            print()


# ── info ──────────────────────────────────────────────────────────────────────

def _tag_keys(trials: list[dict]) -> list[str]:
    """Tag keys present in at least one trial — used for auto-breakdowns."""
    keys: set[str] = set()
    for t in trials:
        keys.update((t.get("tags") or {}).keys())
    return sorted(keys)


def info(session: str,
         experiment: str | None = None,
         version: str | None = None,
         group_by: str | None = None):
    """Detailed session breakdown — text-only, designed for LLM consumption.

    Sections: header, overall, by zone, per tag (auto-detected), steps,
    failures, notes.

    --group_by <key>   add an explicit breakdown by `zone` or any tag key
                       (e.g. scene, target_color, target_side).
    """
    data    = load_session(session, experiment, version)
    sdir    = Path(data["_dir"])
    trials  = data["trials"]
    cfg     = data.get("config", {})
    n       = len(trials)
    s       = _success_count(trials)

    print(f"\n━━━ {sdir.name} ━━━")
    print(f"checkpoint:  {cfg.get('checkpoint_label', '?')}")
    print(f"protocol:    {data.get('protocol', '?')}")
    print(f"date:        {data.get('date', '?')}")
    print(f"config:      ah={cfg.get('action_horizon')}  "
          f"delta={cfg.get('delta_actions')}  "
          f"rtc={cfg.get('enable_rtc')}  "
          f"record={cfg.get('record', False)}\n")

    if n == 0:
        print("No trials.")
        return

    print(f"OVERALL:  {fmt_sr(s, n)}\n")

    # Per zone
    print("BY ZONE:")
    for z in ("ID", "OOD"):
        zt = [t for t in trials if t["zone"] == z]
        if zt:
            print(f"  {z:<3}  {fmt_sr(_success_count(zt), len(zt))}")
    print()

    # Per tag (auto-detected from trial.tags)
    for tag in _tag_keys(trials):
        bucket = _bucket(trials, lambda t, k=tag: (t.get("tags") or {}).get(k, "—"))
        if len(bucket) <= 1:
            continue
        print(f"BY {tag.upper()}:")
        for k in sorted(bucket, key=str):
            grp = bucket[k]
            print(f"  {str(k):<14} {fmt_sr(_success_count(grp), len(grp))}")
        print()

    # Steps stats
    succ = [t["steps"] for t in trials if t["result"] == "success"]
    fail = [t["steps"] for t in trials if t["result"] == "failure"]
    print("STEPS:")
    if succ:
        print(f"  success  n={len(succ)}  "
              f"mean={sum(succ)/len(succ):.0f}  "
              f"min={min(succ)}  max={max(succ)}")
    if fail:
        print(f"  failure  n={len(fail)}  "
              f"mean={sum(fail)/len(fail):.0f}  "
              f"min={min(fail)}  max={max(fail)}")
    print()

    # Custom group_by — zone or any tag key
    if group_by:
        print(f"BY {group_by.upper()}:")
        if group_by == "zone":
            keyfn = lambda t: t["zone"]
        elif group_by in _tag_keys(trials):
            keyfn = lambda t, k=group_by: (t.get("tags") or {}).get(k, "—")
        else:
            available = ["zone"] + _tag_keys(trials)
            print(f"  unknown group_by: {group_by}  (available: {available})")
            keyfn = None
        if keyfn:
            for k, grp in sorted(_bucket(trials, keyfn).items(), key=lambda x: str(x[0])):
                print(f"  {str(k):<14} {fmt_sr(_success_count(grp), len(grp))}")
            print()

    # Failures
    failures = [t for t in trials if t["result"] == "failure"]
    if failures:
        print(f"FAILURES ({len(failures)}):")
        for t in failures:
            note   = f"  — {t['note']}" if t.get("note") else ""
            vid    = f"  [video]"        if t.get("video") else ""
            tags   = t.get("tags") or {}
            tagstr = " ".join(f"{k}={v}" for k, v in tags.items()) if tags else ""
            target = f" target={t['target']}" if "target" in t else ""
            print(f"  T{t['trial_id']:02d} [{t['zone']}]{target}  "
                  f"steps:{t['steps']}  {tagstr}{vid}{note}")
        print()

    # Notes
    noted = [t for t in trials if t.get("note")]
    if noted:
        print(f"NOTES ({len(noted)}):")
        for t in noted:
            print(f"  T{t['trial_id']:02d} ({t['result']}): {t['note']}")
        print()



# ── play ──────────────────────────────────────────────────────────────────────

def play(session: str,
         trial_id: int,
         experiment: str | None = None,
         version: str | None = None):
    """Open the recorded video for a trial in the system video player."""
    data   = load_session(session, experiment, version)
    sdir   = Path(data["_dir"])

    candidates = sorted((sdir / "videos").glob(f"trial_{int(trial_id):02d}_*.mp4"))
    if not candidates:
        print(f"No video for trial {trial_id} in {sdir.name}")
        # show trials that *do* have video
        with_v = [t for t in data["trials"] if t.get("video")]
        if with_v:
            ids = sorted({t["trial_id"] for t in with_v})
            print(f"Available trial videos: {ids}")
        else:
            print("This session has no recorded videos. "
                  "Re-run eval_engine with --record=True next time.")
        return

    path = candidates[0]
    print(f"Opening: {path}")
    subprocess.Popen(["xdg-open", str(path)],
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


# ── Fire CLI ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import fire
    fire.core.Display = lambda lines, out: print(*lines, file=out)
    fire.Fire({
        "ls":         ls,
        "info":       info,
        "play":       play,
        # exposed for the copilot to import
        "load":       load_session,
        "resolve":    lambda spec, **kw: print(resolve_session(spec, **kw)),
    })
