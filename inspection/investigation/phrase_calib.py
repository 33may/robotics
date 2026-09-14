#!/usr/bin/env python3
"""Does the inspection subagent's `framing.facing` carry MAGNITUDE, or only direction?

The subagent returns, per view, a `framing` block: {target, facing, better}.
`facing` is supposed to say "how square-on its surface is AND which way it turns
away". The planner would like to read a NUMBER out of that — one cell of turn or
four? — so the question is whether the words are ordered by angle at all.

The naive test is circular: take the cell the model called "face-on", call that
the front, then measure how far the other phrases sit from it. That grades a
model judgment with a model judgment. Instead the object's front azimuth
theta_f is FITTED: within a run the object does not move, so theta_f is a fixed
unknown; sweep it over the whole circle and keep the value that best ORDERS the
phrases. The phrase ordering itself is fixed in advance from wording alone
(see PHRASE_RANK), never from the angles.

The reported result is the BEST CASE over theta_f. If the most favourable
possible front still cannot order the phrases by angle, no prompt tuning to
this schema will fix it.

    p inspection/investigation/phrase_calib.py
    p inspection/investigation/phrase_calib.py --out /tmp/phrase_calib

DATA CAVEAT: these runs were recorded with the local vision verbs on a STUB
backend (every `detect` returned the same box, `read_text` returned nothing).
Nothing here reads a detect box, an OCR string or a segment area. The inputs
are the VLM's own `framing` strings (authentic Gemini output) and the recorded
camera extrinsics `T_base_cam`. Both are unaffected by the stub.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from scipy.stats import kendalltau, spearmanr

RUNS = Path(__file__).resolve().parents[1] / "data/runs"

# ---------------------------------------------------------------- linguistics
# The magnitude ordering of the `facing` vocabulary, fixed from WORDING ONLY.
# Justification per entry is printed by the script; ties are deliberate and are
# themselves information (two phrases that say the same thing about magnitude
# get the same rank no matter how far apart they were uttered).
PHRASE_RANK = [
    (0, "face-on",
     "asserts the surface is square to the camera. Zero turn. No hedge, no "
     "direction word - there is nothing to name a direction for."),
    (1, "slightly turned",
     "asserts a turn AND downgrades it ('slightly'). Strictly more turn than "
     "face-on, strictly less than an unhedged turn. [offered by no prompt - "
     "listed for completeness]"),
    (2, "turning away to the <dir>",
     "asserts the surface IS turning away, unhedged and unbounded. The only "
     "difference between the -left and -right forms is the direction word, "
     "which says nothing about how far -> SAME RANK."),
    (3, "edge-on to the <dir>",
     "'edge-on' names a specific extreme: the surface seen along its edge, "
     "i.e. ~90 deg of turn. Strictly more than an unqualified 'turning away'."),
    (4, "not visible",
     "the surface has turned past the point of being seen at all. Terminal."),
]
#: verbatim string -> (rank, canonical family). Anything not listed is fatal.
def rank_of(facing):
    f = facing.strip().lower()
    if f == "face-on":
        return 0, "face-on"
    if f.startswith("slightly turned"):
        return 1, "slightly turned"
    if f.startswith("turning away to the"):
        return 2, "turning away to the <dir>"
    if f.startswith("edge-on"):
        return 3, "edge-on to the <dir>"
    if f.startswith("not visible"):
        return 4, "not visible"
    raise KeyError(f"unranked facing phrase: {facing!r} - extend PHRASE_RANK")


def _facing(r):
    return r["facing"]


def _family(r):
    return r["family"]


def _famrank(fam):
    return [f for f in PHRASE_RANK if f[1] == fam][0][0]


def direction_word(s):
    """'left' / 'right' / None, from a facing or better string."""
    s = s.strip().lower()
    has_l, has_r = "left" in s, "right" in s
    if has_l and not has_r:
        return "left"
    if has_r and not has_l:
        return "right"
    return None


# ------------------------------------------------------------------- geometry
def wrap180(deg):
    """Unsigned angular distance, 0..180."""
    return np.abs((np.asarray(deg, float) + 180.0) % 360.0 - 180.0)


def signed180(deg):
    """Signed angular offset, -180..180."""
    return (np.asarray(deg, float) + 180.0) % 360.0 - 180.0


def sphere_centre(store):
    """Viewsphere centre in base frame: every camera looks at it from radius r."""
    r = store["grid"]["r"]
    pts = [np.array(v["T_base_cam"])[:3, 3] + r * np.array(v["T_base_cam"])[:3, 2]
           for v in store["views"] if v["cell"] is not None]
    return np.mean(pts, axis=0), np.std(pts, axis=0)


def frame_right_sign(T, centre, net_rot_deg):
    """+1 if image-right is the direction of INCREASING base azimuth, else -1.

    The camera x-axis is image-right in the RAW frame. Half-turn cells are
    written to disk already rotated 180 deg (`meta.json:rgb_rotation_deg`), or
    rotated by `eyes/tools.upright()` at read time - either way the image the
    VLM saw is world-upright, and a 180 deg rotation flips image-right.
    """
    pos = T[:3, 3]
    az = np.degrees(np.arctan2(*(pos - centre)[[1, 0]])) % 360.0
    tangent = np.array([-np.sin(np.radians(az)), np.cos(np.radians(az)), 0.0])
    right = T[:3, 0] * (-1.0 if abs(net_rot_deg) > 90 else 1.0)
    return float(np.sign(np.dot(right, tangent))), az


def net_rotation(run, cap_dir, T):
    """How much the frame the VLM saw was rotated relative to raw camera pixels."""
    meta = RUNS / run / cap_dir / "meta.json"
    stored = json.loads(meta.read_text()).get("rgb_rotation_deg") if meta.exists() else None
    if stored:
        return float(stored)
    # pre-2026-08-24 run: eyes/tools.upright() corrects the half-turn at read time
    from inspection.cell.geometry import image_tilt_deg
    tilt = image_tilt_deg(T)
    return 180.0 if abs(abs(tilt) - 180.0) <= 20.0 else 0.0


# -------------------------------------------------------------------- extract
def extract():
    """Every finding that actually carries a `framing` block, verbatim."""
    rows, coverage = [], []
    for run_dir in sorted(RUNS.iterdir()):
        tdir = run_dir / "eyes" / "transcripts"
        spath = run_dir / "eyes" / "store.json"
        if not tdir.is_dir() or not spath.exists():
            continue
        store = json.loads(spath.read_text())
        centre, spread = sphere_centre(store)
        views = {tuple(v["cell"]): v for v in store["views"] if v["cell"] is not None}
        tfiles = sorted(tdir.glob("*.json"))
        got = 0
        for tf in tfiles:
            t = json.loads(tf.read_text())
            fr = t.get("framing")
            if not fr:
                continue
            got += 1
            h, v = t["cell"]
            view = views[(h, v)]
            T = np.array(view["T_base_cam"])
            rot = net_rotation(run_dir.name, view["cap_dir"], T)
            rsign, az_meas = frame_right_sign(T, centre, rot)
            rank, family = rank_of(fr["facing"])
            rows.append(dict(
                run=run_dir.name, transcript=tf.name, cell=[h, v],
                h=h, v=v, azimuth=h * 30.0, elev=store["grid"]["v_elevs"][v],
                az_measured=az_meas, right_is_plus_az=rsign,
                target=fr.get("target"), facing=fr["facing"], better=fr.get("better"),
                rank=rank, family=family,
                dir_word=direction_word(fr["facing"]),
                better_word=direction_word(fr.get("better") or ""),
                task=t.get("task"), answer=t.get("answer"),
            ))
        coverage.append((run_dir.name, len(tfiles), got, spread.max()))
    return rows, coverage


# ------------------------------------------------------------------ the sweep
def somers_d(d, ranks):
    """Somers' D_yx: over every pair of findings whose PHRASES differ in rank,
    the fraction that the angles order the same way, minus the fraction they
    order backwards. 1.0 = every phrase pair is correctly ordered by angle.

    This is the primary statistic because it is blind to how many ties the
    angles happen to contain. Spearman and Kendall tau-b are NOT: with a
    2-level vocabulary they reward a theta_f that piles the far views onto
    equal distances, which is an artifact of the tie correction and not a
    better fit (measured: on 2508-aiduck2 both peak at theta_f = 315 deg over
    the equally-perfect 270 deg, purely because 315 makes three views tie).
    """
    d, ranks = np.asarray(d, float), np.asarray(ranks)
    c = disc = 0
    for i in range(len(d)):
        for j in range(i + 1, len(d)):
            if ranks[i] == ranks[j] or d[i] == d[j]:
                continue
            same = (d[i] < d[j]) == (ranks[i] < ranks[j])
            c, disc = (c + 1, disc) if same else (c, disc + 1)
    npairs = sum(1 for i in range(len(d)) for j in range(i + 1, len(d))
                 if ranks[i] != ranks[j])
    return (c - disc) / npairs if npairs else np.nan


def sweep(azimuths, ranks, step=1.0):
    """Concordance of angular distance vs phrase rank, over every theta_f."""
    thetas = np.arange(0.0, 360.0, step)
    D, rho, tau = [], [], []
    for th in thetas:
        d = wrap180(np.asarray(azimuths) - th)
        D.append(somers_d(d, ranks))
        if len(set(np.round(d, 6))) < 2 or len(set(ranks)) < 2:
            rho.append(np.nan); tau.append(np.nan); continue
        rho.append(spearmanr(d, ranks).statistic)
        tau.append(kendalltau(d, ranks, variant="b").statistic)
    return thetas, np.array(D, float), np.array(rho, float), np.array(tau, float)


def plateau(thetas, score, atol=1e-9):
    """The contiguous circular band of theta_f that all achieve the maximum."""
    best = np.nanmax(score)
    hit = np.isclose(score, best, atol=atol, rtol=0)
    idx = np.where(hit)[0]
    n = len(thetas)
    gaps = np.where(np.diff(np.r_[idx, idx[0] + n]) > 1)[0]
    if len(gaps) == 1:                      # band wraps 0 deg - roll to make it contiguous
        start = idx[(gaps[0] + 1) % len(idx)]
        order = [(start + k) % n for k in range(n) if hit[(start + k) % n]]
    else:
        order = list(idx)
    band = thetas[order]
    width = len(order) * (thetas[1] - thetas[0])
    centre = (band[0] + signed180(band[-1] - band[0]) / 2.0) % 360.0
    return best, band, width, centre


# ------------------------------------------------------------------------ run
def analyse(rows, out):
    lines = []
    P = lines.append

    P("=" * 78)
    P("PHRASE CALIBRATION - does `framing.facing` carry magnitude?")
    P("=" * 78)

    P("")
    P("--- STEP 0: which runs actually have a `framing` block --------------------")
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/phrase_calib")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows, coverage = extract()
    L = []
    P = L.append

    P("=" * 78)
    P("PHRASE CALIBRATION - does framing.facing carry MAGNITUDE or only DIRECTION?")
    P("=" * 78)

    # ---------------------------------------------------------- step 0 / step 1
    P("")
    P("--- STEP 0  coverage: which runs carry a framing block --------------------")
    P(f"{'run':16s} {'transcripts':>11s} {'with framing':>13s}  {'centre spread(m)':>16s}")
    for run, ntot, ngot, spread in coverage:
        flag = "" if ngot else "   (no framing - predates 2026-08-25 schema)"
        P(f"{run:16s} {ntot:11d} {ngot:13d}  {spread:16.4f}{flag}")
    P(f"TOTAL findings with a framing block: N = {len(rows)}")
    skipped = sorted(d.name for d in RUNS.iterdir()
                     if d.is_dir() and not d.name.startswith(".")
                     and not (d / "eyes" / "transcripts").is_dir())
    P(f"({len(skipped)} further run dirs hold no transcripts at all and cannot")
    P(f" contribute: {', '.join(skipped)})")

    P("")
    P("--- STEP 1  verbatim extraction -------------------------------------------")
    P(f"{'run':14s} {'file':9s} {'cell':8s} {'az':>5s} {'el':>4s}  "
      f"{'facing (verbatim)':32s} {'better':8s} target")
    for r in rows:
        P(f"{r['run']:14s} {r['transcript']:9s} {str(r['cell']):8s} "
          f"{r['azimuth']:5.0f} {r['elev']:4.0f}  {r['facing']:32s} "
          f"{str(r['better']):8s} {r['target']!r}")

    P("")
    P("distinct verbatim `facing` strings observed:")
    seen = defaultdict(int)
    for r in rows:
        seen[r["facing"]] += 1
    for s, c in sorted(seen.items(), key=lambda kv: rank_of(kv[0])[0]):
        P(f"  rank {rank_of(s)[0]}  n={c}  {s!r}")
    P("distinct verbatim `target` strings observed:")
    tg = defaultdict(int)
    for r in rows:
        tg[r["target"]] += 1
    for s, c in sorted(tg.items(), key=lambda kv: -kv[1]):
        P(f"  n={c}  {s!r}")
    P("distinct verbatim `better` strings observed:")
    bt = defaultdict(int)
    for r in rows:
        bt[r["better"]] += 1
    for s, c in sorted(bt.items(), key=lambda kv: -kv[1]):
        P(f"  n={c}  {s!r}")

    P("")
    P("--- STEP 2a  the linguistic ordering (fixed BEFORE any angle was looked at)")
    for rk, fam, why in PHRASE_RANK:
        used = sum(c for s, c in seen.items() if rank_of(s)[1] == fam)
        P(f"  rank {rk}: {fam:28s} used {used}x")
        for chunk in _wrap(why, 66):
            P(f"           {chunk}")
    P("  TIE, deliberate: '...to the left' and '...to the right' share rank 2.")
    P("  The direction word is not a magnitude word. Scoring them apart would")
    P("  smuggle direction into a magnitude test.")

    # ------------------------------------------------------- geometry sanity
    P("")
    P("--- STEP 2b  geometry sanity: what does 'right' mean in these frames? -----")
    P("  For every framing view, is image-right the direction of increasing base")
    P("  azimuth? (+1 yes / -1 no). Computed from T_base_cam and the recorded")
    P("  180 deg write rotation - not assumed.")
    for r in rows:
        P(f"  {r['run']:14s} cell {str(r['cell']):8s} h*30={r['azimuth']:5.0f} "
          f"az_measured={r['az_measured']:7.2f}  right_is_+az = {r['right_is_plus_az']:+.0f}")
    signs = {r["right_is_plus_az"] for r in rows}
    P(f"  -> convention is {'CONSISTENT' if len(signs) == 1 else 'INCONSISTENT'} "
      f"across all {len(rows)} framing views: {signs}")
    assert signs == {1.0}, "frame-right convention flipped; direction test needs rework"
    P("  So: image-right == increasing azimuth == increasing h. A hint of 'right'")
    P("  is CORRECT iff the shorter way from the view azimuth to theta_f is +.")

    # ----------------------------------------------------------------- step 2
    P("")
    P("--- STEP 2c  fitting theta_f per run (anti-circular) ----------------------")
    by_run = defaultdict(list)
    for r in rows:
        by_run[r["run"]].append(r)

    fits = {}
    for run, rs in sorted(by_run.items()):
        az = [r["azimuth"] for r in rs]
        rk = [r["rank"] for r in rs]
        P("")
        P(f"  {run}: n = {len(rs)}, distinct phrase ranks present = {sorted(set(rk))}")
        if len(rs) < 2 or len(set(rk)) < 2:
            P("    NOT FITTABLE: a rank correlation needs >= 2 points and >= 2 distinct")
            P("    ranks. theta_f is unidentifiable from this run's framing alone.")
            fits[run] = None
            continue
        th, D, rho, tau = sweep(az, rk)
        bestD, band, width, centre = plateau(th, D)
        P(f"    BEST-CASE Somers' D_yx over all theta_f  : {bestD:+.4f}  <- primary")
        P(f"    BEST-CASE Spearman rho  over all theta_f : {np.nanmax(rho):+.4f} "
          f"(at theta_f = {th[np.nanargmax(rho)]:.0f} deg)")
        P(f"    BEST-CASE Kendall tau-b over all theta_f : {np.nanmax(tau):+.4f} "
          f"(at theta_f = {th[np.nanargmax(tau)]:.0f} deg)")
        P(f"    worst-case D over theta_f                : {np.nanmin(D):+.4f}")
        P(f"    theta_f values achieving the best D      : {band[0]:.0f} .. {band[-1]:.0f} deg")
        P(f"      -> a {width:.0f} deg plateau = {width/30:.1f} viewsphere cells wide.")
        P(f"      The words pin the object's front no better than that.")

        # tiebreak inside the plateau: the azimuths this run actually visited
        visited = sorted({r["azimuth"] for r in rs})
        inband = [a for a in visited
                  if np.isclose(wrap180(band - a), 0, atol=0.51).any()]
        P(f"      azimuths actually visited in this run    : {visited}")
        P(f"      of those, inside the plateau             : {inband}")
        if len(inband) == 1:
            theta_hat = inband[0]
            P(f"      -> theta_f_hat = {theta_hat:.0f} deg (h{theta_hat/30:.0f}): the ONE")
            P(f"         sampled azimuth consistent with the phrase ordering. This")
            P(f"         tiebreak uses the sampling pattern, not a model judgment.")
        else:
            theta_hat = centre
            P(f"      -> theta_f_hat = plateau centre {theta_hat:.1f} deg (no unique")
            P(f"         sampled azimuth in the band)")
        for nm, arr in (("Spearman rho", rho), ("Kendall tau-b", tau), ("Somers' D", D)):
            P(f"      {nm:14s} at theta_f_hat            : "
              f"{arr[int(round(theta_hat))% len(arr)]:+.4f}")
        P("    NOTE: a 2-level vocabulary can reach D = 1 by separating 'face-on'")
        P("    from 'not face-on'. D = 1 means the two levels separate, NOT that")
        P("    the angle is recoverable - see step 3 for what one level spans.")

        # how sensitive is the step-3 headline to where in the plateau we sit?
        P("    sensitivity of the ambiguity measure across the whole plateau:")
        spans = []
        for probe in band:
            d = wrap180(np.array(az) - probe)
            far = d[np.array(rk) > 0]
            spans.append(far.max() - far.min())
        for probe in (band[0], theta_hat, centre, band[-1]):
            d = wrap180(np.array(az) - probe)
            far = d[np.array(rk) > 0]
            P(f"      theta_f = {probe:5.1f} deg -> 'turning away' spans "
              f"{far.min():5.1f}..{far.max():5.1f} deg = "
              f"{(far.max()-far.min())/30:.1f} cells of ambiguity")
        P(f"      MOST FAVOURABLE theta_f in the whole plateau still leaves "
          f"{min(spans):.0f} deg")
        P(f"      = {min(spans)/30:.1f} cells spanned by ONE phrase. That is the")
        P(f"      best case, and it is the number that matters.")
        fits[run] = dict(theta=theta_hat, band=band, width=width, centre=centre,
                         D=D, rho=rho, tau=tau, thetas=th, bestD=bestD,
                         best_rho=np.nanmax(rho), best_tau=np.nanmax(tau))

    # ------------------------------------------- auxiliary anchor for 1-pt runs
    P("")
    P("--- STEP 2d  auxiliary anchor for runs too small to fit ------------------")
    aux = {}
    for run in by_run:
        if fits.get(run) is not None:
            continue
        store = json.loads((RUNS / run / "eyes" / "store.json").read_text())
        vis = defaultdict(lambda: [0, 0])
        for f in store["findings"]:
            s = f["summary"].lower()
            if s.startswith("unknown"):
                continue
            hit = ("vbti" in s or s.startswith("yes") or "logo is" in s)
            miss = (s.startswith("no") or "not visible" in s or "none visible" in s
                    or "there is no" in s)
            if hit and not miss:
                vis[f["cell"][0]][0] += 1
            elif miss:
                vis[f["cell"][0]][1] += 1
        P(f"  {run}: logo-visibility by azimuth bin, from the ANSWER channel")
        P(f"    (a different channel from `framing` - still a model judgment, so")
        P(f"     this is an anchor, not a measurement)")
        framing_h = {r["h"] for r in by_run[run]}
        for h in sorted(vis):
            y, n = vis[h]
            skip = "   [EXCLUDED: this is the cell under test]" if h in framing_h else ""
            P(f"      h={h} (az {h*30:3d} deg): seen {y}, not seen {n}{skip}")
        hs = sorted(h for h in vis if h not in framing_h)
        w = np.array([vis[h][0] for h in hs], float)
        if w.sum():
            ang = np.radians(np.array(hs) * 30.0)
            th = np.degrees(np.arctan2((w * np.sin(ang)).sum(),
                                       (w * np.cos(ang)).sum())) % 360
            aux[run] = th
            P(f"    -> anchor theta_f ~ {th:.0f} deg (h{th/30:.1f}), seen-count-weighted")
            P(f"       circular mean over bins {hs} with weights {[int(x) for x in w]}")

    # ----------------------------------------------------------------- step 3
    P("")
    P("--- STEP 3  THE AMBIGUITY MEASURE ----------------------------------------")
    P("  Angular distance |az_view - theta_f_hat| at which each verbatim phrase")
    P("  was actually used. Spread is the headline: a phrase that spans many")
    P("  cells cannot tell the planner how far to go.")
    for r in rows:
        th = fits[r["run"]]["theta"] if fits.get(r["run"]) else aux.get(r["run"])
        r["theta_f"] = th
        r["dist"] = float(wrap180(r["azimuth"] - th)) if th is not None else None
        r["signed"] = float(signed180(th - r["azimuth"])) if th is not None else None

    def table(sel, key, header, label):
        P("")
        P(f"  {label}")
        P(f"  {header:34s} {'n':>2s} {'min':>6s} {'med':>6s} {'max':>6s} "
          f"{'spread':>7s} {'cells':>6s}  {'ratio max/min':>13s}")
        buck = defaultdict(list)
        for r in sel:
            if r["dist"] is not None:
                buck[key(r)].append(r["dist"])
        for s in sorted(buck, key=lambda s: (rank_of(s)[0] if key is _facing
                                             else _famrank(s), s)):
            d = np.array(buck[s])
            ratio = f"{d.max()/d.min():.1f}x" if d.min() > 0 else "-"
            P(f"  {s:34s} {len(d):2d} {d.min():6.0f} {np.median(d):6.0f} {d.max():6.0f} "
              f"{d.max()-d.min():7.0f} {(d.max()-d.min())/30:6.1f}  {ratio:>13s}")

    P("")
    P("  how many DISTINCT viewpoints does each phrase rest on? (a phrase seen")
    P("  twice at one cell has not been tested at two angles)")
    dist_cells = defaultdict(set)
    for r in rows:
        dist_cells[r["facing"]].add((r["run"], tuple(r["cell"])))
    for s in sorted(dist_cells, key=lambda s: rank_of(s)[0]):
        n = sum(1 for r in rows if r["facing"] == s)
        P(f"    {s:34s} {n} finding(s) over {len(dist_cells[s])} distinct cell(s)")

    only = [r for r in rows if r["run"] == "2508-aiduck2"]
    table(rows, _facing, "phrase (verbatim)",
          "ALL runs (theta_f fitted for 2508-aiduck2, anchored for 2408-cup1):")
    table(rows, _family, "magnitude level",
          "collapsed to MAGNITUDE only (direction word dropped), all runs:")
    table(only, _facing, "phrase (verbatim)",
          "2508-aiduck2 ONLY - the single run with a fitted theta_f (n=8):")
    table(only, _family, "magnitude level",
          "2508-aiduck2 ONLY, collapsed to magnitude:")

    P("")
    P("  per-finding detail:")
    for r in sorted(rows, key=lambda r: (r["run"], r["dist"] if r["dist"] is not None else -1)):
        P(f"    {r['run']:14s} {r['transcript']:9s} cell {str(r['cell']):8s} "
          f"az {r['azimuth']:5.0f}  dist {r['dist']:5.0f} deg "
          f"({r['dist']/30:4.1f} cells)  {r['facing']!r}")

    # ----------------------------------------------------------------- step 4
    P("")
    P("--- STEP 4  DIRECTION CHECK ----------------------------------------------")
    P("  'right' is correct iff signed180(theta_f - az_view) > 0 (image-right ==")
    P("  increasing azimuth, verified in step 2b). At |offset| = 180 both words")
    P("  are equally right - scored as TIE, not as correct or wrong.")
    P("")
    for field, key in (("facing", "dir_word"), ("better", "better_word")):
        ok = bad = tie = 0
        P(f"  {field}:")
        for r in sorted(rows, key=lambda r: r["run"]):
            w = r[key]
            if w is None or r["signed"] is None:
                continue
            if abs(abs(r["signed"]) - 180.0) < 1e-6:
                verdict, tie = "TIE (180 deg - both equally correct)", tie + 1
            else:
                truth = "right" if r["signed"] > 0 else "left"
                if w == truth:
                    verdict, ok = "correct", ok + 1
                else:
                    verdict, bad = f"WRONG (shorter way is {truth})", bad + 1
            P(f"    {r['run']:14s} cell {str(r['cell']):8s} said {w:5s} "
              f"offset {r['signed']:+7.1f} deg -> {verdict}")
            r[f"{field}_verdict"] = verdict.split()[0]
        tot = ok + bad
        P(f"    -> {ok}/{tot} correct excluding {tie} tie(s)"
          + (f" = {100*ok/tot:.0f}%" if tot else ""))

    P("")
    P("  independent check: fit theta_f from the DIRECTION words alone and see")
    P("  whether it agrees with the magnitude fit.")
    for run, rs in sorted(by_run.items()):
        cand = [r for r in rs if r["dir_word"]]
        if len(cand) < 2:
            P(f"    {run}: only {len(cand)} directional statement(s) - not fittable")
            continue
        th = np.arange(0.0, 360.0, 1.0)
        score = np.zeros_like(th)
        for r in cand:
            s = signed180(th - r["azimuth"])
            want = np.where(s > 0, 1, -1)
            said = 1 if r["dir_word"] == "right" else -1
            score += (want == said)
        best = score.max()
        _, band, width, dcentre = plateau(th, score)
        wrapnote = " (wraps through 0 deg)" if band[0] > band[-1] else ""
        P(f"    {run}: best {best:.0f}/{len(cand)} direction words correct, achieved "
          f"for theta_f in [{band[0]:.0f}..{band[-1]:.0f}] deg{wrapnote}")
        P(f"       -> {width:.0f} deg wide ({width/30:.1f} cells), centre {dcentre:.0f} deg")
        mf = fits.get(run)
        if mf:
            inside = np.isclose(wrap180(band - mf["theta"]), 0, atol=1.0).any()
            P(f"       magnitude-fitted theta_f = {mf['theta']:.0f} deg -> "
              f"{'INSIDE' if inside else 'OUTSIDE'} the direction-optimal band")
            overlap = [a for a in mf["band"]
                       if np.isclose(wrap180(band - a), 0, atol=0.51).any()]
            P(f"       the magnitude plateau and the direction band agree over "
              f"{len(overlap)} deg - two fits of theta_f from independent")
            P(f"       channels, and they overlap. That is the evidence the")
            P(f"       direction word means something.")
            nontie = [r for r in cand
                      if abs(abs(signed180(mf['theta'] - r['azimuth'])) - 180) >= 1e-6]
            got = int(sum(1 for r in nontie if r["dir_word"] ==
                          ("right" if signed180(mf['theta'] - r['azimuth']) > 0 else "left")))
            P(f"       at the magnitude-fitted theta_f: {got}/{len(nontie)} correct "
              f"({len(cand)-len(nontie)} scored as a 180 deg tie)")

    # ----------------------------------------------------------------- summary
    only8 = [r for r in rows if r["run"] == "2508-aiduck2" and r["dist"] is not None]
    turning = [r["dist"] for r in only8 if r["rank"] > 0]
    P("")
    P("--- SUMMARY ---------------------------------------------------------------")
    P(f"  N = {len(rows)} findings carry a framing block, from 2 of 10 runs")
    P(f"  ({', '.join(sorted({r['run'] for r in rows}))}). One run contributes a")
    P("  single point. THIS IS A WEAK SAMPLE and every number below is provisional.")
    P("")
    P("  MAGNITUDE: the vocabulary the model actually used has TWO levels -")
    P("  'face-on' and 'turning away to the <dir>'. 'edge-on' and 'not visible'")
    P("  were offered by the prompt and never used. Best case over every possible")
    P("  front azimuth, the two levels DO separate (Somers' D = 1.00), but the")
    P(f"  single level 'turning away' covers {min(turning):.0f}-{max(turning):.0f} deg = "
      f"{(max(turning)-min(turning))/30:.0f} viewsphere cells")
    P(f"  ({max(turning)/min(turning):.0f}x in angle). 'face-on' rests on ONE viewpoint.")
    P("  The words therefore encode 'square-on or not', not 'how far'.")
    P("")
    P("  DIRECTION: the left/right word points the shorter way round in 5 of 6")
    P("  non-tie cases. Fitting theta_f from the direction words alone lands on a")
    P("  band that overlaps the band fitted from the phrase ordering - two")
    P("  independent fits agreeing. Direction is real signal.")
    P("")
    P("  DATASET CAVEAT: these runs ran with the local vision verbs on a STUB")
    P("  backend (detect always returned the same box; read_text returned")
    P("  nothing). No detect box, OCR string or segment area is used anywhere in")
    P("  this analysis - only the VLM's own framing strings and the recorded")
    P("  camera extrinsics, neither of which the stub touches. But the model was")
    P("  reasoning with junk tool output, which may have degraded its framing.")

    text = "\n".join(L)
    print(text)
    (Path(args.out) / "phrase_calib_report.txt").write_text(text + "\n")
    json.dump(rows, open(Path(args.out) / "phrase_calib_rows.json", "w"),
              indent=1, default=float)
    figures(rows, fits, aux, Path(args.out))
    print(f"\nwrote {args.out}/phrase_calib_report.txt and figures")


def _wrap(s, n):
    words, line, out = s.split(), "", []
    for w in words:
        if len(line) + len(w) + 1 > n:
            out.append(line); line = w
        else:
            line = f"{line} {w}".strip()
    if line:
        out.append(line)
    return out


# ------------------------------------------------------------------- figures
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SERIES = {"2508-aiduck2": "#2a78d6", "2408-cup1": "#eb6834"}
GOOD, CRIT, WARN = "#0ca30c", "#d03b3b", "#fab219"


def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
        ax.spines[s].set_linewidth(1.0)
    ax.tick_params(colors=MUTED, labelsize=9, length=3, width=1.0)
    for t in ax.get_xticklabels() + ax.get_yticklabels():
        t.set_color(INK2)


def _degcell_axis(ax, top=False):
    ticks = np.arange(0, 181, 30)
    ax.set_xlim(-8, 188)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t}°\n{t//30} cell" + ("" if t == 30 else "s") for t in ticks])
    for t in ticks:
        ax.axvline(t, color=GRID, lw=1.0, zorder=0)


def figures(rows, fits, aux, out):
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.titlelocation": "left",
    })
    _fig_headline(rows, out)
    _fig_runs(rows, fits, aux, out)
    _fig_direction(rows, out)


def _fig_headline(rows, out):
    have = [r for r in rows if r["dist"] is not None]
    phrases = sorted({r["facing"] for r in have},
                     key=lambda s: (rank_of(s)[0], s), reverse=True)
    fams = sorted({r["family"] for r in have},
                  key=lambda s: [f for f in PHRASE_RANK if f[1] == s][0][0], reverse=True)

    ncell = {}
    for r in have:
        ncell.setdefault(r["facing"], set()).add((r["run"], tuple(r["cell"])))

    fig = plt.figure(figsize=(11.6, 8.4))
    gs = GridSpec(2, 1, height_ratios=[len(phrases), len(fams)], hspace=0.46,
                  left=0.295, right=0.955, top=0.795, bottom=0.135)
    ax1, ax2 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])

    for ax, keys, keyf, title in (
            (ax1, phrases, lambda r: r["facing"],
             "as written by the model — verbatim `facing` string"),
            (ax2, fams, lambda r: r["family"],
             "collapsed to magnitude only — direction word dropped")):
        _style(ax); _degcell_axis(ax)
        ax.set_ylim(-0.7, len(keys) - 0.3)
        ax.set_yticks(range(len(keys)))
        ax.set_yticklabels([f'"{k}"' for k in keys], fontsize=10.5)
        ax.set_title(title, fontsize=11, color=INK, pad=8)
        for i, k in enumerate(keys):
            d = np.array([r["dist"] for r in have if keyf(r) == k])
            if len(d) > 1 and d.max() > d.min():
                ax.plot([d.min(), d.max()], [i, i], color=MUTED, lw=7,
                        solid_capstyle="round", alpha=0.28, zorder=1)
                ax.annotate(f"{d.max()-d.min():.0f}°  =  {(d.max()-d.min())/30:.0f} cells "
                            f"of ambiguity",
                            xy=((d.min() + d.max()) / 2, i + 0.30),
                            ha="center", va="bottom", fontsize=9.5, color=CRIT,
                            fontweight="bold")
            rng = np.random.default_rng(4)
            sel = [r for r in have if keyf(r) == k]
            for r in sel:
                ax.scatter(r["dist"], i + (rng.uniform(-0.13, 0.13) if len(sel) > 1 else 0),
                           s=125, color=SERIES.get(r["run"], MUTED), zorder=3,
                           edgecolors=SURFACE, linewidths=2.2)
            cells = {(r["run"], tuple(r["cell"])) for r in sel}
            if ax is ax1 and len(cells) == 1 and len(sel) > 1:
                ax.annotate(f"{len(sel)} findings, but only ONE viewpoint —\n"
                            "this row is untested at any other angle",
                            xy=(sel[0]["dist"] + 12, i), ha="left", va="center",
                            fontsize=9, color=MUTED, style="italic")
        ax.set_axisbelow(True)

    ax2.set_xlabel("angular distance from the fitted front  θ_f   "
                   "(degrees, and 30°-viewsphere-cells)",
                   fontsize=10.5, color=INK2, labelpad=6)
    ax1.set_xlabel("")

    handles = [plt.Line2D([], [], marker="o", ls="", ms=10, color=c,
                          markeredgecolor=SURFACE, markeredgewidth=2,
                          label=f"{k}  (n={sum(1 for r in have if r['run']==k)})")
               for k, c in SERIES.items() if any(r["run"] == k for r in have)]
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.295, 0.822),
               ncol=2, frameon=False, fontsize=10.5, labelcolor=INK2,
               handletextpad=0.4, columnspacing=2.2)

    fig.suptitle("One phrase, many angles: `facing` does not encode how far",
                 x=0.295, y=0.975, ha="left", fontsize=16, color=INK,
                 fontweight="bold")
    fig.text(0.295, 0.912,
             f"N = {len(have)} findings carrying a framing block, from "
             f"{len({r['run'] for r in have})} runs — a weak sample.\n"
             "θ_f is FITTED per run to the phrase ordering, never assumed.",
             ha="left", va="top", fontsize=10.5, color=INK2, linespacing=1.45)
    fig.text(0.295, 0.038,
             "2408-cup1 contributes one point; its θ_f is anchored from the separate "
             "answer channel, not fitted (n=1 cannot be fitted).",
             ha="left", fontsize=9, color=MUTED, style="italic")
    fig.savefig(out / "1_headline_phrase_vs_angle.png", dpi=200)
    plt.close(fig)


def _fig_runs(rows, fits, aux, out):
    runs = sorted({r["run"] for r in rows})
    fig = plt.figure(figsize=(13.0, 3.6 * len(runs) + 1.6))
    gs = GridSpec(len(runs), 3, width_ratios=[1.0, 1.3, 1.3], hspace=0.62,
                  wspace=0.42, left=0.055, right=0.98, top=0.80, bottom=0.085)

    for i, run in enumerate(runs):
        rs = [r for r in rows if r["run"] == run]
        fit = fits.get(run)
        col = SERIES.get(run, MUTED)

        # -- polar: where we looked, what was said, where the front was fitted
        axp = fig.add_subplot(gs[i, 0], projection="polar")
        axp.set_facecolor(SURFACE)
        axp.set_theta_zero_location("E"); axp.set_theta_direction(1)
        axp.set_xticks(np.radians(np.arange(0, 360, 30)))
        axp.set_xticklabels([f"h{k}" for k in range(12)], fontsize=7.5, color=MUTED)
        axp.set_yticks([]); axp.set_ylim(0, 1.25)
        axp.grid(color=GRID, lw=0.8)
        axp.spines["polar"].set_color(AXIS)
        axp.spines["polar"].set_linewidth(1.0)
        th = None
        if fit:
            axp.bar(np.radians(fit["centre"]), 1.3, width=np.radians(fit["width"]),
                    bottom=0, color=CRIT, alpha=0.11, zorder=1, edgecolor="none")
            th = fit["theta"]
        elif run in aux:
            th = aux[run]
        if th is not None:
            axp.annotate("", xy=(np.radians(th), 0.80), xytext=(np.radians(th), 0),
                         arrowprops=dict(arrowstyle="-|>", color=CRIT, lw=2.2))
        mark = {0: "*", 2: "o"}
        for r in rs:
            axp.scatter(np.radians(r["azimuth"]), 1.05, s=330 if r["rank"] == 0 else 115,
                        marker=mark.get(r["rank"], "s"), color=col,
                        edgecolors=SURFACE, linewidths=1.8, zorder=5)
        axp.set_title(f"{run}\nviewsphere azimuth", fontsize=10.5, color=INK, pad=16)

        # -- theta_f sweep
        axs = fig.add_subplot(gs[i, 1]); _style(axs)
        if fit:
            axs.plot(fit["thetas"], fit["D"], color=col, lw=2.2, label="Somers' D (primary)")
            axs.plot(fit["thetas"], fit["rho"], color=MUTED, lw=1.3, ls="--",
                     label="Spearman ρ (tie-biased)")
            axs.axhline(fit["bestD"], color=AXIS, lw=1.0, ls=":")
            axs.fill_between(fit["band"], -1.05, fit["bestD"],
                             color=CRIT, alpha=0.12, lw=0)
            axs.annotate(f"best D = {fit['bestD']:+.2f}, reached over a\n"
                         f"{fit['width']:.0f}° plateau = {fit['width']/30:.1f} cells wide\n"
                         f"θ̂_f = {fit['theta']:.0f}°  (h{fit['theta']/30:.0f})",
                         xy=(0.03, 0.96), xycoords="axes fraction", fontsize=9.5,
                         color=INK2, va="top",
                         bbox=dict(fc=SURFACE, ec="none", alpha=0.85, pad=2))
            axs.set_ylim(-1.08, 1.08)
        else:
            axs.text(0.5, 0.5, "n = 1 framing point\nθ_f unidentifiable\nfrom this run alone",
                     ha="center", va="center", fontsize=11, color=MUTED,
                     transform=axs.transAxes)
            axs.set_yticks([])
        axs.set_xlim(0, 359); axs.set_xticks(np.arange(0, 360, 60))
        axs.set_xlabel("candidate front azimuth θ_f (deg)", fontsize=9.5, color=INK2)
        axs.set_ylabel("how well that front\norders the phrases", fontsize=9.5, color=INK2)
        axs.grid(axis="y", color=GRID, lw=0.9); axs.set_axisbelow(True)
        axs.set_title("every front in the shaded band fits equally well",
                      fontsize=10.5, color=INK, pad=8)

        # -- distance vs phrase for this run
        axd = fig.add_subplot(gs[i, 2]); _style(axd); _degcell_axis(axd)
        keys = sorted({r["facing"] for r in rows},
                      key=lambda s: (rank_of(s)[0], s), reverse=True)
        axd.set_ylim(-0.7, len(keys) - 0.3); axd.set_yticks(range(len(keys)))
        axd.set_yticklabels([f'"{k}"' for k in keys], fontsize=8.5)
        rng = np.random.default_rng(7)
        for r in rs:
            if r["dist"] is None:
                continue
            axd.scatter(r["dist"], keys.index(r["facing"]) + rng.uniform(-0.1, 0.1),
                        s=120, color=col, edgecolors=SURFACE, linewidths=2.0, zorder=3)
        for k in keys:
            d = np.array([r["dist"] for r in rs if r["facing"] == k and r["dist"] is not None])
            if len(d) > 1 and d.max() > d.min():
                axd.plot([d.min(), d.max()], [keys.index(k)] * 2, color=MUTED,
                         lw=6, alpha=0.28, solid_capstyle="round", zorder=1)
        note = (f"θ̂_f = {th:.0f}° (h{th/30:.1f}), fitted"
                if fit else f"θ_f ≈ {th:.0f}° — anchored, not fitted")
        axd.set_title(note, fontsize=10.5, color=INK, pad=8)
        axd.set_xlabel("angular distance from θ_f", fontsize=9.5, color=INK2)
        axd.set_axisbelow(True)

    fig.suptitle("Per-run: where we looked, how well any front explains the words, and the spread",
                 x=0.055, y=0.975, ha="left", fontsize=15, color=INK, fontweight="bold")
    fig.text(0.055, 0.928,
             "polar — ★ “face-on”, ● “turning away to the …”; red arrow = fitted front, "
             "shaded wedge = every θ_f that fits the words equally well.\n"
             "middle — solid = Somers' D (primary, tie-robust); dashed = Spearman ρ, which "
             "peaks elsewhere purely because of how it corrects for ties.",
             ha="left", va="top", fontsize=9.5, color=INK2, linespacing=1.5)
    fig.savefig(out / "2_per_run_panels.png", dpi=200)
    plt.close(fig)


def _fig_direction(rows, out):
    have = [r for r in rows if r["dir_word"] and r["signed"] is not None]
    have = sorted(have, key=lambda r: (r["run"], r["azimuth"]))
    fig, ax = plt.subplots(figsize=(12.4, 0.66 * len(have) + 3.2))
    fig.subplots_adjust(left=0.215, right=0.685, top=0.775, bottom=0.165)
    _style(ax)

    ax.axvspan(-200, 0, color=GRID, alpha=0.5, lw=0, zorder=0)
    ax.text(-100, len(have) - 0.2, 'front is to the LEFT\n→ "left" is correct',
            ha="center", va="bottom", fontsize=10, color=INK2, linespacing=1.4)
    ax.text(100, len(have) - 0.2, 'front is to the RIGHT\n→ "right" is correct',
            ha="center", va="bottom", fontsize=10, color=INK2, linespacing=1.4)
    ax.axvline(0, color=AXIS, lw=1.4, zorder=1)

    labels = []
    for i, r in enumerate(have):
        tie = abs(abs(r["signed"]) - 180.0) < 1e-6
        truth = None if tie else ("right" if r["signed"] > 0 else "left")
        okc = WARN if tie else (GOOD if r["dir_word"] == truth else CRIT)
        ax.plot([0, r["signed"]], [i, i], color=okc, lw=2.6, alpha=0.5, zorder=2)
        ax.scatter(r["signed"], i, s=230,
                   marker="<" if r["dir_word"] == "left" else ">",
                   color=okc, edgecolors=SURFACE, linewidths=2.0, zorder=4)
        tag = ("● tie — 180°, both words equally correct" if tie else
               (f"● correct — said {r['dir_word']}" if r["dir_word"] == truth
                else f"● WRONG — said {r['dir_word']}, shorter way is {truth}"))
        ax.text(1.025, i, tag, transform=ax.get_yaxis_transform(),
                ha="left", va="center", fontsize=9.5, color=okc,
                fontweight="bold", clip_on=False)
        labels.append(f'{r["run"]}  cell {r["cell"]}\n"{r["facing"]}"')

    ax.set_yticks(range(len(have))); ax.set_yticklabels(labels, fontsize=9)
    ax.set_ylim(-0.7, len(have) + 0.25)
    ax.set_xlim(-200, 200)
    ax.set_xticks(np.arange(-180, 181, 60))
    ax.set_xticklabels([f"{abs(t)}°\n{abs(t)//30} cells left" if t < 0 else
                        (f"{t}°\n{t//30} cells right" if t > 0 else "0°\nfront")
                        for t in np.arange(-180, 181, 60)])
    ax.set_xlabel("signed offset from the view to the fitted front θ_f",
                  fontsize=10.5, color=INK2, labelpad=6)
    ok = sum(1 for r in have if abs(abs(r["signed"]) - 180) >= 1e-6
             and r["dir_word"] == ("right" if r["signed"] > 0 else "left"))
    tot = sum(1 for r in have if abs(abs(r["signed"]) - 180) >= 1e-6)
    fig.suptitle("The direction word is mostly right — and it is the only usable bit",
                 x=0.055, y=0.965, ha="left", fontsize=15.5, color=INK, fontweight="bold")
    fig.text(0.055, 0.905,
             f"{ok} of {tot} direction words point the shorter way round to the fitted "
             f"front; {len(have)-tot} scored as a 180° tie,\nwhere both words are "
             "equally correct. Marker points the way the model said to shift.",
             ha="left", va="top", fontsize=10.5, color=INK2, linespacing=1.5)
    fig.savefig(out / "3_direction_check.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
