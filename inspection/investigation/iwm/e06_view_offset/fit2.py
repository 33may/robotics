"""e06 fit2: depth model x hand-eye correction, fitted jointly, scored on held-out runs.

Models (camera-frame point p, z = its depth; T_base_cam' = T_base_cam @ dT):
  S      p' = s p                          (one depth scale; = fit.py)
  SB     p' = p (s + b / z)                (scale + constant range offset)
  SQ     p' = p (s + c z)                  (scale + range-squared term: disparity bias)
  H      dT only (6 dof hand-eye correction), depth untouched
  S+H, SQ+H   the two together

Residuals (all mm, soft-L1 at 3 mm, the two groups weighted equally):
  table  9 points per view on its measured camera-frame plane (rays x,y in {-.3,0,.3}),
         distance to the probed slab after the model;
  pairs  up to 20 SIFT points per pair (pairs.py store), 3D difference in base.
FIT = fit.FIT_RUNS. Scored on the other runs: table |err|, pair distance (all / without the
survey step 0), the named c5.2/c5.6 parts. Then leave-one-run-out on the pair runs.

Run:  p -m inspection.investigation.iwm.e06_view_offset.fit2
Out:  evidence/fit2.json, evidence/fit2.png
"""
import json

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from inspection.investigation.iwm.e06_view_offset import common as C
from inspection.investigation.iwm.e06_view_offset import fit as F

MODELS = {"S": ("s",), "SB": ("s", "b"), "SQ": ("s", "c"), "H": ("he",),
          "S+H": ("s", "he"), "SQ+H": ("s", "c", "he")}
RAYS = np.array([[x, y, 1.0] for x in (-.3, 0, .3) for y in (-.3, 0, .3)])
PAIR_PTS, F_SCALE = 20, 3.0
rng = np.random.default_rng(0)


# ---------- data
def table_views():
    d = json.load(open(C.EVID / "table.json"))
    out = []
    for v in d["views"]:
        if v["range_hit_m"] > F.MAX_RANGE or v["elev_class"] == "grazing":
            continue
        n, dd = np.array(v["plane_cam"][:3]), v["plane_cam"][3]
        den = RAYS @ n
        lam = -dd / den
        P = RAYS[lam > 0] * lam[lam > 0, None]
        T = C.load_view(v["run"], v["step"], arrays=False)["T"]
        out.append({"run": v["run"], "step": v["step"], "cls": v["elev_class"], "P": P, "T": T})
    return out


def pair_data():
    out = []
    for run, i, j, pci, pcj, Ti, Tj in F.sift_pairs():
        k = rng.choice(len(pci), min(PAIR_PTS, len(pci)), replace=False)
        out.append({"run": run, "i": i, "j": j, "a": pci, "b": pcj, "ka": pci[k], "kb": pcj[k], "Ti": Ti, "Tj": Tj})
    return out


# ---------- model
def unpack(name, x):
    p = {"s": 1.0, "b": 0.0, "c": 0.0, "dT": np.eye(4)}
    k = 0
    for f in MODELS[name]:
        if f == "he":
            p["dT"][:3, :3] = Rotation.from_rotvec(x[k:k + 3]).as_matrix()
            p["dT"][:3, 3] = x[k + 3:k + 6]
            k += 6
        else:
            p[f] = x[k]
            k += 1
    return p


def x0(name):
    return np.concatenate([np.zeros(6) if f == "he" else [1.0 if f == "s" else 0.0] for f in MODELS[name]])


def apply(p, P, T):
    z = P[:, 2:3]
    Pc = P * (p["s"] + p["b"] / z + p["c"] * z)
    return C.to_base(Pc, T @ p["dT"])


def table_res(p, views, truth):
    return np.concatenate([(apply(p, v["P"], v["T"]) @ truth[:3] + truth[3]) * 1000 for v in views])


def pair_res(p, pairs, key=("ka", "kb")):
    return np.concatenate([((apply(p, q[key[1]], q["Tj"]) - apply(p, q[key[0]], q["Ti"])) * 1000).ravel() for q in pairs])


def fit(name, views, pairs, truth):
    def f(x):
        p = unpack(name, x)
        rt, rp = table_res(p, views, truth), pair_res(p, pairs)
        return np.concatenate([rt / np.sqrt(len(rt)), rp / np.sqrt(len(rp))]) * 10
    r = least_squares(f, x0(name), loss="soft_l1", f_scale=F_SCALE / np.sqrt(500) * 10)
    return r.x


# ---------- scores
def pair_dist(p, q):
    D = apply(p, q["b"], q["Tj"]) - apply(p, q["a"], q["Ti"])
    return float(np.median(np.linalg.norm(D, axis=1)) * 1000)


def score(p, views, pairs, truth, named):
    t = [float(np.median(np.abs(table_res(p, [v], truth)))) for v in views]
    pd = [pair_dist(p, q) for q in pairs]
    close = [d for d, q in zip(pd, pairs) if q["i"] != 0]
    surv = [d for d, q in zip(pd, pairs) if q["i"] == 0]
    nm = [float(np.linalg.norm(np.median(apply(p, x["pc_b"], x["T_b"]), 0) - np.median(apply(p, x["pc_a"], x["T_a"]), 0)) * 1000) for x in named]
    md = lambda a: round(float(np.median(a)), 2) if len(a) else None
    p90 = lambda a: round(float(np.percentile(a, 90)), 2) if len(a) else None
    return {"table_med": md(t), "table_p90": p90(t), "pairs_med": md(pd), "pairs_p90": p90(pd),
            "survey_pairs_med": md(surv), "close_pairs_med": md(close), "named_med": md(nm)}


def describe(name, x):
    p = unpack(name, x)
    d = {"s": round(p["s"], 4)}
    if "b" in MODELS[name]:
        d["b_mm"] = round(p["b"] * 1000, 2)
    if "c" in MODELS[name]:
        d["c_per_m"] = round(p["c"], 4)
    if "he" in MODELS[name]:
        d["he_rot_mrad"] = (Rotation.from_matrix(p["dT"][:3, :3]).as_rotvec() * 1000).round(2).tolist()
        d["he_t_mm"] = (p["dT"][:3, 3] * 1000).round(2).tolist()
    return d


def main():
    truth = C.table_truth()[0]
    views, pairs, named = table_views(), pair_data(), F.named_parts()
    fv = [v for v in views if v["run"] in F.FIT_RUNS]
    vv = [v for v in views if v["run"] not in F.FIT_RUNS]
    fp = [q for q in pairs if q["run"] in F.FIT_RUNS]
    vp = [q for q in pairs if q["run"] not in F.FIT_RUNS]
    print(f"table views fit {len(fv)} / held-out {len(vv)}; pairs fit {len(fp)} / held-out {len(vp)}; named parts {len(named)}")
    ident = {"s": 1.0, "b": 0.0, "c": 0.0, "dT": np.eye(4)}
    out = {"none": {"params": {}, "heldout": score(ident, vv, vp, truth, named), "fitset": score(ident, fv, fp, truth, named)}}
    for name in MODELS:
        x = fit(name, fv, fp, truth)
        p = unpack(name, x)
        out[name] = {"params": describe(name, x), "heldout": score(p, vv, vp, truth, named), "fitset": score(p, fv, fp, truth, named)}
        print(f"{name:5s} {out[name]['params']}")
    hdr = ["table_med", "table_p90", "pairs_med", "pairs_p90", "survey_pairs_med", "close_pairs_med", "named_med"]
    print("\nHELD-OUT runs (mm)   " + "  ".join(f"{h:>16s}" for h in hdr))
    for k, r in out.items():
        print(f"{k:20s} " + "  ".join(f"{str(r['heldout'][h]):>16s}" for h in hdr))
    # leave-one-run-out over the pair runs, for the depth-only and joint models
    loro = {}
    for name in ("S", "SQ", "S+H", "SQ+H"):
        rows = []
        for run in sorted({q["run"] for q in pairs}):
            x = fit(name, [v for v in views if v["run"] != run], [q for q in pairs if q["run"] != run], truth)
            p = unpack(name, x)
            rows.append({"run": run, **{k: v for k, v in score(p, [v for v in views if v["run"] == run], [q for q in pairs if q["run"] == run], truth, []).items() if v is not None}})
        loro[name] = rows
        print(f"LORO {name:5s} pairs med per run: " + ", ".join(f"{r['run'][5:]} {r.get('pairs_med')}" for r in rows))
    out["loro"] = loro
    json.dump(out, open(C.EVID / "fit2.json", "w"), indent=1, cls=C.NpEncoder)
    figure(out)


def figure(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = [k for k in out if k != "loro"]
    keys = [("table_med", "table height error"), ("survey_pairs_med", "survey <-> close pairs"),
            ("close_pairs_med", "close <-> close pairs"), ("named_med", "named parts c5.2/c5.6")]
    fig, ax = plt.subplots(1, len(keys), figsize=(18, 4.2), sharey=True)
    for a, (k, t) in zip(ax, keys):
        vals = [out[n]["heldout"][k] or 0 for n in names]
        a.bar(names, vals, color=["#999999"] + ["#1f77b4"] * (len(names) - 1))
        for i, v in enumerate(vals):
            a.text(i, v + 0.2, f"{v:.1f}", ha="center", fontsize=8)
        a.axhline(3, color="#2ca02c", ls="--", lw=1)
        a.set_title(t)
        a.tick_params(axis="x", rotation=45)
    ax[0].set_ylabel("held-out median (mm)")
    fig.tight_layout()
    fig.savefig(C.EVID / "fit2.png", dpi=130)


if __name__ == "__main__":
    main()
