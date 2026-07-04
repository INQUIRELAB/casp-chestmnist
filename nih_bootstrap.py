"""Patient-clustered bootstrap for the NIH external validation.

Resamples PATIENTS (not images) with replacement so the confidence intervals respect
within-patient correlation. Per model: 5-seed-averaged probabilities, macro-AUC point
estimate, and a patient-clustered 95% percentile CI. Also paired deltas (same resampled
patients for both models) for the decisive comparisons, with a bootstrap two-sided p.
"""
import json
import os
from pathlib import Path

import numpy as np

PROBS = Path(__file__).resolve().parent / "nih_probs"
ORDER = ["searched_nonrec", "searched_nonrec_pm", "searched_rec", "resnet18",
         "resnet50", "densenet121", "mobilenet_v2", "vit_small"]
B = 2000


def load(model):
    fs = sorted(PROBS.glob(f"{model}_seed*.npz"))
    Ps = []
    lab = pid = None
    for f in fs:
        z = np.load(f)
        Ps.append(1.0 / (1.0 + np.exp(-z["test_logits"].astype(np.float64))))
        lab = z["test_labels"]; pid = z["test_patient"]
    return np.mean(Ps, 0), lab.astype(np.int64), pid.astype(np.int64), len(fs)


def macro_auc(P, Y, idx):
    Ps = P[idx]; Ys = Y[idx]
    n = Ps.shape[0]
    order = np.argsort(Ps, axis=0)
    ranks = np.empty((n, Ps.shape[1]), dtype=np.float64)
    rows = np.arange(1, n + 1, dtype=np.float64)[:, None]
    np.put_along_axis(ranks, order, np.broadcast_to(rows, Ps.shape).copy(), axis=0)
    npos = Ys.sum(0); nneg = n - npos
    sr = (ranks * Ys).sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        auc = (sr - npos * (npos + 1) / 2.0) / (npos * nneg)
    valid = (npos > 0) & (nneg > 0)
    return float(np.nanmean(np.where(valid, auc, np.nan)))


def main():
    data = {}
    nseed = {}
    for m in ORDER:
        P, lab, pid, k = load(m)
        data[m] = P; nseed[m] = k
    _, lab, pid, _ = load(ORDER[0])
    upat = np.unique(pid)
    idx_by_pat = [np.where(pid == p)[0] for p in upat]
    full = np.arange(len(pid))
    point = {m: macro_auc(data[m], lab, full) for m in ORDER}

    rng = np.random.default_rng(0)
    boot = {m: np.empty(B) for m in ORDER}
    for b in range(B):
        samp = rng.integers(0, len(upat), size=len(upat))
        idx = np.concatenate([idx_by_pat[s] for s in samp])
        for m in ORDER:
            boot[m][b] = macro_auc(data[m], lab, idx)

    out = {"n_test": int(len(pid)), "n_test_patients": int(len(upat)), "B": B, "models": {}}
    print(f"patient-clustered bootstrap  B={B}  test={len(pid)} images / {len(upat)} patients\n")
    print(f"{'model':22s} {'auc':>7s}  {'95% CI (patient-clustered)':>28s}")
    for m in sorted(ORDER, key=lambda k: -point[k]):
        lo, hi = np.percentile(boot[m], [2.5, 97.5])
        out["models"][m] = dict(auc=round(point[m], 4), ci_lo=round(float(lo), 4),
                                ci_hi=round(float(hi), 4), n_seeds=nseed[m])
        print(f"{m:22s} {point[m]:.4f}  [{lo:.4f}, {hi:.4f}]")

    print("\npaired deltas (same resampled patients), 95% CI and two-sided bootstrap p:")
    pairs = [("searched_nonrec", "searched_rec"), ("searched_nonrec_pm", "searched_rec"),
             ("searched_nonrec", "resnet18"), ("searched_nonrec", "densenet121"),
             ("searched_nonrec", "searched_nonrec_pm")]
    out["paired"] = {}
    for a, b in pairs:
        d = boot[a] - boot[b]
        lo, hi = np.percentile(d, [2.5, 97.5])
        frac = float(np.mean(d > 0))
        p = 2.0 * min(frac, 1.0 - frac)
        out["paired"][f"{a}__vs__{b}"] = dict(delta=round(point[a] - point[b], 4),
                                              ci_lo=round(float(lo), 4), ci_hi=round(float(hi), 4),
                                              p=round(p, 4))
        print(f"  {a:20s} - {b:20s}  d={point[a]-point[b]:+.4f}  [{lo:+.4f}, {hi:+.4f}]  p={p:.4f}")

    (Path(__file__).resolve().parent / "nih_bootstrap.json").write_text(json.dumps(out, indent=2))
    print("\nWROTE nih_bootstrap.json")


if __name__ == "__main__":
    main()
