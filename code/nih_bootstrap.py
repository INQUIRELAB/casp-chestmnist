"""Patient-clustered bootstrap matching Table II (per-seed-mean statistic), vectorized."""
import json
from pathlib import Path
import numpy as np

PROBS = Path(__file__).resolve().parent / "nih_probs"
ORDER = ["searched_nonrec", "searched_nonrec_pm", "searched_rec", "resnet18",
         "resnet50", "densenet121", "mobilenet_v2", "vit_small"]
B = 2000


def load(model):
    fs = sorted(PROBS.glob(f"{model}_seed*.npz"))
    Ps, lab, pid = [], None, None
    for f in fs:
        z = np.load(f)
        Ps.append(1.0 / (1.0 + np.exp(-z["test_logits"].astype(np.float64))))
        lab, pid = z["test_labels"], z["test_patient"]
    return np.stack(Ps, 0), lab.astype(np.int64), pid.astype(np.int64)  # [S,N,14]


def seedmean_auc(Pstack, Y, idx):
    Ps = Pstack[:, idx, :]                      # [S,n,14]
    S, n, C = Ps.shape
    order = np.argsort(Ps, axis=1)
    ranks = np.empty_like(Ps)
    rows = np.arange(1, n + 1, dtype=np.float64)[None, :, None]
    np.put_along_axis(ranks, order, np.broadcast_to(rows, Ps.shape).copy(), axis=1)
    Ys = Y[idx]
    npos = Ys.sum(0).astype(np.float64); nneg = n - npos
    sr = (ranks * Ys[None, :, :]).sum(axis=1)   # [S,14]
    with np.errstate(invalid="ignore", divide="ignore"):
        auc = (sr - npos * (npos + 1) / 2.0) / (npos * nneg)
    valid = (npos > 0) & (nneg > 0)
    per_seed = np.nanmean(np.where(valid[None, :], auc, np.nan), axis=1)  # [S]
    return per_seed  # keep per-seed for point/SD; caller means it


def main():
    data = {}; lab = pid = None
    for m in ORDER:
        data[m], lab, pid = load(m)
    full = np.arange(len(pid))
    perseed_full = {m: seedmean_auc(data[m], lab, full) for m in ORDER}
    point = {m: float(perseed_full[m].mean()) for m in ORDER}
    seed_sd = {m: float(perseed_full[m].std(ddof=0)) for m in ORDER}

    upat = np.unique(pid)
    idx_by = [np.where(pid == p)[0] for p in upat]
    rng = np.random.default_rng(0)
    boot = {m: np.empty(B) for m in ORDER}
    for b in range(B):
        s = rng.integers(0, len(upat), size=len(upat))
        idx = np.concatenate([idx_by[j] for j in s])
        for m in ORDER:
            boot[m][b] = seedmean_auc(data[m], lab, idx).mean()

    out = {"n_test": int(len(pid)), "n_test_patients": int(len(upat)), "B": B,
           "statistic": "mean over 5 seeds of per-seed macro-AUC", "models": {}}
    print(f"patient-clustered bootstrap (per-seed-mean, vectorized)  B={B}  {len(pid)} imgs / {len(upat)} patients\n")
    for m in sorted(ORDER, key=lambda k: -point[k]):
        lo, hi = np.percentile(boot[m], [2.5, 97.5])
        out["models"][m] = dict(auc=round(point[m], 4), ci_lo=round(float(lo), 4),
                                ci_hi=round(float(hi), 4), seed_sd=round(seed_sd[m], 4))
        print(f"{m:20s} {point[m]:.4f}  [{lo:.4f},{hi:.4f}]  seedSD={seed_sd[m]:.4f}")

    pairs = [("searched_nonrec", "searched_rec"), ("searched_nonrec_pm", "searched_rec"),
             ("searched_nonrec", "resnet18"), ("searched_nonrec", "densenet121"),
             ("searched_nonrec", "searched_nonrec_pm")]
    out["paired"] = {}
    print("\npaired deltas:")
    for a, b in pairs:
        d = boot[a] - boot[b]
        lo, hi = np.percentile(d, [2.5, 97.5])
        frac = float(np.mean(d > 0)); p = 2.0 * min(frac, 1.0 - frac)
        out["paired"][f"{a}__vs__{b}"] = dict(delta=round(point[a] - point[b], 4),
                                              ci_lo=round(float(lo), 4), ci_hi=round(float(hi), 4),
                                              p=round(max(p, 1.0 / B), 4))
        print(f"  {a:18s}-{b:18s}  d={point[a]-point[b]:+.4f} [{lo:+.4f},{hi:+.4f}] p={max(p,1/B):.4f}")

    (Path(__file__).resolve().parent / "nih_bootstrap.json").write_text(json.dumps(out, indent=2))
    print("\nWROTE nih_bootstrap.json")


if __name__ == "__main__":
    main()
