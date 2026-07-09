"""
reproduce_paper.py -- Deterministically regenerate EVERY number reported in the
manuscript from the shipped per-seed model outputs (nih_probs/) and result JSONs.

This is the reproducibility entry point. It is fully deterministic: it uses only
numpy/scikit-learn on the fixed saved arrays, with a fixed RNG seed for the
patient-clustered bootstrap. Running it on any machine yields identical numbers
(the training step that produced the saved outputs is GPU-hardware dependent and
is NOT re-run here; see README).

Outputs: prints a full report and writes results/reproduced_paper_numbers.txt.
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

HERE = Path(__file__).resolve().parent
PROBS = HERE / "nih_probs"
_pos = [a for a in sys.argv[1:] if not a.startswith("-")]
RESULTS = Path(_pos[0]) if _pos else HERE / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

ORDER = ["searched_nonrec", "searched_nonrec_pm", "searched_rec", "resnet18",
         "resnet50", "densenet121", "mobilenet_v2", "vit_small"]
PRETTY = {"searched_nonrec": "CASP", "searched_nonrec_pm": "CASP-matched",
          "searched_rec": "Recursive", "resnet18": "ResNet-18", "resnet50": "ResNet-50",
          "densenet121": "DenseNet-121", "mobilenet_v2": "MobileNetV2", "vit_small": "Compact ViT"}
B = 2000
_out = []


def log(s=""):
    print(s)
    _out.append(s)


def load(model):
    fs = sorted(PROBS.glob(f"{model}_seed*.npz"))
    Ps, lab, pid = [], None, None
    for f in fs:
        z = np.load(f)
        Ps.append(1.0 / (1.0 + np.exp(-z["test_logits"].astype(np.float64))))
        lab, pid = z["test_labels"], z["test_patient"]
    return np.stack(Ps, 0), lab.astype(np.int64), pid.astype(np.int64)


def jload(name):
    return json.loads((HERE / name).read_text())


# ---------------------------------------------------------------- Table III
def table_main():
    ch = jload("bench_results.json")
    nih = jload("bench_nih_results.json")
    fl = jload("flops_latency.json")
    log("=" * 78)
    log("TABLE III  Matched-protocol comparison (ChestMNIST + patient-disjoint NIH)")
    log("=" * 78)
    log(f"{'Model':14s} {'Params(M)':>9s} {'MACs(M)':>8s} {'Lat(ms)':>8s} "
        f"{'ChestMNIST':>18s} {'NIH-28':>18s}")
    for m in ORDER:
        p = ch[m]["params"] / 1e6
        mac = fl[m]["mmac"]
        lat = fl[m]["lat_b256_ms"]
        cm = f"{ch[m]['test_mean']:.4f}+-{ch[m]['test_std']:.4f}"
        nh = f"{nih[m]['test_mean']:.4f}+-{nih[m]['test_std']:.4f}"
        log(f"{PRETTY[m]:14s} {p:9.2f} {mac:8.0f} {lat:8.1f} {cm:>18s} {nh:>18s}")
    d = ch["searched_nonrec_pm"]["test_mean"] - ch["searched_rec"]["test_mean"]
    log(f"\nChestMNIST matched single-pass minus recursive: {d:+.4f}  (paper +0.030)")
    d2 = ch["searched_nonrec"]["test_mean"] - ch["searched_nonrec_pm"]["test_mean"]
    log(f"ChestMNIST full CASP minus matched control:    {d2:+.4f}  (paper 0.0049)")


# ---------------------------------------------------------------- bootstrap
def seedmean_auc(Pstack, Y, idx):
    Ps = Pstack[:, idx, :]
    S, n, C = Ps.shape
    order = np.argsort(Ps, axis=1)
    ranks = np.empty_like(Ps)
    rows = np.arange(1, n + 1, dtype=np.float64)[None, :, None]
    np.put_along_axis(ranks, order, np.broadcast_to(rows, Ps.shape).copy(), axis=1)
    Ys = Y[idx]
    npos = Ys.sum(0).astype(np.float64); nneg = n - npos
    sr = (ranks * Ys[None, :, :]).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        auc = (sr - npos * (npos + 1) / 2.0) / (npos * nneg)
    valid = (npos > 0) & (nneg > 0)
    return np.nanmean(np.where(valid[None, :], auc, np.nan), axis=1)


def bootstrap(data, lab, pid, fast=False):
    log("\n" + "=" * 78)
    log("PATIENT-CLUSTERED BOOTSTRAP on NIH  (per-seed-mean statistic, fixed rng=0)")
    log("=" * 78)
    point = {m: float(seedmean_auc(data[m], lab, np.arange(len(pid))).mean()) for m in ORDER}
    if fast:
        log("[--fast] skipping B=2000 resampling; showing point AUCs only")
        for m in sorted(ORDER, key=lambda k: -point[k]):
            log(f"  {PRETTY[m]:14s} {point[m]:.4f}")
        return point
    upat = np.unique(pid)
    idx_by = [np.where(pid == p)[0] for p in upat]
    rng = np.random.default_rng(0)
    boot = {m: np.empty(B) for m in ORDER}
    for b in range(B):
        s = rng.integers(0, len(upat), size=len(upat))
        idx = np.concatenate([idx_by[j] for j in s])
        for m in ORDER:
            boot[m][b] = seedmean_auc(data[m], lab, idx).mean()
    for m in sorted(ORDER, key=lambda k: -point[k]):
        lo, hi = np.percentile(boot[m], [2.5, 97.5])
        log(f"  {PRETTY[m]:14s} {point[m]:.4f}  [{lo:.4f}, {hi:.4f}]")
    log("\n  paired deltas (CASP-family vs comparators):")
    for a, b in [("searched_nonrec", "searched_rec"), ("searched_nonrec_pm", "searched_rec"),
                 ("searched_nonrec", "resnet18"), ("searched_nonrec", "densenet121")]:
        d = boot[a] - boot[b]
        lo, hi = np.percentile(d, [2.5, 97.5])
        frac = float(np.mean(d > 0)); p = max(2.0 * min(frac, 1 - frac), 1.0 / B)
        log(f"    {PRETTY[a]:14s} - {PRETTY[b]:14s} {point[a]-point[b]:+.4f} "
            f"[{lo:+.4f},{hi:+.4f}] p={p:.4f}")
    return point


# ---------------------------------------------------- ECE / AP / micro / Brier
def ece_1d(p, y, nb=15):
    bins = np.linspace(0, 1, nb + 1); e = 0.0; n = len(y)
    for i in range(nb):
        m = (p >= bins[i]) & (p < bins[i + 1]) if i < nb - 1 else (p >= bins[i]) & (p <= bins[i + 1])
        k = int(m.sum())
        if k: e += abs(y[m].mean() - p[m].mean()) * k / n
    return e


def calibration(data, lab):
    log("\n" + "=" * 78)
    log("CALIBRATION & SECONDARY METRICS (5-seed-averaged NIH probabilities)")
    log("=" * 78)
    Y = lab
    for m in ["searched_nonrec", "searched_nonrec_pm", "searched_rec"]:
        Pr = data[m].mean(0)
        valid = [c for c in range(14) if Y[:, c].sum() > 0]
        macro_ap = float(np.mean([average_precision_score(Y[:, c], Pr[:, c]) for c in valid]))
        micro_auc = float(roc_auc_score(Y.ravel(), Pr.ravel()))
        micro_ap = float(average_precision_score(Y.ravel(), Pr.ravel()))
        brier = float(np.mean((Pr - Y) ** 2))
        per = [ece_1d(Pr[:, c], Y[:, c]) for c in range(14)]
        ece_lab = float(np.mean(per)); ece_pool = ece_1d(Pr.ravel(), Y.ravel())
        log(f"  {PRETTY[m]:14s} macroAP={macro_ap:.4f} microAUC={micro_auc:.4f} "
            f"microAP={micro_ap:.4f} Brier={brier:.4f} ECE(lab/pool)={ece_lab:.4f}/{ece_pool:.4f}")
    log("  paper: macroAP 0.19/0.18/0.17 (matched/CASP/rec), microAUC 0.82, microAP 0.30, Brier 0.061,")
    log("         CASP ECE 0.012 per-label / 0.009 pooled")


# ---------------------------------------------------------------- ablation/cycles
def ablation_and_cycles():
    ab = jload("bench_ablation.json")
    log("\n" + "=" * 78)
    log("TABLE V  Ablation of CASP on ChestMNIST (3 seeds, vs full-3-seed baseline 0.7815)")
    log("=" * 78)
    base = 0.7815  # full CASP seeds 0,1,2 (see bench_results.json searched_nonrec seeds[:3])
    names = {"casp_no_ws": "- weight standardization", "casp_mean_pool": "- multi-query pooling",
             "casp_std_down": "- pixel-unshuffle stem", "casp_no_se": "- squeeze-and-excite",
             "casp_no_local": "- local token mixing"}
    log(f"  {'Full CASP (3 seeds)':28s} {base:.4f}   --")
    for k in ["casp_no_ws", "casp_mean_pool", "casp_std_down", "casp_no_se", "casp_no_local"]:
        log(f"  {names[k]:28s} {ab[k]['test_mean']:.4f}   {ab[k]['test_mean']-base:+.4f}")
    rc = jload("rec_percycle.json")
    log("\n" + "=" * 78)
    log("TABLE IV  Recursive per-cycle diagnostic (seed 0)")
    log("=" * 78)
    log(f"  {'cycle':>5s} {'macroAUC':>9s} {'L2 change':>10s} {'cos-to-prev':>11s}")
    for r in rc["rows"]:
        l2 = "--" if r["mean_state_change_l2"] is None else f"{r['mean_state_change_l2']:.2f}"
        cs = "--" if r["mean_cosine_to_prev"] is None else f"{r['mean_cosine_to_prev']:.3f}"
        log(f"  {r['cycle']:>5d} {r['macro_auc']:>9.4f} {l2:>10s} {cs:>11s}")


def main():
    fast = "--fast" in sys.argv
    data = {}; lab = pid = None
    for m in ORDER:
        data[m], lab, pid = load(m)
    log(f"Loaded NIH test outputs: {len(pid)} images / {len(np.unique(pid))} patients, "
        f"5 seeds x 8 models\n")
    table_main()
    bootstrap(data, lab, pid, fast=fast)
    calibration(data, lab)
    ablation_and_cycles()
    (RESULTS / "reproduced_paper_numbers.txt").write_text("\n".join(_out), encoding="utf-8")
    log(f"\nWROTE {RESULTS/'reproduced_paper_numbers.txt'}")


if __name__ == "__main__":
    main()
