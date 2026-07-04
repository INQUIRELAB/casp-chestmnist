"""Publication-quality statistical figures for the TMI paper, from the real results.
Reads bench_results.json (ChestMNIST), bench_nih_results.json (NIH), nih_bootstrap.json
(patient-clustered CIs + paired deltas), flops_latency.json, and nih_probs/ (per-label AUC,
calibration). Saves PNGs into ./figs/."""
import json
import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

B = Path(__file__).resolve().parent
FIG = B / "figs"
FIG.mkdir(exist_ok=True)
PROB = B / "nih_probs"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11, "axes.linewidth": 0.9,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 220, "savefig.dpi": 220, "savefig.bbox": "tight",
    "axes.grid": True, "grid.alpha": 0.22, "grid.linewidth": 0.6, "grid.color": "#9aa0a6",
    "xtick.direction": "out", "ytick.direction": "out",
})

cm = json.load(open(B / "bench_results.json"))
nih = json.load(open(B / "bench_nih_results.json"))
boot = json.load(open(B / "nih_bootstrap.json"))
fl = json.load(open(B / "flops_latency.json"))

ORDER = ["searched_nonrec", "searched_nonrec_pm", "densenet121", "resnet18",
         "searched_rec", "resnet50", "vit_small", "mobilenet_v2"]
DISP = {"searched_nonrec": "CASP", "searched_nonrec_pm": "CASP-matched",
        "densenet121": "DenseNet-121", "resnet18": "ResNet-18", "searched_rec": "Recursive",
        "resnet50": "ResNet-50", "vit_small": "Compact ViT", "mobilenet_v2": "MobileNetV2"}
C_CASP, C_MATCH, C_REC, C_BB = "#1b6ca8", "#5aa9dd", "#e08214", "#9aa0a6"


def col(m):
    return {"searched_nonrec": C_CASP, "searched_nonrec_pm": C_MATCH, "searched_rec": C_REC}.get(m, C_BB)


# ---------- Fig 1: ChestMNIST benchmark bars with 5-seed SD ----------
fig, ax = plt.subplots(figsize=(7.4, 4.2))
names = [DISP[m] for m in ORDER]
means = [cm[m]["test_mean"] for m in ORDER]
sds = [cm[m]["test_std"] for m in ORDER]
cols = [col(m) for m in ORDER]
y = np.arange(len(ORDER))[::-1]
ax.barh(y, means, xerr=sds, color=cols, edgecolor="white", height=0.68,
        error_kw=dict(ecolor="#333", elinewidth=1.1, capsize=3))
ax.axvline(0.768, ls="--", lw=1.2, color="#555")
ax.text(0.768, len(ORDER) - 0.3, " published ResNet-18 (0.768)", color="#555", fontsize=9, va="top")
for yi, m, v, s in zip(y, ORDER, means, sds):
    ax.text(v + s + 0.001, yi, f"{v:.3f}", va="center", fontsize=9.5,
            fontweight="bold" if m == "searched_nonrec" else "normal")
ax.set_yticks(y); ax.set_yticklabels(names)
ax.set_xlim(0.70, 0.80); ax.set_xlabel("ChestMNIST test macro-AUC  (mean $\\pm$ SD, 5 seeds)")
ax.set_title("Matched-protocol benchmark on ChestMNIST", fontweight="bold", loc="left")
fig.savefig(FIG / "fig_benchmark_bars.png"); plt.close(fig)

# ---------- Fig 2: ranking-preserved slope (ChestMNIST -> NIH) ----------
fig, ax = plt.subplots(figsize=(6.6, 5.0))
x0, x1 = 0, 1
for m in ORDER:
    a, b = cm[m]["test_mean"], nih[m]["test_mean"]
    lw = 3.2 if m in ("searched_nonrec", "searched_nonrec_pm", "searched_rec") else 1.8
    ax.plot([x0, x1], [a, b], "-o", color=col(m), lw=lw, ms=6, zorder=3 if m == "searched_nonrec" else 2)
    ax.text(x0 - 0.03, a, f"{DISP[m]} ", ha="right", va="center", fontsize=9.5,
            color=col(m) if col(m) != C_BB else "#444",
            fontweight="bold" if m == "searched_nonrec" else "normal")
    ax.text(x1 + 0.03, b, f" {b:.3f}", ha="left", va="center", fontsize=9, color=col(m) if col(m) != C_BB else "#444")
ax.set_xticks([x0, x1]); ax.set_xticklabels(["ChestMNIST\n(MedMNIST split)", "NIH ChestX-ray14\n(patient-grouped)"])
ax.set_xlim(-0.42, 1.32); ax.set_ylabel("test macro-AUC")
ax.set_title("External validation on patient-grouped NIH", fontweight="bold", loc="left")
ax.grid(axis="x", visible=False)
fig.savefig(FIG / "fig_ranking_preserved.png"); plt.close(fig)

# ---------- Fig 3: forest plot of patient-clustered paired deltas ----------
pairs = [("searched_nonrec_pm__vs__searched_rec", "CASP-matched  $-$  Recursive\n(matched parameters)"),
         ("searched_nonrec__vs__searched_rec", "CASP  $-$  Recursive"),
         ("searched_nonrec__vs__resnet18", "CASP  $-$  ResNet-18"),
         ("searched_nonrec__vs__densenet121", "CASP  $-$  DenseNet-121"),
         ("searched_nonrec__vs__searched_nonrec_pm", "CASP  $-$  CASP-matched")]
fig, ax = plt.subplots(figsize=(7.2, 3.9))
yy = np.arange(len(pairs))[::-1]
for yi, (k, lab) in zip(yy, pairs):
    d = boot["paired"][k]
    excl = (d["ci_lo"] > 0) or (d["ci_hi"] < 0)
    c = "#2f7d32" if excl else "#999"
    ax.plot([d["ci_lo"], d["ci_hi"]], [yi, yi], "-", color=c, lw=2.4, solid_capstyle="round")
    ax.plot(d["delta"], yi, "o", color=c, ms=8, zorder=3)
    ax.text(d["ci_hi"] + 0.001, yi, f"  {d['delta']:+.3f} [{d['ci_lo']:+.3f}, {d['ci_hi']:+.3f}]",
            va="center", fontsize=9, color=c)
ax.axvline(0, color="#333", lw=1.1)
ax.set_yticks(yy); ax.set_yticklabels([p[1] for p in pairs], fontsize=9.5)
ax.set_xlim(-0.012, 0.055); ax.set_xlabel("$\\Delta$ macro-AUC (patient-clustered bootstrap, 95% CI)")
ax.set_title("Paired differences on patient-grouped NIH (B = 2000)", fontweight="bold", loc="left")
ax.grid(axis="y", visible=False)
fig.savefig(FIG / "fig_forest.png"); plt.close(fig)

# ---------- Fig 4: accuracy vs parameters (efficiency), marker ~ MACs ----------
fig, ax = plt.subplots(figsize=(7.0, 4.6))
for m in ORDER:
    p = fl[m]["params"] / 1e6
    a = cm[m]["test_mean"]
    mac = fl[m]["mmac"]
    ax.scatter(p, a, s=28 + mac * 0.9, color=col(m), edgecolor="white", linewidth=1.2, zorder=3,
               alpha=0.92)
    dx = 1.06 if m not in ("searched_nonrec_pm",) else 0.9
    ax.annotate(DISP[m], (p, a), xytext=(p * dx, a + 0.0016), fontsize=9.5,
                fontweight="bold" if m == "searched_nonrec" else "normal",
                color=col(m) if col(m) != C_BB else "#444")
ax.axhline(0.768, ls="--", lw=1.0, color="#888")
ax.text(1.05, 0.7685, "published ResNet-18", color="#888", fontsize=8.5)
ax.set_xscale("log"); ax.set_xlabel("parameters (millions, log scale)")
ax.set_ylabel("ChestMNIST test macro-AUC")
ax.set_title("Accuracy vs. size  (marker area $\\propto$ forward MACs)", fontweight="bold", loc="left")
ax.set_xlim(1.2, 30)
fig.savefig(FIG / "fig_efficiency.png"); plt.close(fig)

# ---------- per-label + calibration helpers on CASP NIH probs ----------
LABELS = ["Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule",
          "Pneumonia", "Pneumothorax", "Consolidation", "Edema", "Emphysema", "Fibrosis",
          "Pleural thick.", "Hernia"]


def load_probs(model):
    fs = sorted(PROB.glob(f"{model}_seed*.npz"))
    Ps, lab, pid = [], None, None
    for f in fs:
        z = np.load(f)
        Ps.append(1.0 / (1.0 + np.exp(-z["test_logits"].astype(np.float64))))
        lab, pid = z["test_labels"], z["test_patient"]
    return np.mean(Ps, 0), lab.astype(int), pid.astype(np.int64)


def auc_col(p, y):
    npos = int(y.sum()); nneg = len(y) - npos
    if npos == 0 or nneg == 0:
        return np.nan
    r = np.empty(len(y)); r[np.argsort(p)] = np.arange(1, len(y) + 1)
    return (r[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg)


P, Y, PID = load_probs("searched_nonrec")
prev = Y.mean(0)
o = np.argsort(-prev)  # order by prevalence desc
# patient-clustered bootstrap for per-label CI
upat = np.unique(PID)
idx_by = [np.where(PID == pp)[0] for pp in upat]
rng = np.random.default_rng(0)
Bb = 800
boots = np.zeros((Bb, 14))
for b in range(Bb):
    s = rng.integers(0, len(upat), len(upat))
    ix = np.concatenate([idx_by[j] for j in s])
    for c in range(14):
        boots[b, c] = auc_col(P[ix, c], Y[ix, c])
pt = np.array([auc_col(P[:, c], Y[:, c]) for c in range(14)])
lo = np.nanpercentile(boots, 2.5, 0); hi = np.nanpercentile(boots, 97.5, 0)

# ---------- Fig 5: per-label AUC with CI (CASP, NIH) ----------
fig, ax = plt.subplots(figsize=(7.2, 4.6))
yy = np.arange(14)[::-1]
for k, c in enumerate(o):
    ax.plot([lo[c], hi[c]], [yy[k], yy[k]], "-", color=C_CASP, lw=2.0, alpha=0.55, solid_capstyle="round")
    ax.plot(pt[c], yy[k], "o", color=C_CASP, ms=6, zorder=3)
ax.axvline(0.5, ls="--", color="#888", lw=1.0)
ax.text(0.5, 13.6, " chance", color="#888", fontsize=8.5, va="top")
ax.set_yticks(yy); ax.set_yticklabels([f"{LABELS[c]}  ({prev[c]*100:.1f}%)" for c in o], fontsize=9)
ax.set_xlim(0.45, 0.92); ax.set_xlabel("per-label test AUC (patient-clustered 95% CI)")
ax.set_title("CASP per-label AUC on patient-grouped NIH (by prevalence)", fontweight="bold", loc="left")
ax.grid(axis="y", visible=False)
fig.savefig(FIG / "fig_perlabel.png"); plt.close(fig)

# ---------- Fig 6: calibration reliability (CASP, NIH pooled) ----------
pp = P.ravel(); yy2 = Y.ravel()
nb = 12
bins = np.linspace(0, 1, nb + 1)
xs, ys, ws = [], [], []
for i in range(nb):
    m = (pp >= bins[i]) & (pp < bins[i + 1]) if i < nb - 1 else (pp >= bins[i]) & (pp <= bins[i + 1])
    if m.sum() == 0:
        continue
    xs.append(pp[m].mean()); ys.append(yy2[m].mean()); ws.append(m.sum())
ece = sum(abs(a - b) * w for a, b, w in zip(ys, xs, ws)) / len(pp)
fig, ax = plt.subplots(figsize=(5.2, 5.0))
ax.plot([0, 1], [0, 1], "--", color="#888", lw=1.1, label="perfect calibration")
ax.plot(xs, ys, "-o", color=C_CASP, lw=2.0, ms=6, label="CASP")
ax.set_xlabel("mean predicted probability"); ax.set_ylabel("observed frequency")
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.text(0.04, 0.9, f"ECE = {ece:.3f}", fontsize=11, fontweight="bold", color=C_CASP)
ax.set_title("Reliability diagram (CASP, NIH test)", fontweight="bold", loc="left")
ax.legend(loc="lower right", frameon=False, fontsize=9)
ax.set_aspect("equal")
fig.savefig(FIG / "fig_calibration.png"); plt.close(fig)

print("saved:", sorted(p.name for p in FIG.glob("*.png")), flush=True)
print(f"ECE(pooled)={ece:.4f}", flush=True)
