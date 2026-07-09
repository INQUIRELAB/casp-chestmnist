"""Publication-quality statistical figures for the TMI paper, from the real results.
Reads bench_results.json (ChestMNIST), bench_nih_results.json (NIH), nih_bootstrap.json
(patient-clustered CIs + paired deltas), flops_latency.json, and nih_probs/ (per-label AUC,
calibration). Saves PNGs into ~/bench/figs/."""
import json
import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

B = Path(__file__).resolve().parent
FIG = B / "figures"
FIG.mkdir(exist_ok=True)
PROB = B / "nih_probs"

# All six figures are placed at the journal COLUMN width in the 2-column IEEE
# (IEEEtran) layout. The paper includes each at \columnwidth, so the native
# figure width MUST equal that column width (3.5 in) for a 1:1 mapping with NO
# LaTeX downscaling. That way the 8 pt figure text set here renders as 8 pt on
# the page, matching the paper's 10 pt Times body / 8 pt figure text.
FIGW = 3.5  # journal single-column width, inches

plt.rcParams.update({
    # --- fonts: serif to match IEEEtran (Times), STIX math to match ---
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "pdf.fonttype": 42,  # embed TrueType so PDF text stays editable/selectable
    # --- spines / grid (unchanged style) ---
    "axes.linewidth": 0.9,
    "axes.spines.top": False, "axes.spines.right": False,
    # Fixed native width: keep the figure exactly FIGW wide (do NOT use
    # bbox="tight", which would grow the media box past FIGW to fit long y-tick
    # labels and then get downscaled by LaTeX at \columnwidth, shrinking the
    # font). constrained_layout instead fits all content *inside* FIGW.
    "figure.dpi": 220, "savefig.dpi": 400,
    "figure.constrained_layout.use": True,
    "figure.constrained_layout.h_pad": 0.02, "figure.constrained_layout.w_pad": 0.02,
    "figure.constrained_layout.hspace": 0.0, "figure.constrained_layout.wspace": 0.0,
    "axes.grid": True, "grid.alpha": 0.22, "grid.linewidth": 0.6, "grid.color": "#9aa0a6",
    "xtick.direction": "out", "ytick.direction": "out",
})


def save(fig, stem):
    """Save both a PDF (used by the paper) and a PNG (for inspection)."""
    fig.savefig(FIG / f"{stem}.pdf")
    fig.savefig(FIG / f"{stem}.png")
    plt.close(fig)

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
fig, ax = plt.subplots(figsize=(FIGW, 2.7))
names = [DISP[m] for m in ORDER]
means = [cm[m]["test_mean"] for m in ORDER]
sds = [cm[m]["test_std"] for m in ORDER]
cols = [col(m) for m in ORDER]
y = np.arange(len(ORDER))[::-1]
ax.barh(y, means, xerr=sds, color=cols, edgecolor="white", height=0.68,
        error_kw=dict(ecolor="#333", elinewidth=0.9, capsize=2))
for yi, m, v, s in zip(y, ORDER, means, sds):
    ax.text(v + s + 0.0015, yi, f"{v:.3f}", va="center", fontsize=7,
            fontweight="bold" if m == "searched_nonrec" else "normal")
ax.set_yticks(y); ax.set_yticklabels(names)
ax.set_xlim(0.70, 0.805); ax.set_xlabel("ChestMNIST test macro-AUC  (mean $\\pm$ SD, 5 seeds)")
ax.set_title("Matched-protocol benchmark on ChestMNIST", fontweight="bold", loc="left")
save(fig, "fig_benchmark_bars")

# ---------- Fig 2: ranking-preserved slope (ChestMNIST -> NIH) ----------
def dodge(vals, min_gap, lo=None, hi=None):
    """Return adjusted positions (same order as vals) so that, sorted, no two are
    closer than min_gap. Spreads collisions symmetrically around their mean, then
    (if lo/hi given) clamps the whole block inside [lo, hi] without re-colliding."""
    vals = np.array(vals, float)
    order = np.argsort(vals)
    ys = vals[order].copy()
    # forward pass: enforce min_gap going up
    for i in range(1, len(ys)):
        if ys[i] - ys[i - 1] < min_gap:
            ys[i] = ys[i - 1] + min_gap
    # recentre the whole block on the original mean so it is not net-shifted
    ys += vals[order].mean() - ys.mean()
    # clamp inside axis limits, preserving the fixed spacing
    if lo is not None and ys[0] < lo:
        ys += lo - ys[0]
    if hi is not None and ys[-1] > hi:
        ys += hi - ys[-1]
    out = np.empty_like(ys)
    out[order] = ys
    return out


fig, ax = plt.subplots(figsize=(FIGW, 4.3))
x0, x1 = 0, 1
a_vals = np.array([cm[m]["test_mean"] for m in ORDER])
b_vals = np.array([nih[m]["test_mean"] for m in ORDER])
ylo = min(b_vals.min(), a_vals.min()) - 0.008
yhi = max(a_vals.max(), b_vals.max()) + 0.012
ax.set_ylim(ylo, yhi)
# min vertical label separation = one 7.5 pt line-box, converted to data units
fig.canvas.draw()
px_per_data = ax.transData.transform((0, 1))[1] - ax.transData.transform((0, 0))[1]
GAP = (7.5 * 1.45 / 72.0 * fig.dpi) / px_per_data  # ~1.45 line-heights, in data units
# keep labels inside the axes with a small margin
mlo, mhi = ylo + GAP * 0.55, yhi - GAP * 0.55
a_lab = dodge(a_vals, GAP, mlo, mhi)
b_lab = dodge(b_vals, GAP, mlo, mhi)
for i, m in enumerate(ORDER):
    a, b = a_vals[i], b_vals[i]
    c = col(m)
    lw = 2.6 if m in ("searched_nonrec", "searched_nonrec_pm", "searched_rec") else 1.5
    ax.plot([x0, x1], [a, b], "-o", color=c, lw=lw, ms=4.5, zorder=3 if m == "searched_nonrec" else 2)
    lc = c if c != C_BB else "#444"
    # left label (model name), dodged to avoid overlap, thin leader to its point
    ax.annotate(f"{DISP[m]} ", xy=(x0, a), xytext=(x0 - 0.055, a_lab[i]),
                ha="right", va="center", fontsize=7.5, color=lc,
                fontweight="bold" if m == "searched_nonrec" else "normal",
                arrowprops=dict(arrowstyle="-", color=lc, lw=0.6,
                                shrinkA=1.5, shrinkB=2, alpha=0.6)
                if abs(a_lab[i] - a) > 1e-4 else None)
    # right label (NIH value), dodged, thin leader to its point
    ax.annotate(f" {b:.3f}", xy=(x1, b), xytext=(x1 + 0.055, b_lab[i]),
                ha="left", va="center", fontsize=7.5, color=lc,
                arrowprops=dict(arrowstyle="-", color=lc, lw=0.6,
                                shrinkA=1.5, shrinkB=2, alpha=0.6)
                if abs(b_lab[i] - b) > 1e-4 else None)
ax.set_xticks([x0, x1]); ax.set_xticklabels(["ChestMNIST\n(MedMNIST split)", "NIH ChestX-ray14\n(patient-grouped)"])
ax.set_xlim(-0.52, 1.42); ax.set_ylabel("test macro-AUC")
ax.set_title("Patient-disjoint validation on NIH", fontweight="bold", loc="left", pad=8)
ax.grid(axis="x", visible=False)
save(fig, "fig_ranking_preserved")

# ---------- Fig 3: forest plot of patient-clustered paired deltas ----------
pairs = [("searched_nonrec_pm__vs__searched_rec", "CASP-matched  $-$  Recursive\n(matched parameters)"),
         ("searched_nonrec__vs__searched_rec", "CASP  $-$  Recursive"),
         ("searched_nonrec__vs__resnet18", "CASP  $-$  ResNet-18"),
         ("searched_nonrec__vs__densenet121", "CASP  $-$  DenseNet-121"),
         ("searched_nonrec__vs__searched_nonrec_pm", "CASP  $-$  CASP-matched")]
fig, ax = plt.subplots(figsize=(FIGW, 2.9))
yy = np.arange(len(pairs))[::-1]
for yi, (k, lab) in zip(yy, pairs):
    d = boot["paired"][k]
    excl = (d["ci_lo"] > 0) or (d["ci_hi"] < 0)
    c = "#2f7d32" if excl else "#999"
    ax.plot([d["ci_lo"], d["ci_hi"]], [yi, yi], "-", color=c, lw=2.0, solid_capstyle="round")
    ax.plot(d["delta"], yi, "o", color=c, ms=6, zorder=3)
    # right-aligned at the axis right edge so the label always ends INSIDE the frame
    ax.text(0.0955, yi, f"{d['delta']:+.3f} [{d['ci_lo']:+.3f}, {d['ci_hi']:+.3f}]",
            va="center", ha="right", fontsize=6.5, color=c)
ax.axvline(0, color="#333", lw=1.0)
ax.set_yticks(yy); ax.set_yticklabels([p[1] for p in pairs], fontsize=7.5)
ax.set_xlim(-0.012, 0.098)
# only tick the CI region; the right band is reserved for the numeric labels
ax.set_xticks([0.0, 0.02, 0.04])
ax.set_xlabel("$\\Delta$ macro-AUC (patient-clustered 95% CI)")
# suptitle lives in FIGURE space (constrained_layout reserves room for it), so a
# wide left y-label column cannot push it off-frame like an axes-left title.
fig.suptitle("Paired differences on patient-disjoint NIH ($B = 2000$)",
             fontweight="bold", fontsize=8.5, x=0.012, ha="left")
ax.grid(axis="y", visible=False)
save(fig, "fig_forest")

# ---------- Fig 4: accuracy vs parameters (efficiency), marker ~ MACs ----------
fig, ax = plt.subplots(figsize=(FIGW, 3.2))
# per-model label offsets in points (dx, dy) + text anchor, tuned so no label
# overlaps its marker, another label, or another marker. Thin leaders connect
# each label back to its point.
LOFF = {
    "searched_nonrec":    (8, 7, "left"),     # CASP  -> up-right
    "searched_nonrec_pm": (-6, 7, "right"),   # CASP-matched -> up-left (clears CASP marker)
    "searched_rec":       (8, -1, "left"),    # Recursive -> right
    "vit_small":          (8, 6, "left"),     # Compact ViT -> up-right
    "mobilenet_v2":       (8, -9, "left"),    # MobileNetV2 -> down-right
    "densenet121":        (9, 3, "left"),     # DenseNet-121 -> right
    "resnet18":           (9, 3, "left"),     # ResNet-18 -> right
    "resnet50":           (-9, -12, "right"), # ResNet-50 -> down-left (near right edge)
}
for m in ORDER:
    p = fl[m]["params"] / 1e6
    a = cm[m]["test_mean"]
    mac = fl[m]["mmac"]
    ax.scatter(p, a, s=22 + mac * 0.55, color=col(m), edgecolor="white", linewidth=1.0,
               zorder=3, alpha=0.92)
    dx, dy, ha = LOFF[m]
    lc = col(m) if col(m) != C_BB else "#444"
    ax.annotate(DISP[m], (p, a), xytext=(dx, dy), textcoords="offset points",
                ha=ha, va="center", fontsize=7, color=lc,
                fontweight="bold" if m == "searched_nonrec" else "normal",
                arrowprops=dict(arrowstyle="-", color=lc, lw=0.5, shrinkA=1, shrinkB=3,
                                alpha=0.55))
ax.set_xscale("log"); ax.set_xlabel("parameters (millions, log scale)")
ax.set_ylabel("ChestMNIST test macro-AUC")
ax.set_title("Accuracy vs. size  (marker area $\\propto$ forward MACs)", fontweight="bold", loc="left")
ax.set_xlim(0.98, 34)
# vertical headroom so top/bottom labels clear the frame
ymin, ymax = ax.get_ylim()
ax.set_ylim(ymin - 0.004, ymax + 0.008)
save(fig, "fig_efficiency")

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
fig, ax = plt.subplots(figsize=(FIGW, 4.2))
yy = np.arange(14)[::-1]
for k, c in enumerate(o):
    ax.plot([lo[c], hi[c]], [yy[k], yy[k]], "-", color=C_CASP, lw=1.6, alpha=0.55, solid_capstyle="round")
    ax.plot(pt[c], yy[k], "o", color=C_CASP, ms=4.5, zorder=3)
ax.axvline(0.5, ls="--", color="#888", lw=1.0)
ax.text(0.5, 13.7, " chance", color="#888", fontsize=7, va="top")
ax.set_yticks(yy); ax.set_yticklabels([f"{LABELS[c]}  ({prev[c]*100:.1f}%)" for c in o], fontsize=7.5)
ax.set_ylim(-0.6, 14.0)
ax.set_xlim(0.45, 0.92); ax.set_xlabel("per-label test AUC (patient-clustered 95% CI)")
# suptitle in FIGURE space so the wide left label column can't push it off-frame
fig.suptitle("CASP per-label AUC on patient-disjoint NIH", fontweight="bold",
             fontsize=8.5, x=0.012, ha="left")
ax.grid(axis="y", visible=False)
save(fig, "fig_perlabel")

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
fig, ax = plt.subplots(figsize=(FIGW, 3.5))
ax.plot([0, 1], [0, 1], "--", color="#888", lw=1.1, label="perfect calibration")
ax.plot(xs, ys, "-o", color=C_CASP, lw=1.8, ms=4.5, label="CASP")
ax.set_xlabel("mean predicted probability"); ax.set_ylabel("observed frequency")
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.text(0.04, 0.9, f"ECE = {ece:.3f}", fontsize=8, fontweight="bold", color=C_CASP)
ax.set_title("Reliability diagram (CASP, NIH test)", fontweight="bold", loc="left")
ax.legend(loc="lower right", frameon=False, fontsize=7.5)
ax.set_aspect("equal")
save(fig, "fig_calibration")

print("saved:", sorted(p.name for p in FIG.glob("*.png")), flush=True)
print(f"ECE(pooled)={ece:.4f}", flush=True)
