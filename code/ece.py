"""Expected calibration error for the headline model on the patient-grouped NIH test set,
from the saved 5-seed test probabilities (seed-averaged). Per-label 15-bin ECE, averaged
over the 14 labels, plus the pooled ECE."""
import os
from pathlib import Path

import numpy as np

P = Path(__file__).resolve().parent / "nih_probs"


def load(model):
    fs = sorted(P.glob(f"{model}_seed*.npz"))
    probs = []
    lab = None
    for f in fs:
        z = np.load(f)
        probs.append(1.0 / (1.0 + np.exp(-z["test_logits"].astype(np.float64))))
        lab = z["test_labels"]
    return np.mean(probs, 0), lab.astype(int)


def ece(p, y, nb=15):
    bins = np.linspace(0, 1, nb + 1)
    e = 0.0
    n = len(y)
    for i in range(nb):
        if i < nb - 1:
            m = (p >= bins[i]) & (p < bins[i + 1])
        else:
            m = (p >= bins[i]) & (p <= bins[i + 1])
        k = int(m.sum())
        if k == 0:
            continue
        e += abs(y[m].mean() - p[m].mean()) * k / n
    return e


for model in ["searched_nonrec", "searched_rec"]:
    Pr, Y = load(model)
    per = [ece(Pr[:, c], Y[:, c]) for c in range(14)]
    print(f"{model}: ECE per-label mean = {np.mean(per):.4f}   pooled = {ece(Pr.ravel(), Y.ravel()):.4f}", flush=True)
