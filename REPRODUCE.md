# Reproducing the results

This capsule reproduces **every number and figure** in the manuscript
*"Compact Chest-Radiograph Classification: Can a Single Convolutional-Attention
Pass Outperform Recursive Refinement?"*

## One command

```
bash run
```

This runs `reproduce_paper.py` (all tables and statistics), `plots.py` (the six
figures), and `flops.py` (forward MACs), writing everything to `results/`
(`reproduced_paper_numbers.txt` plus the figure PDFs/PNGs).

## What is reproduced, and why it is exact

The reproducible run is **fully deterministic**. It recomputes all reported
statistics from the **saved per-seed model outputs** in `nih_probs/` (40 files:
per-patient prediction probabilities, labels, and patient IDs for 8 models × 5
seeds on the patient-disjoint NIH test set) and the result JSONs:

| Reported quantity | Reproduced by | Determinism |
|---|---|---|
| Table III (macro-AUC, params, MACs) | `reproduce_paper.py`, `flops.py` | exact (fixed arrays) |
| Patient-clustered bootstrap CIs + paired deltas | `reproduce_paper.py`, `nih_bootstrap.py` | exact (`np.random.default_rng(0)`, B=2000) |
| Calibration (ECE), AP, micro-AUC, Brier | `reproduce_paper.py`, `ece.py` | exact (numpy / scikit-learn) |
| Table V ablation, Table IV per-cycle | `reproduce_paper.py` | exact (saved JSONs) |
| Figures 3–8 | `plots.py` | exact |

Because these steps use only numpy / scikit-learn on the fixed saved arrays with
fixed random seeds, they produce **byte-identical numbers on any machine**.

## What is NOT re-run here (and why)

The GPU **training** that produced `nih_probs/` (`bench_chestmnist.py`,
`bench_nih.py`, 5 seeds under the matched protocol) is included for full
transparency but is **not** part of the reproducible run. GPU floating-point is
not bitwise-identical across different GPUs / driver / cuDNN versions, so
retraining on other hardware yields statistically equivalent but not identical
point estimates. The reported numbers are therefore reproduced from the saved
model outputs, which is the standard for such studies. The **latency** column is
likewise hardware-dependent (measured on an NVIDIA H200); `flops.py` re-measures
it on the capsule hardware for reference, but the forward **MACs** it reports are
exact.

## Full pipeline (optional, requires GPU + datasets)

To retrain from scratch: `python bench_chestmnist.py` and `python bench_nih.py`
(needs a CUDA GPU, the `medmnist` ChestMNIST/PneumoniaMNIST downloads, and the
NIH ChestX-ray14 images). `run_ablation.py` and `run_rec_diag.py` regenerate the
ablation and per-cycle diagnostics.

## Environment

See `requirements.txt`. The reproducible run needs only `numpy`,
`scikit-learn`, and `matplotlib`; `torch`/`torchvision`/`medmnist`/`fvcore` are
needed only for `flops.py` and the optional retraining.
