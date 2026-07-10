# CASP: A Compact Convolutional-Attention Classifier for Chest Radiographs

Code for the paper *Compact Chest-Radiograph Classification: Can a Single Convolutional-Attention Pass Outperform Recursive Refinement?*

CASP is a compact single-pass convolutional-attention classifier. This repository releases it together with a parameter-matched recursive (TRM-style) latent-refinement baseline and a matched-protocol training harness, so the two designs can be compared directly on ChestMNIST and on patient-grouped NIH ChestX-ray14.

## Key result

All eight models are trained under one identical protocol (AdamW, 5% warmup then cosine, 4000 steps, best-validation checkpoint, per-model learning-rate selection on seed 0, five seeds, official test split evaluated once). CASP leads at a fraction of the parameters, and a single convolutional-attention pass beats iterated recursive refinement at matched parameters.

| Model | Params (M) | MACs (M) | ChestMNIST AUC | NIH-28 AUC |
|---|---:|---:|:---:|:---:|
| **CASP** (single pass) | **1.90** | 167 | **0.7826 ± 0.0019** | 0.7288 |
| CASP-matched (control) | 1.46 | 129 | 0.7777 ± 0.0024 | **0.7334** |
| DenseNet-121 | 6.96 | 680 | 0.7631 ± 0.0043 | 0.7158 |
| ResNet-18 | 11.17 | 457 | 0.7545 ± 0.0026 | 0.7039 |
| Recursive (TRM-style) | 1.49 | 293 | 0.7480 ± 0.0015 | 0.6914 |
| ResNet-50 | 23.53 | 1052 | 0.7454 ± 0.0021 | 0.6913 |
| Compact ViT | 1.80 | 89 | 0.7231 ± 0.0036 | 0.6731 |
| MobileNetV2 | 2.24 | 23 | 0.7167 ± 0.0041 | 0.6684 |

On patient-grouped NIH (patient-clustered bootstrap, B = 2000), a single-pass attention model exceeds recursion at matched parameters by +0.039 macro-AUC (95% CI [+0.034, +0.044], p < 5e-4).

*ChestMNIST is a downsampled research benchmark, so these are architectural comparisons and not clinical performance claims.*

## Repository layout

```
best_nonrec.py        CASP: compact single-pass convolutional-attention classifier
best_rec.py           Recursive (TRM-style) latent-refinement baseline
bench_chestmnist.py   Matched-protocol head-to-head on ChestMNIST (all 8 models)
bench_nih.py          Patient-grouped external validation on NIH ChestX-ray14 (28x28)
nih_data.py           Builds the patient-disjoint NIH-28 splits from ChestX-ray14
nih_bootstrap.py      Patient-clustered bootstrap (point AUC, 95% CI, paired deltas)
ece.py                Expected calibration error (per-label mean and pooled)
flops.py              Parameters, forward MACs, and latency for every model
plots.py              Regenerates the figures from the result files
trm_chest/            Data loaders and the macro-AUC metric
*_results.json        Result files behind the reported numbers
nih_probs/            Per-patient prediction probabilities (8 models x 5 seeds)
logs/                 Per-seed training logs and a per-seed results summary
```

Both models expose the same interface:

```python
from best_nonrec import build_model      # CASP
net = build_model(num_classes=14, in_ch=1)
logits = net(x)                           # x: [B, 1, 28, 28] -> [B, 14]
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

ChestMNIST and PneumoniaMNIST download automatically through the `medmnist` package on first use. NIH ChestX-ray14, used for external validation, must be obtained from its official source; `nih_data.py` then builds the patient-disjoint 28x28 splits.

## Running the code

```bash
# Matched-protocol ChestMNIST benchmark (all 8 models, 5 seeds)
python bench_chestmnist.py

# Patient-grouped NIH external validation (saves per-patient probabilities)
python nih_data.py
python bench_nih.py

# Statistics, calibration, and cost
python nih_bootstrap.py
python ece.py
python flops.py

# Figures
python plots.py
```

Each script is deterministic given the fixed seeds, so the committed `*_results.json` and `nih_probs/` regenerate the tables, statistics, and figures without retraining.

## License

Released under the MIT License (see `LICENSE`).

## Citation

A BibTeX entry will be added with the camera-ready version of the paper.
