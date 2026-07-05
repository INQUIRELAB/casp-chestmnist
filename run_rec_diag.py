"""Experiment B: recursive per-cycle diagnostic.

Trains the instrumented recursive model ONCE on ChestMNIST under the matched
protocol (seed 0, AdamW wd 1e-4, 5% warmup+cosine, best-val checkpoint), then on
the official ChestMNIST TEST split computes, per outer cycle c=1..h_cycles:
  (i)  macro-AUC of the cycle-c intermediate answer logits
  (ii) mean L2 norm of the answer-latent CHANGE from cycle c-1 to c
  (iii) mean cosine similarity between consecutive answer-latent states

The training loop below is a copy of bench_chestmnist.train_eval, kept verbatim
except that it returns the trained (best-val) model so the diagnostic can be run
on it. It reuses bench_chestmnist.lr_factor, eval_auc, and the shared data loaders /
metrics, so the protocol is identical to the main benchmark.

By default it does a single LR (1e-3, the LR selected for searched_rec in
bench_results.json). Pass --lr-select to instead LR-select on seed 0 from the
grid exactly like the harness; --lr to force a value.
"""
import argparse
import json
import sys
import time
from itertools import cycle as icycle
from pathlib import Path

import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import bench_chestmnist as BENCH
from trm_chest.data import DataConfig, make_loaders, unpack_batch
from trm_chest.metrics import macro_auc_from_logits
import rec_diag as RD


def train_best_val(build_fn, dataset, steps, seed, lr, eval_every, data_root, batch, nc):
    """Verbatim bench_chestmnist.train_eval loop, but returns the trained best-val
    model (and loaders) instead of only metrics."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tr, va, te = make_loaders(DataConfig(data_root=data_root, batch_size=batch,
                                         num_workers=0, seed=seed, dataset=dataset))
    model = build_fn(nc, 1).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    bce = nn.BCEWithLogitsLoss()
    warm = int(0.05 * steps)
    best_val = -1.0
    best_state = None
    it = icycle(tr)
    t0 = time.time()
    for step in range(steps):
        for g in opt.param_groups:
            g["lr"] = lr * BENCH.lr_factor(step, steps, warm)
        model.train()
        x, y = unpack_batch(next(it), dev)
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            o = model(x)
            loss = bce(o, y)
        if not torch.isfinite(loss):
            raise RuntimeError(f"nonfinite@{step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if (step + 1) % eval_every == 0 or step == steps - 1:
            v = BENCH.eval_auc(model, va, dev)
            if v > best_val:
                best_val = v
                best_state = {k: val.detach().cpu().clone() for k, val in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    test_auc = BENCH.eval_auc(model, te, dev)
    return model, dev, te, dict(val_auc=round(best_val, 4), test_auc=round(test_auc, 4),
                                params=sum(p.numel() for p in model.parameters()),
                                seconds=round(time.time() - t0, 1))


@torch.no_grad()
def per_cycle_diag(model, loader, dev, h_cycles):
    """Accumulate, over the whole test set: per-cycle logits+labels (for AUC) and
    per-cycle running sums for L2 change and cosine sim between consecutive
    answer states."""
    model.eval()
    logits_cat = [[] for _ in range(h_cycles)]
    labels_all = []
    # running sums for change stats: index c in 1..h_cycles-1 measures c vs c-1
    l2_sum = [0.0 for _ in range(h_cycles)]
    cos_sum = [0.0 for _ in range(h_cycles)]
    n_samples = 0
    for batch in loader:
        x, y = unpack_batch(batch, dev)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            lpc, states, _ = model.forward_diag(x)
        bsz = x.shape[0]
        n_samples += bsz
        for c in range(h_cycles):
            logits_cat[c].append(lpc[c].float().cpu().numpy())
        labels_all.append(y.cpu().numpy())
        # state change stats (float32 for numerical stability)
        st = [s.float() for s in states]
        for c in range(1, h_cycles):
            delta = st[c] - st[c - 1]
            l2 = delta.norm(dim=-1)  # [b]
            cs = torch.nn.functional.cosine_similarity(st[c], st[c - 1], dim=-1)  # [b]
            l2_sum[c] += float(l2.sum().item())
            cos_sum[c] += float(cs.sum().item())
    labels = np.concatenate(labels_all, 0)
    rows = []
    for c in range(h_cycles):
        auc = macro_auc_from_logits(np.concatenate(logits_cat[c], 0), labels)
        if c == 0:
            l2_mean = None
            cos_mean = None
        else:
            l2_mean = l2_sum[c] / n_samples
            cos_mean = cos_sum[c] / n_samples
        rows.append(dict(cycle=c + 1, macro_auc=round(float(auc), 4),
                         mean_state_change_l2=(None if l2_mean is None else round(l2_mean, 4)),
                         mean_cosine_to_prev=(None if cos_mean is None else round(cos_mean, 4))))
    return rows, n_samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="chestmnist")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lr-select", action="store_true",
                    help="LR-select on seed 0 from the grid like the harness")
    ap.add_argument("--lrs", default="1e-3,3e-4")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--data-root", default=str(HERE / "data"))
    ap.add_argument("--out", default=str(HERE / "rec_percycle.json"))
    a = ap.parse_args()
    nc = {"chestmnist": 14, "pneumoniamnist": 1}[a.dataset]
    h_cycles = RD.build_model(nc, 1).h_cycles

    lr = a.lr
    if a.lr_select:
        lrs = [float(x) for x in a.lrs.split(",")]
        best_lr, best_v = None, -1.0
        for cand in lrs:
            _, _, _, m = train_best_val(RD.build_model, a.dataset, a.steps, a.seed, cand,
                                        a.eval_every, a.data_root, a.batch, nc)
            print(f"  [lr-select] lr={cand:.0e} -> val={m['val_auc']} test={m['test_auc']}",
                  flush=True)
            if m["val_auc"] > best_v:
                best_v, best_lr = m["val_auc"], cand
        lr = best_lr
        print(f"  [lr-select] chosen lr={lr:.0e}", flush=True)

    model, dev, te, meta = train_best_val(RD.build_model, a.dataset, a.steps, a.seed, lr,
                                          a.eval_every, a.data_root, a.batch, nc)
    print(f"[trained] rec_diag seed{a.seed} lr={lr:.0e} val={meta['val_auc']} "
          f"test={meta['test_auc']} params={meta['params']/1e6:.4f}M "
          f"({meta['seconds']}s)", flush=True)

    rows, n = per_cycle_diag(model, te, dev, h_cycles)
    out = dict(seed=a.seed, lr=lr, steps=a.steps, params=meta["params"],
               final_test_auc=meta["test_auc"], val_auc=meta["val_auc"],
               n_test=n, h_cycles=h_cycles, l_cycles=RD.build_model(nc, 1).l_cycles,
               rows=rows)
    Path(a.out).write_text(json.dumps(out, indent=2), encoding="utf-8")

    print("\n=== RECURSIVE PER-CYCLE DIAGNOSTIC (ChestMNIST test) ===", flush=True)
    print(f"n_test={n}  final_test_auc={meta['test_auc']}  (matches searched_rec last cycle)",
          flush=True)
    print(f"{'cycle':>5s} {'macro_auc':>10s} {'state_dL2':>10s} {'cos_prev':>9s}", flush=True)
    for r in rows:
        l2 = '     -   ' if r['mean_state_change_l2'] is None else f"{r['mean_state_change_l2']:>10.4f}"
        cs = '    -   ' if r['mean_cosine_to_prev'] is None else f"{r['mean_cosine_to_prev']:>9.4f}"
        print(f"{r['cycle']:>5d} {r['macro_auc']:>10.4f} {l2} {cs}", flush=True)


if __name__ == "__main__":
    main()
