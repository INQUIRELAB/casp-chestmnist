"""External validation on patient-grouped NIH ChestX-ray14 (resized to 28x28) for the
eight matched-protocol architectures.

Identical protocol to the ChestMNIST benchmark: AdamW + 5% warmup then cosine, a fixed
step budget with best-validation-checkpoint selection, per-model learning-rate selection
on seed 0 from one shared grid, then five seeds, and the test split evaluated once.

The crucial difference from the MedMNIST ChestMNIST splits is that the NIH splits here are
PATIENT-DISJOINT (official test split; train/validation split made by patient), so no
patient appears in both training and test. This removes the patient leakage that the
MedMNIST splits do not control, making this a genuine external check of the benchmark
ranking. Per-seed test logits are saved with patient IDs for a patient-clustered bootstrap.
"""
import argparse
import importlib.util
import json
import math
import statistics
import sys
import time
from itertools import cycle
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torchvision.models as tvm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from nih_data import make_nih_loaders  # noqa: E402
from trm_chest.metrics import macro_auc_from_logits  # noqa: E402


def load_module(path):
    spec = importlib.util.spec_from_file_location("m_" + Path(path).stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


NONREC = load_module(HERE / "best_nonrec.py")
REC = load_module(HERE / "best_rec.py")


class PNBlock(nn.Module):
    """Pre-norm attention + GELU MLP block for the compact ViT baseline."""

    def __init__(self, h, heads=6, mlp=4.0):
        super().__init__()
        self.n1 = nn.LayerNorm(h)
        self.attn = nn.MultiheadAttention(h, heads, batch_first=True)
        self.n2 = nn.LayerNorm(h)
        mh = int(h * mlp)
        self.mlp = nn.Sequential(nn.Linear(h, mh), nn.GELU(), nn.Linear(mh, h))

    def forward(self, x):
        y = self.n1(x)
        a, _ = self.attn(y, y, y, need_weights=False)
        x = x + a
        return x + self.mlp(self.n2(x))


class SmallViT(nn.Module):
    def __init__(self, nc, in_ch=1, h=192, heads=6, depth=4, patch=4, img=28):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, h, patch, patch)
        n = (img // patch) ** 2
        self.cls = nn.Parameter(torch.zeros(1, 1, h))
        self.pos = nn.Parameter(torch.zeros(1, n + 1, h))
        nn.init.trunc_normal_(self.pos, std=0.02)
        nn.init.trunc_normal_(self.cls, std=0.02)
        self.blocks = nn.ModuleList([PNBlock(h, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(h)
        self.head = nn.Linear(h, nc)

    def forward(self, x):
        b = x.shape[0]
        t = self.proj(x).flatten(2).transpose(1, 2)
        t = torch.cat([self.cls.expand(b, -1, -1), t], 1) + self.pos
        for blk in self.blocks:
            t = blk(t)
        return self.head(self.norm(t[:, 0]))


def searched_nonrec(nc, in_ch=1):
    m = NONREC.build_model(nc, in_ch)
    m.eval_shift_ensemble = False  # single forward pass for a fair comparison
    return m


def searched_nonrec_pm(nc, in_ch=1):
    m = NONREC.ConvAttnClassifier(nc, in_ch=in_ch, h=168, heads=6, blocks=2,
                                  pool_queries=4, drop_path_rate=0.06, eval_shift_ensemble=False)
    return m


def searched_rec(nc, in_ch=1):
    return REC.build_model(nc, in_ch)


# Small-image stems for ImageNet backbones (28x28 is far smaller than the 224 these
# stems assume); identical to the ChestMNIST benchmark so the two tables are comparable.
def resnet18(nc, in_ch=1):
    m = tvm.resnet18(weights=None, num_classes=nc)
    m.conv1 = nn.Conv2d(in_ch, 64, 3, 1, 1, bias=False)
    m.maxpool = nn.Identity()
    return m


def resnet50(nc, in_ch=1):
    m = tvm.resnet50(weights=None, num_classes=nc)
    m.conv1 = nn.Conv2d(in_ch, 64, 3, 1, 1, bias=False)
    m.maxpool = nn.Identity()
    return m


def densenet121(nc, in_ch=1):
    m = tvm.densenet121(weights=None, num_classes=nc)
    m.features.conv0 = nn.Conv2d(in_ch, 64, 3, 1, 1, bias=False)
    m.features.pool0 = nn.Identity()
    return m


def mobilenet_v2(nc, in_ch=1):
    m = tvm.mobilenet_v2(weights=None, num_classes=nc)
    m.features[0][0] = nn.Conv2d(in_ch, 32, 3, 1, 1, bias=False)
    return m


MODELS = {
    "searched_nonrec": searched_nonrec,
    "searched_rec": searched_rec,
    "searched_nonrec_pm": searched_nonrec_pm,
    "resnet18": resnet18,
    "resnet50": resnet50,
    "densenet121": densenet121,
    "mobilenet_v2": mobilenet_v2,
    "vit_small": (lambda nc, in_ch=1: SmallViT(nc, in_ch)),
}


def count_p(m):
    return sum(p.numel() for p in m.parameters())


@torch.no_grad()
def eval_logits(model, loader, dev):
    model.eval()
    L, Y = [], []
    for x, y in loader:
        x = x.to(dev, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            o = model(x)
        L.append(o.float().cpu().numpy())
        Y.append(y.numpy())
    return np.concatenate(L, 0), np.concatenate(Y, 0)


def lr_factor(step, total, warm):
    if step < warm:
        return step / max(1, warm)
    p = (step - warm) / max(1, total - warm)
    return 0.5 * (1 + math.cos(math.pi * p))


def train_eval(build_fn, size, steps, seed, lr, eval_every, batch, nc):
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tr, va, te, te_pat, te_lab, info = make_nih_loaders(size, seed=seed, batch_size=batch,
                                                        limit_train_patients=None, num_workers=0)
    try:
        model = build_fn(nc, 1).to(dev)
    except Exception as e:  # noqa: BLE001
        return dict(ok=False, error=f"build: {e}"), None
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    bce = nn.BCEWithLogitsLoss()
    warm = int(0.05 * steps)
    best_val = -1.0
    best_state = None
    it = cycle(tr)
    t0 = time.time()
    for step in range(steps):
        for g in opt.param_groups:
            g["lr"] = lr * lr_factor(step, steps, warm)
        model.train()
        x, y = next(it)
        x = x.to(dev, non_blocking=True)
        y = y.to(dev, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            o = model(x)
            loss = bce(o, y)
        if not torch.isfinite(loss):
            return dict(ok=False, error=f"nonfinite@{step}"), None
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if (step + 1) % eval_every == 0 or step == steps - 1:
            vl, vy = eval_logits(model, va, dev)
            v = macro_auc_from_logits(vl, vy)
            if v > best_val:
                best_val = v
                best_state = {k: val.detach().cpu().clone() for k, val in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    tl, tlab = eval_logits(model, te, dev)
    t = macro_auc_from_logits(tl, tlab)
    return (dict(ok=True, val_auc=round(best_val, 4), test_auc=round(t, 4),
                 params=count_p(model), seconds=round(time.time() - t0, 1)),
            (tl, tlab, te_pat))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=28)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--eval-every", type=int, default=400)
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--lrs", default="1e-3,3e-4")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--out", default=str(HERE / "bench_nih_results.json"))
    ap.add_argument("--probs", default=str(HERE / "nih_probs"))
    a = ap.parse_args()
    nc = 14
    seeds = [int(s) for s in a.seeds.split(",")]
    lrs = [float(x) for x in a.lrs.split(",")]
    models = [m for m in a.models.split(",") if m in MODELS]
    out = Path(a.out)
    probs = Path(a.probs)
    probs.mkdir(exist_ok=True)
    results = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}

    for name in models:
        if results.get(name, {}).get("done"):
            print(f"[skip] {name} already done", flush=True)
            continue
        bf = MODELS[name]
        lr_runs = {}
        for lr in lrs:
            r, p = train_eval(bf, a.size, a.steps, seeds[0], lr, a.eval_every, a.batch, nc)
            lr_runs[lr] = (r, p)
            print(f"  [{name}] lr={lr:.0e} seed{seeds[0]} -> {r.get('test_auc', r.get('error'))}", flush=True)
        ok_lrs = {lr: rp for lr, rp in lr_runs.items() if rp[0].get("ok")}
        if not ok_lrs:
            results[name] = dict(done=True, ok=False, error="all LRs failed: " + str({lr: rp[0] for lr, rp in lr_runs.items()}))
            out.write_text(json.dumps(results, indent=2), encoding="utf-8")
            continue
        best_lr = max(ok_lrs, key=lambda lr: ok_lrs[lr][0]["val_auc"])
        r0, p0 = ok_lrs[best_lr]
        runs = [r0]
        if p0 is not None:
            np.savez_compressed(probs / f"{name}_seed{seeds[0]}.npz", test_logits=p0[0], test_labels=p0[1], test_patient=p0[2])
        for s in seeds[1:]:
            r, p = train_eval(bf, a.size, a.steps, s, best_lr, a.eval_every, a.batch, nc)
            runs.append(r)
            if p is not None:
                np.savez_compressed(probs / f"{name}_seed{s}.npz", test_logits=p[0], test_labels=p[1], test_patient=p[2])
            print(f"  [{name}] lr={best_lr:.0e} seed{s} -> {r.get('test_auc', r.get('error'))}", flush=True)
        ok = [r for r in runs if r.get("ok")]
        tests = [r["test_auc"] for r in ok]
        vals = [r["val_auc"] for r in ok]
        results[name] = dict(done=True, ok=True, best_lr=best_lr, params=ok[0]["params"],
                             test_mean=round(statistics.mean(tests), 4),
                             test_std=round(statistics.pstdev(tests), 4),
                             val_mean=round(statistics.mean(vals), 4),
                             test_seeds=tests, n_ok=len(ok))
        out.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"[done] {name}: test={results[name]['test_mean']}+-{results[name]['test_std']} "
              f"params={ok[0]['params']/1e6:.2f}M lr={best_lr:.0e}", flush=True)

    print("\n=== PATIENT-GROUPED NIH (28x28) EXTERNAL VALIDATION (test macro-AUC) ===", flush=True)
    print(f"{'model':22s} {'params(M)':>9s} {'test':>8s} {'std':>7s} {'val':>7s} {'lr':>6s}", flush=True)
    for name, r in sorted(results.items(), key=lambda kv: -(kv[1].get("test_mean") or 0)):
        if not r.get("ok"):
            print(f"{name:22s}   FAILED ({r.get('error','')[:40]})", flush=True)
            continue
        print(f"{name:22s} {r['params']/1e6:>9.2f} {r['test_mean']:>8.4f} {r['test_std']:>7.4f} "
              f"{r['val_mean']:>7.4f} {r['best_lr']:>6.0e}", flush=True)


if __name__ == "__main__":
    main()
