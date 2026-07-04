"""Matched-protocol head-to-head benchmark on ChestMNIST (28x28, 14-label macro-AUC).

Every model is trained under ONE identical protocol: AdamW + 5% warmup then cosine,
a fixed step budget with best-validation-checkpoint selection, and the official test
split evaluated exactly once at the best-validation checkpoint. For each model a small
learning-rate selection is run on seed 0 (the same grid for all models), then the
selected rate is run over five seeds. Reports test macro-AUC mean +/- std, parameters,
and the selected rate.

Models: the two searched architectures (the non-recurrent one with its eval-time shift
ensemble disabled so all models are compared on a single forward pass), a
parameter-matched non-recurrent control (scaled to the recurrent budget), and
ResNet-18, ResNet-50, DenseNet-121, MobileNetV2, and a compact ViT.

This script only produces numbers (bench_results.json + a printed table). It writes
nothing into the manuscript.
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
from trm_chest.data import DataConfig, make_loaders, unpack_batch  # noqa: E402
from trm_chest.metrics import macro_auc_from_logits  # noqa: E402


def load_module(path):
    spec = importlib.util.spec_from_file_location("m_" + Path(path).parent.name, path)
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
    m.eval_shift_ensemble = False  # disable test-time augmentation for a fair single-pass eval
    return m


def searched_nonrec_pm(nc, in_ch=1):
    # parameter-matched non-recurrent control, scaled to the recurrent budget (~1.5M)
    m = NONREC.ConvAttnClassifier(nc, in_ch=in_ch, h=168, heads=6, blocks=2,
                                  pool_queries=4, drop_path_rate=0.06, eval_shift_ensemble=False)
    return m


def searched_rec(nc, in_ch=1):
    return REC.build_model(nc, in_ch)


# ImageNet stems downsample too aggressively for 28x28, so a small-image stem
# (3x3 stride-1 first conv, no initial maxpool) is used. This is the standard
# adaptation for running these backbones on small inputs.
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
def eval_auc(model, loader, dev):
    model.eval()
    L, Y = [], []
    for batch in loader:
        x, y = unpack_batch(batch, dev)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            o = model(x)
        L.append(o.float().cpu().numpy())
        Y.append(y.cpu().numpy())
    return macro_auc_from_logits(np.concatenate(L, 0), np.concatenate(Y, 0))


def lr_factor(step, total, warm):
    if step < warm:
        return step / max(1, warm)
    p = (step - warm) / max(1, total - warm)
    return 0.5 * (1 + math.cos(math.pi * p))


def train_eval(build_fn, dataset, steps, seed, lr, eval_every, data_root, batch, nc):
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tr, va, te = make_loaders(DataConfig(data_root=data_root, batch_size=batch, num_workers=0, seed=seed, dataset=dataset))
    try:
        model = build_fn(nc, 1).to(dev)
    except Exception as e:  # noqa: BLE001
        return dict(ok=False, error=f"build: {e}")
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
        x, y = unpack_batch(next(it), dev)
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            o = model(x)
            loss = bce(o, y)
        if not torch.isfinite(loss):
            return dict(ok=False, error=f"nonfinite@{step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if (step + 1) % eval_every == 0 or step == steps - 1:
            v = eval_auc(model, va, dev)
            if v > best_val:
                best_val = v
                best_state = {k: val.detach().cpu().clone() for k, val in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    t = eval_auc(model, te, dev)
    return dict(ok=True, val_auc=round(best_val, 4), test_auc=round(t, 4),
                params=count_p(model), seconds=round(time.time() - t0, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="chestmnist")
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--lrs", default="1e-3,3e-4")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--data-root", default=str(HERE / "data"))
    ap.add_argument("--out", default=str(HERE / "bench_results.json"))
    a = ap.parse_args()
    nc = {"chestmnist": 14, "pneumoniamnist": 1}[a.dataset]
    seeds = [int(s) for s in a.seeds.split(",")]
    lrs = [float(x) for x in a.lrs.split(",")]
    models = [m for m in a.models.split(",") if m in MODELS]
    out = Path(a.out)
    results = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}

    for name in models:
        if results.get(name, {}).get("done"):
            print(f"[skip] {name} already done", flush=True)
            continue
        bf = MODELS[name]
        lr_runs = {}
        for lr in lrs:
            r = train_eval(bf, a.dataset, a.steps, seeds[0], lr, a.eval_every, a.data_root, a.batch, nc)
            lr_runs[lr] = r
            print(f"  [{name}] lr={lr:.0e} seed{seeds[0]} -> {r.get('test_auc', r.get('error'))}", flush=True)
        ok_lrs = {lr: r for lr, r in lr_runs.items() if r.get("ok")}
        if not ok_lrs:
            results[name] = dict(done=True, ok=False, error="all LRs failed: " + str(lr_runs))
            out.write_text(json.dumps(results, indent=2), encoding="utf-8")
            continue
        best_lr = max(ok_lrs, key=lambda lr: ok_lrs[lr]["val_auc"])
        runs = [ok_lrs[best_lr]]
        for s in seeds[1:]:
            r = train_eval(bf, a.dataset, a.steps, s, best_lr, a.eval_every, a.data_root, a.batch, nc)
            runs.append(r)
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

    print("\n=== MATCHED-PROTOCOL CHESTMNIST BENCHMARK (test macro-AUC) ===", flush=True)
    print(f"{'model':22s} {'params(M)':>9s} {'test':>8s} {'std':>7s} {'val':>7s} {'lr':>6s}", flush=True)
    for name, r in sorted(results.items(), key=lambda kv: -(kv[1].get("test_mean") or 0)):
        if not r.get("ok"):
            print(f"{name:22s}   FAILED ({r.get('error','')[:40]})", flush=True)
            continue
        print(f"{name:22s} {r['params']/1e6:>9.2f} {r['test_mean']:>8.4f} {r['test_std']:>7.4f} "
              f"{r['val_mean']:>7.4f} {r['best_lr']:>6.0e}", flush=True)


if __name__ == "__main__":
    main()
