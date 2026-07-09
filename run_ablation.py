"""Run the CASP single-component ablations under the EXACT matched protocol.

Reuses bench_chestmnist.train_eval (AdamW wd 1e-4, 5% warmup + cosine, best-val
checkpoint, per-model LR-select on seed 0 from the shared grid then the chosen LR
over the given seeds, ChestMNIST test macro-AUC). Writes to a SEPARATE json
(bench_ablation.json) so bench_results.json is never touched.

Usage:
  python run_ablation.py --steps 4000 --eval-every 500 --seeds 0,1,2 \
      --lrs 1e-3,3e-4 --batch 256 --out bench_ablation.json
"""
import argparse
import importlib.util
import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import bench_chestmnist as BENCH  # reuse train_eval + protocol verbatim
import casp_ablations as ABL


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="chestmnist")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--lrs", default="1e-3,3e-4")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--models", default=",".join(ABL.ABLATIONS))
    ap.add_argument("--data-root", default=str(HERE / "data"))
    ap.add_argument("--out", default=str(HERE / "bench_ablation.json"))
    a = ap.parse_args()

    nc = {"chestmnist": 14, "pneumoniamnist": 1}[a.dataset]
    seeds = [int(s) for s in a.seeds.split(",")]
    lrs = [float(x) for x in a.lrs.split(",")]
    models = [m for m in a.models.split(",") if m in ABL.ABLATIONS]
    out = Path(a.out)
    results = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}

    for name in models:
        if results.get(name, {}).get("done"):
            print(f"[skip] {name} already done", flush=True)
            continue
        bf = ABL.ABLATIONS[name]
        lr_runs = {}
        for lr in lrs:
            r = BENCH.train_eval(bf, a.dataset, a.steps, seeds[0], lr, a.eval_every,
                                 a.data_root, a.batch, nc)
            lr_runs[lr] = r
            print(f"  [{name}] lr={lr:.0e} seed{seeds[0]} -> {r.get('test_auc', r.get('error'))}",
                  flush=True)
        ok_lrs = {lr: r for lr, r in lr_runs.items() if r.get("ok")}
        if not ok_lrs:
            results[name] = dict(done=True, ok=False, error="all LRs failed: " + str(lr_runs))
            out.write_text(json.dumps(results, indent=2), encoding="utf-8")
            continue
        best_lr = max(ok_lrs, key=lambda lr: ok_lrs[lr]["val_auc"])
        runs = [ok_lrs[best_lr]]
        for s in seeds[1:]:
            r = BENCH.train_eval(bf, a.dataset, a.steps, s, best_lr, a.eval_every,
                                 a.data_root, a.batch, nc)
            runs.append(r)
            print(f"  [{name}] lr={best_lr:.0e} seed{s} -> {r.get('test_auc', r.get('error'))}",
                  flush=True)
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

    # Reference row: full CASP on the SAME 3 seeds (0,1,2) as the ablations, so every
    # delta is like-for-like. Seeds 0,1,2 of the 5-seed run were [0.7838, 0.7808, 0.7799]
    # (mean 0.7815, std 0.0017); the 5-seed mean is 0.7826 and is reported in the main table.
    ref = dict(params=1896566, test_mean=0.7815, test_std=0.0017, n_seeds=3,
               seeds=[0.7838, 0.7808, 0.7799], reused=True)
    print("\n=== CASP ABLATION (ChestMNIST test macro-AUC, matched protocol) ===", flush=True)
    print(f"{'variant':16s} {'params(M)':>9s} {'test':>8s} {'std':>7s} {'delta_vs_full':>13s}",
          flush=True)
    print(f"{'searched_nonrec':16s} {ref['params']/1e6:>9.4f} {ref['test_mean']:>8.4f} "
          f"{ref['test_std']:>7.4f} {'(reference)':>13s}", flush=True)
    for name in models:
        r = results.get(name, {})
        if not r.get("ok"):
            print(f"{name:16s}   FAILED ({r.get('error','')[:50]})", flush=True)
            continue
        delta = r["test_mean"] - ref["test_mean"]
        print(f"{name:16s} {r['params']/1e6:>9.4f} {r['test_mean']:>8.4f} {r['test_std']:>7.4f} "
              f"{delta:>+13.4f}", flush=True)


if __name__ == "__main__":
    main()
