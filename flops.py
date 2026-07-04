"""Parameters, forward FLOPs (MACs), and GPU latency for the 8 matched-protocol models
at 28x28, single forward pass in eval mode (TTA disabled). The recursive model's
multi-cycle cost is captured because fvcore traces the actual forward and latency is
wall-clock. Used for the efficiency table."""
import json
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from bench_chestmnist import MODELS  # noqa: E402

try:
    from fvcore.nn import FlopCountAnalysis
    HAVE_FV = True
except Exception as e:  # noqa: BLE001
    print("no fvcore:", e, flush=True)
    HAVE_FV = False


def count_p(m):
    return sum(p.numel() for p in m.parameters())


dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def latency(m, bs, iters=60, warm=15):
    m.eval().to(dev)
    x = torch.zeros(bs, 1, 28, 28, device=dev)
    for _ in range(warm):
        m(x)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        m(x)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    return (time.time() - t0) / iters * 1000.0


xc = torch.zeros(1, 1, 28, 28)
print(f"{'model':22s} {'params(M)':>10s} {'MMac@28':>10s} {'lat_b1(ms)':>11s} {'lat_b256(ms)':>13s}", flush=True)
out = {}
for name, bf in MODELS.items():
    p = count_p(bf(14, 1))
    f = float("nan")
    if HAVE_FV:
        try:
            mm = bf(14, 1).eval()
            fca = FlopCountAnalysis(mm, xc)
            fca.unsupported_ops_warnings(False)
            fca.uncalled_modules_warnings(False)
            f = fca.total() / 1e6
        except Exception as e:  # noqa: BLE001
            print("flop err", name, e, flush=True)
    l1 = latency(bf(14, 1), 1)
    l256 = latency(bf(14, 1), 256)
    out[name] = dict(params=p, mmac=round(f, 2), lat_b1_ms=round(l1, 4), lat_b256_ms=round(l256, 4))
    print(f"{name:22s} {p/1e6:>10.3f} {f:>10.1f} {l1:>11.3f} {l256:>13.2f}", flush=True)
Path(HERE / "flops_latency.json").write_text(json.dumps(out, indent=2))
print("WROTE flops_latency.json", flush=True)
