"""CASP single-component ablations for the IEEE TMI ablation table.

Each variant removes EXACTLY ONE component of the full CASP model
(best_nonrec.ConvAttnClassifier) and changes nothing else. The full model is
imported from best_nonrec; variants are built by subclassing the relevant
submodules and passing flags, so the shared training/eval protocol in
bench_chestmnist.py is reused verbatim.

Variants:
  casp_no_local  - remove the local token-mixing residual branch in SelfAttnBlock
  casp_mean_pool - plain mean pooling over image tokens (drop pooled queries + max/std)
  casp_no_se     - squeeze-and-excite refinement in the stem made identity
  casp_std_down  - pixel-unshuffle stem downsample -> strided-conv downsample (same 7x7 grid)
  casp_no_ws     - weight-standardized convs -> standard nn.Conv2d

Every build_fn matches the incumbent signature build_fn(num_classes, in_ch=1)->nn.Module
and, like searched_nonrec in bench_chestmnist.py, disables the eval-time shift ensemble
so the comparison is a single forward pass (identical to how the full CASP reference
row was produced).
"""
import importlib.util
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent


def _load(path):
    spec = importlib.util.spec_from_file_location("best_nonrec_casp", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


NR = _load(HERE / "best_nonrec.py")


# --------------------------------------------------------------------------------------
# Variant 1: casp_no_local -- drop the local token-mixing residual branch.
# --------------------------------------------------------------------------------------
class SelfAttnBlockNoLocal(NR.SelfAttnBlock):
    """SelfAttnBlock with the depthwise LOCAL token-mixing residual removed.

    forward keeps attention + MLP only (the exact two other residuals, unchanged).
    The self.local / ls_local / dp_local submodules are deleted so no parameters
    linger and the param count actually drops.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Remove the local branch and its LayerScale + DropPath so params reflect the ablation.
        del self.local
        del self.ls_local
        del self.dp_local

    def forward(self, x):
        y = self.n1(x)
        a, _ = self.attn(y, y, y, need_weights=False)
        x = x + self.dp1(self.ls_attn(a))
        # (local branch removed)
        return x + self.dp2(self.ls_mlp(self.mlp(self.n2(x))))


# --------------------------------------------------------------------------------------
# Variant 2: casp_mean_pool -- plain mean pooling over image tokens only.
# --------------------------------------------------------------------------------------
class ConvAttnMeanPool(NR.ConvAttnClassifier):
    """CASP with the multi-query + mean/max/std pooling head replaced by plain
    mean pooling over the image tokens only.

    The pooling query parameters are dropped (no query tokens are prepended, so
    the blocks/local mixer see image tokens only), and the fused head takes a
    single h-dim mean vector instead of h*(pool_queries+3).
    """

    def __init__(self, num_classes, in_ch=1, h=192, heads=6, blocks=2,
                 pool_queries=4, drop_path_rate=0.06, eval_shift_ensemble=True):
        # pool_queries=0 makes SelfAttnBlock/LocalTokenMixer treat all tokens as image tokens.
        super().__init__(num_classes, in_ch=in_ch, h=h, heads=heads, blocks=blocks,
                         pool_queries=0, drop_path_rate=drop_path_rate,
                         eval_shift_ensemble=eval_shift_ensemble)
        # Replace the pool-query parameter with an empty (unused) buffer-like param.
        del self.pool
        self.register_parameter("pool", None)
        # Fused dim is now just h (single mean vector); rebuild the norm + head accordingly.
        self.fuse_norm = nn.LayerNorm(h)
        self.head = nn.Linear(h, num_classes)

    def _forward_once(self, x):
        x = self.input_shift(x)
        tok = self.stem(x)
        tok = tok + self.pos[:, : tok.shape[1]]
        t = tok  # no pooling queries prepended
        for blk in self.blocks:
            t = blk(t)
        image_tokens = self.norm(t)
        pooled = image_tokens.mean(dim=1)  # plain mean pool over image tokens
        pooled = self.fuse_norm(pooled)
        return self.head(self.drop(pooled))


# --------------------------------------------------------------------------------------
# Variant 3: casp_no_se -- squeeze-and-excite refinement made identity.
# --------------------------------------------------------------------------------------
class ConvStemNoSE(NR.ConvStem):
    """ConvStem with the SERefineBlock channel refinement replaced by identity."""

    def __init__(self, in_ch=1, h=192):
        super().__init__(in_ch=in_ch, h=h)
        self.refine = nn.Identity()


class ConvAttnNoSE(NR.ConvAttnClassifier):
    def __init__(self, num_classes, in_ch=1, h=192, heads=6, blocks=2,
                 pool_queries=4, drop_path_rate=0.06, eval_shift_ensemble=True):
        super().__init__(num_classes, in_ch=in_ch, h=h, heads=heads, blocks=blocks,
                         pool_queries=pool_queries, drop_path_rate=drop_path_rate,
                         eval_shift_ensemble=eval_shift_ensemble)
        self.stem = ConvStemNoSE(in_ch, h)


# --------------------------------------------------------------------------------------
# Variant 4: casp_std_down -- pixel-unshuffle stem downsample -> strided conv.
# --------------------------------------------------------------------------------------
class StridedConvDownsample(nn.Module):
    """Standard strided-conv 2x downsample to the same grid, matching the compact
    WS projection style of PixelUnshuffleDownsample but WITHOUT the lossless
    space-to-channel pixel-unshuffle. A single stride-2 3x3 WS conv performs the
    decimation (in_ch -> out_ch), followed by the same norm/act + depthwise refine.
    """

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.proj = nn.Sequential(
            NR.WSConv2d(in_ch, out_ch, 3, stride=2, padding=1),
            NR.ChannelLayerNorm2d(out_ch),
            nn.SiLU(),
            NR.WSConv2d(out_ch, out_ch, 3, padding=1, groups=out_ch),
            NR.ChannelLayerNorm2d(out_ch),
            nn.SiLU(),
        )

    def forward(self, x):
        return self.proj(x)


class ConvStemStdDown(NR.ConvStem):
    def __init__(self, in_ch=1, h=192):
        super().__init__(in_ch=in_ch, h=h)
        c = h // 2
        self.down1 = StridedConvDownsample(c, h)


class ConvAttnStdDown(NR.ConvAttnClassifier):
    def __init__(self, num_classes, in_ch=1, h=192, heads=6, blocks=2,
                 pool_queries=4, drop_path_rate=0.06, eval_shift_ensemble=True):
        super().__init__(num_classes, in_ch=in_ch, h=h, heads=heads, blocks=blocks,
                         pool_queries=pool_queries, drop_path_rate=drop_path_rate,
                         eval_shift_ensemble=eval_shift_ensemble)
        self.stem = ConvStemStdDown(in_ch, h)


# --------------------------------------------------------------------------------------
# Variant 5: casp_no_ws -- weight-standardized convs -> standard nn.Conv2d.
# --------------------------------------------------------------------------------------
def _swap_wsconv_to_plain(module):
    """Recursively replace every WSConv2d with a plain nn.Conv2d of identical
    geometry (removing weight standardization). Bias presence is preserved.
    """
    for name, child in list(module.named_children()):
        if isinstance(child, NR.WSConv2d):
            plain = nn.Conv2d(
                child.in_channels, child.out_channels, child.kernel_size,
                stride=child.stride, padding=child.padding, dilation=child.dilation,
                groups=child.groups, bias=child.bias is not None,
            )
            setattr(module, name, plain)
        else:
            _swap_wsconv_to_plain(child)


class ConvAttnNoWS(NR.ConvAttnClassifier):
    def __init__(self, num_classes, in_ch=1, h=192, heads=6, blocks=2,
                 pool_queries=4, drop_path_rate=0.06, eval_shift_ensemble=True):
        super().__init__(num_classes, in_ch=in_ch, h=h, heads=heads, blocks=blocks,
                         pool_queries=pool_queries, drop_path_rate=drop_path_rate,
                         eval_shift_ensemble=eval_shift_ensemble)
        _swap_wsconv_to_plain(self)


# --------------------------------------------------------------------------------------
# ConvAttnClassifier with the NoLocal block substituted (variant 1 needs a block swap).
# --------------------------------------------------------------------------------------
class ConvAttnNoLocal(NR.ConvAttnClassifier):
    def __init__(self, num_classes, in_ch=1, h=192, heads=6, blocks=2,
                 pool_queries=4, drop_path_rate=0.06, eval_shift_ensemble=True):
        super().__init__(num_classes, in_ch=in_ch, h=h, heads=heads, blocks=blocks,
                         pool_queries=pool_queries, drop_path_rate=drop_path_rate,
                         eval_shift_ensemble=eval_shift_ensemble)
        if blocks > 1:
            dpr = torch.linspace(0.0, drop_path_rate, blocks).tolist()
        else:
            dpr = [drop_path_rate]
        self.blocks = nn.ModuleList([
            SelfAttnBlockNoLocal(h, heads, drop_path=dpr[i], pool_queries=pool_queries,
                                 layerscale_init=0.1)
            for i in range(blocks)
        ])


# --------------------------------------------------------------------------------------
# Build functions (signature matches bench_chestmnist.py: build_fn(num_classes, in_ch=1)).
# All disable the eval-time shift ensemble, exactly like searched_nonrec, so the
# comparison is a single forward pass.
# --------------------------------------------------------------------------------------
def _mk(cls):
    def _build(nc, in_ch=1):
        m = cls(nc, in_ch=in_ch, h=192, heads=6, blocks=2, pool_queries=4,
                drop_path_rate=0.06, eval_shift_ensemble=True)
        m.eval_shift_ensemble = False
        return m
    return _build


build_no_local = _mk(ConvAttnNoLocal)
build_mean_pool = _mk(ConvAttnMeanPool)
build_no_se = _mk(ConvAttnNoSE)
build_std_down = _mk(ConvAttnStdDown)
build_no_ws = _mk(ConvAttnNoWS)


ABLATIONS = {
    "casp_no_local": build_no_local,
    "casp_mean_pool": build_mean_pool,
    "casp_no_se": build_no_se,
    "casp_std_down": build_std_down,
    "casp_no_ws": build_no_ws,
}


def _selfcheck():
    """Build each variant, run a CPU forward on zeros(2,1,28,28)->(2,14), report params."""
    full = NR.build_model(14, 1)
    full_p = sum(p.numel() for p in full.parameters())
    print(f"{'variant':16s} {'params':>10s} {'params(M)':>10s} {'delta_vs_full':>14s}  forward")
    print(f"{'searched_nonrec':16s} {full_p:>10d} {full_p/1e6:>10.4f} {'0':>14s}  (reference)")
    x = torch.zeros(2, 1, 28, 28)
    for name, bf in ABLATIONS.items():
        m = bf(14, 1)
        m.eval()
        p = sum(pp.numel() for pp in m.parameters())
        with torch.no_grad():
            out = m(x)
        shape_ok = tuple(out.shape) == (2, 14)
        finite_ok = bool(torch.isfinite(out).all())
        print(f"{name:16s} {p:>10d} {p/1e6:>10.4f} {p-full_p:>14d}  "
              f"shape={tuple(out.shape)} ok={shape_ok and finite_ok}")


if __name__ == "__main__":
    _selfcheck()
