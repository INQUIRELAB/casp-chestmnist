'''
CASP: a compact single-pass convolutional-attention chest-radiograph classifier with detail-preserving pixel-unshuffle stem downsampling, SE refinement, weight-standardized convolutions, spatial-contrast-preserving channel-only stem normalization, anti-aliased stem transition, local token mixing, LayerScale residual calibration, variance-augmented diagnostic pooling, train-time micro-translation regularization, and eval-time micro-shift phase averaging.

Interface:
  * define build_model(num_classes:int, in_ch:int=1) -> nn.Module
  * forward(x: float[b, in_ch, 28, 28]) -> logits[b, num_classes]
  * pure PyTorch, one self-contained module, no file/network access.
'''
import torch
from torch import nn
from torch.nn import functional as F


class WSConv2d(nn.Conv2d):
    '''Conv2d with per-output-channel weight standardization.'''

    def __init__(self, *args, eps=1e-5, **kwargs):
        super().__init__(*args, **kwargs)
        self.eps = float(eps)

    def forward(self, x):
        w = self.weight
        w = w - w.mean(dim=(1, 2, 3), keepdim=True)
        var = w.pow(2).mean(dim=(1, 2, 3), keepdim=True)
        w = w * torch.rsqrt(var + self.eps)
        return F.conv2d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)


class ChannelLayerNorm2d(nn.Module):
    '''LayerNorm over channels at each spatial location, preserving spatial contrast.'''

    def __init__(self, channels, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = float(eps)

    def forward(self, x):
        y = x.permute(0, 2, 3, 1)
        y = F.layer_norm(y, (x.shape[1],), self.weight, self.bias, self.eps)
        return y.permute(0, 3, 1, 2).contiguous()


class DropPath(nn.Module):
    '''Per-sample stochastic depth for residual branches.'''

    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep_prob)
        return x * mask.div(keep_prob)


class TrainTimeShift(nn.Module):
    '''Small reflect-padded translation augmentation applied only during training.'''

    def __init__(self, max_shift=1, p=0.35):
        super().__init__()
        self.max_shift = int(max_shift)
        self.p = float(p)

    def forward(self, x):
        if not self.training or self.max_shift <= 0 or self.p <= 0.0:
            return x
        if torch.rand((), device=x.device) >= self.p:
            return x
        shifts = torch.randint(-self.max_shift, self.max_shift + 1, (2,), device=x.device)
        dy = int(shifts[0].item())
        dx = int(shifts[1].item())
        if dy == 0 and dx == 0:
            return x
        pad = self.max_shift
        y = F.pad(x, (pad, pad, pad, pad), mode='reflect')
        top = pad + dy
        left = pad + dx
        return y[:, :, top:top + x.shape[-2], left:left + x.shape[-1]]


class AntiAliasDownsample(nn.Module):
    '''Fixed binomial low-pass filter before 2x spatial decimation.'''

    def __init__(self, channels):
        super().__init__()
        filt = torch.tensor([1.0, 2.0, 1.0])
        kernel = filt[:, None] * filt[None, :]
        kernel = kernel / kernel.sum()
        self.register_buffer('kernel', kernel.view(1, 1, 3, 3).repeat(channels, 1, 1, 1))
        self.channels = int(channels)

    def forward(self, x):
        y = F.pad(x, (1, 1, 1, 1), mode='reflect')
        return F.conv2d(y, self.kernel.to(dtype=x.dtype), stride=2, groups=self.channels)


class SERefineBlock(nn.Module):
    '''Residual depthwise convolutional refinement with channel recalibration.'''

    def __init__(self, h, reduction=8):
        super().__init__()
        hidden = max(16, h // reduction)
        self.conv = nn.Sequential(
            WSConv2d(h, h, 3, padding=1, groups=h),
            ChannelLayerNorm2d(h),
            nn.SiLU(),
            WSConv2d(h, h, 1),
            ChannelLayerNorm2d(h),
            nn.SiLU(),
        )
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            WSConv2d(h, hidden, 1),
            nn.SiLU(),
            WSConv2d(hidden, h, 1),
            nn.Sigmoid(),
        )
        self.out_norm = ChannelLayerNorm2d(h)

    def forward(self, x):
        y = self.conv(x)
        y = y * self.se(y)
        return self.out_norm(x + y)


class PixelUnshuffleDownsample(nn.Module):
    '''Lossless 2x2 spatial-to-channel downsampling followed by compact WS projection.'''

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.proj = nn.Sequential(
            WSConv2d(in_ch * 4, out_ch, 1),
            ChannelLayerNorm2d(out_ch),
            nn.SiLU(),
            WSConv2d(out_ch, out_ch, 3, padding=1, groups=out_ch),
            ChannelLayerNorm2d(out_ch),
            nn.SiLU(),
        )

    def forward(self, x):
        return self.proj(F.pixel_unshuffle(x, 2))


class ConvStem(nn.Module):
    '''Convolutional stem with lossless early downsampling: 28x28x1 -> 7x7xH feature tokens.'''

    def __init__(self, in_ch=1, h=192):
        super().__init__()
        c = h // 2
        self.pre = nn.Sequential(
            WSConv2d(in_ch, c, 3, stride=1, padding=1),
            ChannelLayerNorm2d(c),
            nn.SiLU(),
            WSConv2d(c, c, 3, stride=1, padding=1),
            ChannelLayerNorm2d(c),
            nn.SiLU(),
        )
        self.down1 = PixelUnshuffleDownsample(c, h)
        self.post = nn.Sequential(
            AntiAliasDownsample(h),
            WSConv2d(h, h, 3, stride=1, padding=1),
            ChannelLayerNorm2d(h),
            nn.SiLU(),
            WSConv2d(h, h, 3, stride=1, padding=1),
            ChannelLayerNorm2d(h),
            nn.SiLU(),
        )
        self.refine = SERefineBlock(h)

    def forward(self, x):
        f = self.pre(x)
        f = self.down1(f)
        f = self.refine(self.post(f))
        return f.flatten(2).transpose(1, 2)


class LocalTokenMixer(nn.Module):
    '''Depthwise local mixing over the 7x7 image-token grid, leaving pooling queries untouched.'''

    def __init__(self, h, pool_queries, grid_size=7):
        super().__init__()
        self.pool_queries = int(pool_queries)
        self.grid_size = int(grid_size)
        self.norm = nn.LayerNorm(h)
        self.dwconv = WSConv2d(h, h, 3, padding=1, groups=h)
        self.pwconv = WSConv2d(h, h, 1)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            WSConv2d(h, max(16, h // 8), 1),
            nn.SiLU(),
            WSConv2d(max(16, h // 8), h, 1),
            nn.Sigmoid(),
        )
        self.out_norm = nn.GroupNorm(8, h)

    def forward(self, x):
        q = x[:, : self.pool_queries]
        img = x[:, self.pool_queries:]
        b, n, h = img.shape
        g = self.grid_size
        y = self.norm(img).transpose(1, 2).reshape(b, h, g, g)
        y = self.dwconv(y)
        y = self.pwconv(y)
        y = y * self.gate(y)
        y = self.out_norm(y).flatten(2).transpose(1, 2)
        return torch.cat([q, img + y], dim=1)


class LayerScale(nn.Module):
    '''Learnable per-channel residual scale for short, stable training runs.'''

    def __init__(self, h, init_value=0.1):
        super().__init__()
        self.gamma = nn.Parameter(torch.full((h,), float(init_value)))

    def forward(self, x):
        return x * self.gamma.view(1, 1, -1)


class SelfAttnBlock(nn.Module):
    '''Pre-norm transformer block with local image-token mixing and calibrated residual updates.'''

    def __init__(self, h, heads=6, mlp_ratio=4.0, drop_path=0.0, pool_queries=4, layerscale_init=0.1):
        super().__init__()
        self.n1 = nn.LayerNorm(h)
        self.attn = nn.MultiheadAttention(h, heads, dropout=0.0, batch_first=True)
        self.local = LocalTokenMixer(h, pool_queries=pool_queries, grid_size=7)
        self.n2 = nn.LayerNorm(h)
        mh = int(h * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(h, mh), nn.SiLU(), nn.Linear(mh, h))
        self.ls_attn = LayerScale(h, layerscale_init)
        self.ls_local = LayerScale(h, layerscale_init)
        self.ls_mlp = LayerScale(h, layerscale_init)
        self.dp1 = DropPath(drop_path)
        self.dp_local = DropPath(drop_path)
        self.dp2 = DropPath(drop_path)

    def forward(self, x):
        y = self.n1(x)
        a, _ = self.attn(y, y, y, need_weights=False)
        x = x + self.dp1(self.ls_attn(a))
        x = x + self.dp_local(self.ls_local(self.local(x) - x))
        return x + self.dp2(self.ls_mlp(self.mlp(self.n2(x))))


class ConvAttnClassifier(nn.Module):
    '''Detail-preserving WS conv stem + anti-aliased transition + calibrated local token mixing + multi-query pooling with global token statistics.'''

    def __init__(self, num_classes, in_ch=1, h=192, heads=6, blocks=2, pool_queries=4, drop_path_rate=0.06, eval_shift_ensemble=True):
        super().__init__()
        self.input_shift = TrainTimeShift(max_shift=1, p=0.35)
        self.eval_shift_ensemble = bool(eval_shift_ensemble)
        self.stem = ConvStem(in_ch, h)
        self.pos = nn.Parameter(torch.zeros(1, 49, h))
        self.pool = nn.Parameter(torch.randn(1, pool_queries, h) * 0.02)
        if blocks > 1:
            dpr = torch.linspace(0.0, drop_path_rate, blocks).tolist()
        else:
            dpr = [drop_path_rate]
        self.blocks = nn.ModuleList([
            SelfAttnBlock(h, heads, drop_path=dpr[i], pool_queries=pool_queries, layerscale_init=0.1)
            for i in range(blocks)
        ])
        self.norm = nn.LayerNorm(h)
        fused_dim = h * (pool_queries + 3)
        self.fuse_norm = nn.LayerNorm(fused_dim)
        self.drop = nn.Dropout(0.14)
        self.head = nn.Linear(fused_dim, num_classes)

    def _shift_eval_input(self, x, dy, dx):
        if dy == 0 and dx == 0:
            return x
        y = F.pad(x, (1, 1, 1, 1), mode='reflect')
        top = 1 + int(dy)
        left = 1 + int(dx)
        return y[:, :, top:top + x.shape[-2], left:left + x.shape[-1]]

    def _forward_once(self, x):
        x = self.input_shift(x)
        tok = self.stem(x)
        tok = tok + self.pos[:, : tok.shape[1]]
        q = self.pool.expand(x.shape[0], -1, -1)
        t = torch.cat([q, tok], 1)
        for blk in self.blocks:
            t = blk(t)
        pooled_queries = self.norm(t[:, : self.pool.shape[1]]).flatten(1)
        image_tokens = self.norm(t[:, self.pool.shape[1]:])
        mean_stat = image_tokens.mean(dim=1)
        max_stat = image_tokens.amax(dim=1)
        centered = image_tokens - mean_stat.unsqueeze(1)
        std_stat = torch.sqrt(centered.pow(2).mean(dim=1).clamp_min(1e-6))
        pooled = torch.cat([pooled_queries, mean_stat, max_stat, std_stat], dim=1)
        pooled = self.fuse_norm(pooled)
        return self.head(self.drop(pooled))

    def forward(self, x):
        if self.training or not self.eval_shift_ensemble:
            return self._forward_once(x)
        shifts = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))
        logits = None
        for dy, dx in shifts:
            shifted = self._shift_eval_input(x, dy, dx)
            out = self._forward_once(shifted)
            logits = out if logits is None else logits + out
        return logits * (1.0 / len(shifts))


def build_model(num_classes, in_ch=1):
    return ConvAttnClassifier(num_classes, in_ch=in_ch, h=192, heads=6, blocks=2, pool_queries=4, drop_path_rate=0.06, eval_shift_ensemble=True)
