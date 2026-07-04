'''Recursive (TRM-style) medical classifier with prior-calibrated cycle supervision.

A convolutional stem embeds the 28x28 image, then an answer token and reason tokens
are iteratively refined while attending to fixed image tokens. The recurrent state
keeps a stable normalized gated latent update, and keeps
the gradient-only auxiliary cycle supervision, but initializes the shared classifier
bias to a conservative multi-label prior when num_classes > 2. That calibrates the
BCE gradients seen by both final and auxiliary recurrent readouts during short
training without changing the forward contract.

Contract: build_model(num_classes, in_ch=1) -> nn.Module;
forward(x: float[b, in_ch, 28, 28]) -> logits[b, num_classes]. Pure PyTorch,
self-contained.
'''
import torch
from torch import nn


class ConvStem(nn.Module):
    def __init__(self, in_ch=1, h=192):
        super().__init__()
        c = h // 2
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, c, 3, 1, 1), nn.GroupNorm(8, c), nn.SiLU(),
            nn.Conv2d(c, c, 3, 2, 1), nn.GroupNorm(8, c), nn.SiLU(),
            nn.Conv2d(c, h, 3, 1, 1), nn.GroupNorm(8, h), nn.SiLU(),
            nn.Conv2d(h, h, 3, 2, 1), nn.GroupNorm(8, h), nn.SiLU(),
            nn.Conv2d(h, h, 3, 1, 1), nn.GroupNorm(8, h), nn.SiLU(),
        )

    def forward(self, x):
        f = self.net(x)
        return f.flatten(2).transpose(1, 2)  # [b, 49, h]


class TokenBlock(nn.Module):
    '''Pre-norm self-attention + SiLU MLP, residual.'''

    def __init__(self, h, heads=6, mlp_ratio=4.0):
        super().__init__()
        self.n1 = nn.LayerNorm(h)
        self.attn = nn.MultiheadAttention(h, heads, dropout=0.0, batch_first=True)
        self.n2 = nn.LayerNorm(h)
        mh = int(h * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(h, mh), nn.SiLU(), nn.Linear(mh, h))

    def forward(self, x):
        y = self.n1(x)
        a, _ = self.attn(y, y, y, need_weights=False)
        x = x + a
        return x + self.mlp(self.n2(x))


class GatedLatentUpdate(nn.Module):
    '''Bounded recurrent update: lat <- norm((1-g)*lat + g*proposal).'''

    def __init__(self, h):
        super().__init__()
        self.prev_norm = nn.LayerNorm(h)
        self.prop_norm = nn.LayerNorm(h)
        self.out_norm = nn.LayerNorm(h)
        self.gate = nn.Sequential(
            nn.Linear(2 * h, h),
            nn.SiLU(),
            nn.Linear(h, h),
        )
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.constant_(self.gate[-1].bias, -0.7)

    def forward(self, lat, proposal):
        lat_n = self.prev_norm(lat)
        prop_n = self.prop_norm(proposal)
        gate = torch.sigmoid(self.gate(torch.cat([lat_n, prop_n], dim=-1)))
        mixed = lat_n + gate * (prop_n - lat_n)
        return self.out_norm(mixed)


class RecurrentReasoner(nn.Module):
    def __init__(self, num_classes, in_ch=1, h=192, heads=6, reason_tokens=8, h_cycles=4, l_cycles=2, aux_cycle_scale=0.2):
        super().__init__()
        assert h % heads == 0, 'hidden dimension must be divisible by attention heads'
        assert h_cycles * l_cycles <= 12, 'total recurrent block applications must stay bounded'
        self.h_cycles = h_cycles
        self.l_cycles = l_cycles
        self.aux_cycle_scale = aux_cycle_scale
        self.stem = ConvStem(in_ch, h)
        self.pos = nn.Parameter(torch.zeros(1, 49, h))
        self.answer = nn.Parameter(torch.randn(1, 1, h) * 0.02)
        self.reason = nn.Parameter(torch.randn(1, reason_tokens, h) * 0.02)
        self.block = TokenBlock(h, heads)
        self.lat_update = GatedLatentUpdate(h)
        self.out_norm = nn.LayerNorm(h)
        self.head = nn.Linear(h, num_classes)
        self.n_lat = 1 + reason_tokens

        # ChestMNIST is sparse multi-label; start recurrent cycle heads near a
        # conservative prior while leaving binary guard tasks at the default.
        if num_classes > 2:
            nn.init.constant_(self.head.bias, -1.5)

    def forward(self, x):
        b = x.shape[0]
        tok = self.stem(x) + self.pos[:, :49]
        lat0 = torch.cat([self.answer, self.reason], dim=1)
        lat = lat0.expand(b, -1, -1)
        aux_logits = []

        for cycle in range(self.h_cycles):
            seq = torch.cat([lat, tok], dim=1)
            for _ in range(self.l_cycles):
                seq = self.block(seq)
            proposal = seq[:, : self.n_lat]
            lat = self.lat_update(lat, proposal)

            if self.training and self.aux_cycle_scale > 0.0 and cycle < self.h_cycles - 1:
                aux_logits.append(self.head(self.out_norm(lat[:, 0])))

        logits = self.head(self.out_norm(lat[:, 0]))
        if self.training and aux_logits:
            aux = torch.stack(aux_logits, dim=0).mean(dim=0)
            logits = logits + self.aux_cycle_scale * (aux - aux.detach())
        return logits


def build_model(num_classes, in_ch=1):
    return RecurrentReasoner(num_classes, in_ch=in_ch, h=192, heads=6, reason_tokens=8, h_cycles=4, l_cycles=2, aux_cycle_scale=0.2)
