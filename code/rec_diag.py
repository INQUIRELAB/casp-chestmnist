"""Instrumented copy of the recursive model for the per-cycle diagnostic.

RecurrentReasonerDiag subclasses best_rec.RecurrentReasoner and adds a
forward_diag(x) that returns, for each outer cycle c=1..h_cycles:
  - the intermediate answer LOGITS at that cycle (head(out_norm(answer_latent)))
  - the answer-latent STATE at that cycle (lat[:, 0])

CRITICAL: the standard forward() is inherited UNCHANGED, so training behaviour is
byte-identical to the incumbent build_model. forward_diag reproduces the exact
recurrent computation of the parent forward (same block/lat_update calls, same
order), only additionally recording per-cycle readouts. It is eval-only
(no aux-cycle gradient term), matching how the parent forward behaves at eval
(self.training is False, so the aux term is skipped there too).
"""
import importlib.util
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent


def _load(path):
    spec = importlib.util.spec_from_file_location("best_rec_diag", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


REC = _load(HERE / "best_rec.py")


class RecurrentReasonerDiag(REC.RecurrentReasoner):
    @torch.no_grad()
    def forward_diag(self, x):
        """Single forward pass returning per-cycle intermediate logits and answer
        latent states. Mirrors the parent forward's recurrent loop exactly.

        Returns:
          logits_per_cycle: list[Tensor[b, num_classes]] length h_cycles
          answer_states:    list[Tensor[b, h]] length h_cycles (post lat_update)
          final_logits:     Tensor[b, num_classes] (== logits_per_cycle[-1] at eval)
        """
        b = x.shape[0]
        tok = self.stem(x) + self.pos[:, :49]
        lat0 = torch.cat([self.answer, self.reason], dim=1)
        lat = lat0.expand(b, -1, -1)

        logits_per_cycle = []
        answer_states = []
        for cycle in range(self.h_cycles):
            seq = torch.cat([lat, tok], dim=1)
            for _ in range(self.l_cycles):
                seq = self.block(seq)
            proposal = seq[:, : self.n_lat]
            lat = self.lat_update(lat, proposal)
            ans = lat[:, 0]
            logits_per_cycle.append(self.head(self.out_norm(ans)))
            answer_states.append(ans)

        # At eval the parent returns head(out_norm(lat[:,0])) with no aux term,
        # which is exactly logits_per_cycle[-1].
        final_logits = logits_per_cycle[-1]
        return logits_per_cycle, answer_states, final_logits


def build_model(num_classes, in_ch=1):
    # Same hyperparameters as best_rec.build_model.
    return RecurrentReasonerDiag(num_classes, in_ch=in_ch, h=192, heads=6,
                                 reason_tokens=8, h_cycles=4, l_cycles=2,
                                 aux_cycle_scale=0.2)


def _selfcheck():
    m = build_model(14, 1)
    m.eval()
    x = torch.zeros(2, 1, 28, 28)
    # standard forward (training contract) still works
    tr_out = m(x)
    lpc, states, final = m.forward_diag(x)
    print("standard forward shape:", tuple(tr_out.shape))
    print("n cycles:", len(lpc), "logit shape:", tuple(lpc[0].shape),
          "state shape:", tuple(states[0].shape))
    print("final==last cycle:", bool(torch.allclose(final, lpc[-1])))
    # eval-mode standard forward should equal last-cycle diag logits
    print("eval forward == diag final:", bool(torch.allclose(m(x), final, atol=1e-5)))


if __name__ == "__main__":
    _selfcheck()
