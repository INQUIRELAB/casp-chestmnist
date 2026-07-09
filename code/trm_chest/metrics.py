from __future__ import annotations

import numpy as np
import torch


def sigmoid_np(logits: np.ndarray) -> np.ndarray:
    logits = np.clip(logits, -80.0, 80.0)
    return 1.0 / (1.0 + np.exp(-logits))


def binary_accuracy_from_logits(logits: np.ndarray, labels: np.ndarray, threshold: float = 0.5) -> float:
    probs = sigmoid_np(logits)
    preds = probs >= threshold
    return float((preds == labels.astype(bool)).mean())


def _binary_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    labels = labels.astype(np.int64)
    pos = int(labels.sum())
    neg = int(labels.size - pos)
    if pos == 0 or neg == 0:
        return float("nan")

    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty_like(order, dtype=np.float64)

    start = 0
    n = scores.size
    while start < n:
        end = start + 1
        while end < n and sorted_scores[end] == sorted_scores[start]:
            end += 1
        avg_rank = 0.5 * (start + 1 + end)
        ranks[order[start:end]] = avg_rank
        start = end

    pos_rank_sum = ranks[labels == 1].sum()
    auc = (pos_rank_sum - pos * (pos + 1) / 2.0) / (pos * neg)
    return float(auc)


def macro_auc_from_logits(logits: np.ndarray, labels: np.ndarray) -> float:
    probs = sigmoid_np(logits)
    aucs = [_binary_auc(probs[:, i], labels[:, i]) for i in range(labels.shape[1])]
    finite = [x for x in aucs if np.isfinite(x)]
    return float(np.mean(finite)) if finite else float("nan")


def multilabel_loss_per_sample(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    return loss.mean(dim=-1)

