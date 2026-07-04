from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import medmnist
import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset
from torchvision import transforms


CHESTMNIST_NUM_LABELS = 14


@dataclass(frozen=True)
class DataConfig:
    data_root: str = "data"
    batch_size: int = 256
    num_workers: int = 4
    limit_train: Optional[int] = None
    limit_val: Optional[int] = None
    limit_test: Optional[int] = None
    seed: int = 0
    size: int = 28  # MedMNIST resolution; 28 keeps the canonical path byte-identical (no size= passed)
    dataset: str = "chestmnist"  # chestmnist (14-label) | pneumoniamnist (binary, second chest source)


@dataclass(frozen=True)
class FoldDataConfig(DataConfig):
    folds: int = 5
    fold_index: int = 0


def _subset(dataset: Dataset, limit: Optional[int], seed: int) -> Dataset:
    if limit is None or limit <= 0 or limit >= len(dataset):
        return dataset
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:limit].tolist()
    return Subset(dataset, indices)


_DATASET_CLASSES = {
    "chestmnist": "ChestMNIST",        # 14-label multilabel
    "pneumoniamnist": "PneumoniaMNIST",  # binary, second chest X-ray source (Kermany et al.)
}


def _make_dataset(split: str, root: str, train: bool, size: int = 28, dataset: str = "chestmnist") -> Dataset:
    from pathlib import Path

    Path(root).mkdir(parents=True, exist_ok=True)
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )
    cls = getattr(medmnist, _DATASET_CLASSES[dataset])
    # size=28 is the canonical path: do NOT pass size= so behavior is byte-identical to prior runs.
    extra = {} if size == 28 else {"size": size}
    return cls(split=split, root=root, transform=transform, download=True, **extra)


def chestmnist_labels(dataset: Dataset) -> np.ndarray:
    if isinstance(dataset, Subset):
        labels = chestmnist_labels(dataset.dataset)
        return labels[np.asarray(dataset.indices)]
    if isinstance(dataset, ConcatDataset):
        return np.concatenate([chestmnist_labels(ds) for ds in dataset.datasets], axis=0)
    labels = getattr(dataset, "labels", None)
    if labels is None:
        raise TypeError(f"cannot extract labels from dataset type {type(dataset)!r}")
    labels = np.asarray(labels)
    if labels.ndim == 1:
        labels = labels[:, None]
    return labels.astype(np.int64)


def iterative_multilabel_folds(labels: np.ndarray, folds: int, seed: int = 0) -> list[list[int]]:
    """Greedy multilabel stratification for sparse multilabel datasets.

    This approximates iterative stratification: examples with rarer label
    combinations are assigned first to the fold that currently needs those
    labels most. It has no external dependency and keeps each pathology's
    positives distributed across folds better than random splitting.
    """
    if folds < 2:
        raise ValueError("folds must be >= 2")
    labels = np.asarray(labels).astype(np.int64)
    rng = np.random.default_rng(seed)
    n, num_labels = labels.shape
    desired = labels.sum(axis=0, dtype=np.float64) / folds
    fold_counts = np.zeros((folds, num_labels), dtype=np.float64)
    fold_sizes = np.zeros(folds, dtype=np.int64)
    assignments: list[list[int]] = [[] for _ in range(folds)]

    label_freq = labels.sum(axis=0)
    label_freq = np.maximum(label_freq, 1)
    rarity = labels @ (1.0 / label_freq)
    cardinality = labels.sum(axis=1)
    jitter = rng.random(n) * 1e-6
    order = np.lexsort((jitter, -cardinality, -rarity))

    for idx in order:
        y = labels[idx].astype(np.float64)
        if y.sum() == 0:
            fold = int(np.argmin(fold_sizes))
        else:
            need = desired - fold_counts
            score = need @ y
            score = score - 1e-3 * fold_sizes
            fold = int(np.argmax(score))
        assignments[fold].append(int(idx))
        fold_counts[fold] += y
        fold_sizes[fold] += 1

    return assignments


def make_cv_loaders(config: FoldDataConfig) -> tuple[DataLoader, DataLoader, DataLoader]:
    if config.fold_index < 0 or config.fold_index >= config.folds:
        raise ValueError(f"fold_index must be in [0, {config.folds})")
    train_set = _make_dataset("train", config.data_root, train=True, size=config.size, dataset=config.dataset)
    val_set = _make_dataset("val", config.data_root, train=False, size=config.size, dataset=config.dataset)
    test_set = _subset(_make_dataset("test", config.data_root, train=False, size=config.size, dataset=config.dataset), config.limit_test, config.seed + 2)
    train_val_set = ConcatDataset([train_set, val_set])
    labels = chestmnist_labels(train_val_set)
    folds = iterative_multilabel_folds(labels, config.folds, seed=config.seed)
    val_indices = folds[config.fold_index]
    train_indices = [idx for fold_id, indices in enumerate(folds) if fold_id != config.fold_index for idx in indices]

    train_fold = _subset(Subset(train_val_set, train_indices), config.limit_train, config.seed)
    val_fold = _subset(Subset(train_val_set, val_indices), config.limit_val, config.seed + 1)

    train_loader = DataLoader(
        train_fold,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        persistent_workers=config.num_workers > 0,
    )
    val_loader = DataLoader(
        val_fold,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        persistent_workers=config.num_workers > 0,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        persistent_workers=config.num_workers > 0,
    )
    return train_loader, val_loader, test_loader


def make_loaders(config: DataConfig) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_set = _subset(_make_dataset("train", config.data_root, train=True, size=config.size, dataset=config.dataset), config.limit_train, config.seed)
    val_set = _subset(_make_dataset("val", config.data_root, train=False, size=config.size, dataset=config.dataset), config.limit_val, config.seed + 1)
    test_set = _subset(_make_dataset("test", config.data_root, train=False, size=config.size, dataset=config.dataset), config.limit_test, config.seed + 2)

    train_loader = DataLoader(
        train_set,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        persistent_workers=config.num_workers > 0,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        persistent_workers=config.num_workers > 0,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        persistent_workers=config.num_workers > 0,
    )
    return train_loader, val_loader, test_loader


def unpack_batch(batch: tuple[torch.Tensor, torch.Tensor], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    images, labels = batch
    images = images.to(device, non_blocking=True)
    labels = labels.to(device, non_blocking=True).to(torch.float32)
    return images, labels
