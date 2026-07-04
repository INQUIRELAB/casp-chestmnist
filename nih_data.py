"""NIH ChestX-ray14 loaders with patient-grouped splits (bundle copy).

Official patient-disjoint test split (from test_list.txt). Train/validation split is made by
PATIENT inside the official train_val set (no patient appears in both). Returns the test
patient-id array so the analysis can run a patient-clustered bootstrap. Reads nih{size}_*
from the local nihdata/ directory next to this file.
"""
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

DATA = Path(__file__).resolve().parent / "nihdata"


class NIHSplit(Dataset):
    def __init__(self, images, labels, idx):
        self.images = images; self.labels = labels; self.idx = idx

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, k):
        i = int(self.idx[k])
        x = torch.from_numpy(np.asarray(self.images[i], dtype=np.float32)) / 255.0
        x = (x - 0.5) / 0.5
        return x.unsqueeze(0), torch.from_numpy(self.labels[i].astype(np.float32))


def make_nih_loaders(size, seed=0, batch_size=256, val_frac=0.1, limit_train_patients=None,
                     num_workers=0):
    images = np.load(DATA / f"nih{size}_images.npy", mmap_mode="r")
    m = np.load(DATA / f"nih{size}_meta.npz", allow_pickle=True)
    labels = m["labels"].astype(np.int8); patient = m["patient"]; is_test = m["is_test"]
    tv_idx = np.where(~is_test)[0]; te_idx = np.where(is_test)[0]

    tv_patients = np.unique(patient[tv_idx])
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(tv_patients)); tv_patients = tv_patients[perm]
    n_val = max(1, int(len(tv_patients) * val_frac))
    val_pat = set(tv_patients[:n_val].tolist())
    train_pat_all = tv_patients[n_val:]
    if limit_train_patients and limit_train_patients < len(train_pat_all):
        train_pat = set(train_pat_all[:limit_train_patients].tolist())
    else:
        train_pat = set(train_pat_all.tolist())
    pat_tv = patient[tv_idx]
    train_idx = tv_idx[np.array([p in train_pat for p in pat_tv])]
    val_idx = tv_idx[np.array([p in val_pat for p in pat_tv])]

    def dl(idx, shuffle):
        return DataLoader(NIHSplit(images, labels, idx), batch_size=batch_size, shuffle=shuffle,
                          num_workers=num_workers, pin_memory=True)

    info = dict(n_train=len(train_idx), n_val=len(val_idx), n_test=len(te_idx),
                train_patients=len(train_pat), val_patients=len(val_pat),
                test_patients=int(len(np.unique(patient[te_idx]))))
    return (dl(train_idx, True), dl(val_idx, False), dl(te_idx, False),
            patient[te_idx].astype(np.int64), labels[te_idx].astype(np.int8), info)
