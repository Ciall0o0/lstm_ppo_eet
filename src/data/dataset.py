"""Data loading, train/val/test split, and PyTorch Dataset for elevator data."""

from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedShuffleSplit

DATASETS_DIR = Path(__file__).resolve().parents[2] / "datasets"

SCENARIO_NAMES = {
    1: "morning_peak", 2: "evening_peak", 3: "noon_cross",
    4: "interfloor", 5: "extreme_idle", 6: "meeting_start", 7: "meeting_scatter",
}


def load_raw_data(data_dir: Path | str = DATASETS_DIR) -> dict:
    """Load all NPZ files from datasets directory."""
    data_dir = Path(data_dir)
    data = {}
    for fname in ["global_features.npz", "event_sequences.npz",
                   "labels.npz", "file_ids.npz", "event_lengths.npz"]:
        path = data_dir / fname
        if path.exists():
            loaded = np.load(path, allow_pickle=True)
            data[fname.replace(".npz", "")] = dict(loaded)
    return data


def split_indices(labels: np.ndarray, train_ratio: float = 0.70,
                  val_ratio: float = 0.15, seed: int = 42) -> tuple:
    """Stratified split into train/val/test index arrays."""
    n = len(labels)
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio + (1 - train_ratio - val_ratio),
                                   random_state=seed)
    train_idx, temp_idx = next(sss1.split(np.zeros(n), labels))

    temp_labels = labels[temp_idx]
    val_frac = val_ratio / (val_ratio + (1 - train_ratio - val_ratio))
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=1 - val_frac, random_state=seed)
    val_idx_rel, test_idx_rel = next(sss2.split(np.zeros(len(temp_idx)), temp_labels))
    val_idx = temp_idx[val_idx_rel]
    test_idx = temp_idx[test_idx_rel]
    return train_idx, val_idx, test_idx


class ElevatorDataset(Dataset):
    """PyTorch Dataset for elevator event sequences."""

    def __init__(self, event_sequences: np.ndarray, event_lengths: np.ndarray,
                 labels: np.ndarray, indices: np.ndarray):
        self.sequences = event_sequences[indices]
        self.lengths = event_lengths[indices]
        self.labels = labels[indices]
        self.indices = indices

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = torch.tensor(self.sequences[idx], dtype=torch.float32)
        length = int(self.lengths[idx])
        label = int(self.labels[idx])
        mask = torch.zeros(len(seq), dtype=torch.bool)
        mask[:length] = True
        return seq, mask, label


def collate_fn(batch):
    """Collate variable-length sequences by padding to max length in batch."""
    sequences, masks, labels = zip(*batch)
    max_len = max(s.shape[0] for s in sequences)
    feat_dim = sequences[0].shape[1]
    padded = torch.zeros(len(sequences), max_len, feat_dim)
    padded_masks = torch.zeros(len(sequences), max_len, dtype=torch.bool)
    for i, (s, m) in enumerate(zip(sequences, masks)):
        padded[i, :s.shape[0]] = s
        padded_masks[i, :s.shape[0]] = m
    return padded, padded_masks, torch.tensor(labels, dtype=torch.long)


def create_dataloaders(data_dir: str = DATASETS_DIR, train_ratio: float = 0.70,
                       val_ratio: float = 0.15, batch_size: int = 32,
                       seed: int = 42) -> dict:
    """Load data, split, and return train/val/test DataLoaders."""
    raw = load_raw_data(data_dir)
    labels = raw["labels"]["arr_0"]

    # Handle 2D labels by squeezing
    labels = np.squeeze(labels)

    event_seqs = raw["event_sequences"]["arr_0"]
    event_lens = raw["event_lengths"]["arr_0"]
    file_ids = raw["file_ids"]["arr_0"]

    train_idx, val_idx, test_idx = split_indices(labels, train_ratio, val_ratio, seed)

    train_ds = ElevatorDataset(event_seqs, event_lens, labels, train_idx)
    val_ds = ElevatorDataset(event_seqs, event_lens, labels, val_idx)
    test_ds = ElevatorDataset(event_seqs, event_lens, labels, test_idx)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    return {
        "train": train_loader, "val": val_loader, "test": test_loader,
        "train_idx": train_idx, "val_idx": val_idx, "test_idx": test_idx,
        "labels": labels, "file_ids": file_ids,
    }


if __name__ == "__main__":
    result = create_dataloaders(batch_size=8, seed=42)
    print(f"Total files: {len(result['labels'])}")
    print(f"Train: {len(result['train_idx'])}  Val: {len(result['val_idx'])}  Test: {len(result['test_idx'])}")
    print(f"Label distribution — Train: {np.bincount(result['labels'][result['train_idx']])}")
    print(f"Label distribution — Val:   {np.bincount(result['labels'][result['val_idx']])}")
    print(f"Label distribution — Test:  {np.bincount(result['labels'][result['test_idx']])}")

    seqs, masks, lbls = next(iter(result["train"]))
    print(f"\nSample batch: seqs={seqs.shape}, masks={masks.shape}, labels={lbls}")
