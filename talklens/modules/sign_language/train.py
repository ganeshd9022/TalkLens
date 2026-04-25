"""
Sign Language Model Training Script
─────────────────────────────────────
Loads raw .npy sequence files → trains BiLSTM → saves to models/sign_classifier.pth
Usage:
    python train.py --data data/raw --epochs 80 --lr 1e-3 --batch 32
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from modules.sign_language.model import LandmarkLSTM


# ── Dataset ────────────────────────────────────────────────────────────────────

class SignDataset(Dataset):
    def __init__(self, data_dir: Path, labels: dict[int, str]) -> None:
        self.samples: list[Tuple[np.ndarray, int]] = []
        inv_labels = {v: k for k, v in labels.items()}
        for sign_dir in sorted(data_dir.iterdir()):
            if not sign_dir.is_dir():
                continue
            label = inv_labels.get(sign_dir.name)
            if label is None:
                continue
            for npy in sign_dir.glob("*.npy"):
                seq = np.load(str(npy)).astype(np.float32)
                self.samples.append((seq, label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        seq, label = self.samples[idx]
        return torch.from_numpy(seq), label


# ── Training ───────────────────────────────────────────────────────────────────

def train(args: argparse.Namespace) -> None:
    data_dir   = Path(args.data)
    labels_path = data_dir.parent / "processed" / "labels.json"
    model_out  = Path("models") / "sign_classifier.pth"
    model_out.parent.mkdir(parents=True, exist_ok=True)

    with open(labels_path) as f:
        labels: dict = {int(k): v for k, v in json.load(f).items()}
    num_classes = len(labels)
    print(f"Classes ({num_classes}): {list(labels.values())}")

    dataset = SignDataset(data_dir, labels)
    n_val   = max(1, int(0.15 * len(dataset)))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True,  num_workers=0)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = LandmarkLSTM(num_classes=num_classes).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        # ── Train ──────────────────────────────────────────────────────────────
        model.train()
        train_loss, correct, total = 0.0, 0, 0
        for seqs, targets in tqdm(train_dl, desc=f"Epoch {epoch:>3}/{args.epochs} [train]"):
            seqs, targets = seqs.to(device), targets.to(device)
            optimizer.zero_grad()
            logits = model(seqs)
            loss   = criterion(logits, targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * len(seqs)
            correct    += (logits.argmax(1) == targets).sum().item()
            total      += len(seqs)
        scheduler.step()
        train_acc = correct / total

        # ── Validate ───────────────────────────────────────────────────────────
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for seqs, targets in val_dl:
                seqs, targets = seqs.to(device), targets.to(device)
                logits     = model(seqs)
                val_loss  += criterion(logits, targets).item() * len(seqs)
                val_correct += (logits.argmax(1) == targets).sum().item()
                val_total  += len(seqs)
        val_acc = val_correct / val_total

        print(
            f"  loss={train_loss/total:.4f}  acc={train_acc:.3f}  "
            f"val_loss={val_loss/val_total:.4f}  val_acc={val_acc:.3f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), str(model_out))
            print(f"  ✓ Best model saved → val_acc={best_val_acc:.3f}")

    print(f"\nTraining complete. Best val accuracy: {best_val_acc:.3f}")
    print(f"Model saved at: {model_out}")


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train TalkLens sign language classifier.")
    p.add_argument("--data",   type=str,   default="data/raw")
    p.add_argument("--epochs", type=int,   default=80)
    p.add_argument("--lr",     type=float, default=1e-3)
    p.add_argument("--batch",  type=int,   default=32)
    train(p.parse_args())
