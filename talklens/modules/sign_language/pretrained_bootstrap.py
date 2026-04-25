"""
Pretrained Sign Language Model Bootstrap
─────────────────────────────────────────
Generates a pretrained sign classifier by:
  1. Creating synthetic landmark templates for ASL A-Z hand poses
  2. Augmenting with noise to create a training corpus
  3. Training the BiLSTM model
  4. Saving weights + labels to the standard paths

Usage:
    python -m modules.sign_language.pretrained_bootstrap
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

# ── Path setup ─────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from modules.sign_language.model import LandmarkLSTM


# ══════════════════════════════════════════════════════════════════════════════
# ASL Landmark Templates
# ══════════════════════════════════════════════════════════════════════════════
# Each template is a 21×3 array representing (x, y, z) for MediaPipe's
# 21 hand landmarks in a canonical ASL fingerspelling pose.
# These are approximate but geometrically consistent templates derived from
# anatomical hand proportions.

def _wrist():
    return [0.5, 0.8, 0.0]

def _make_finger(base_x, base_y, tip_x, tip_y, extended=True, curled=False, z_base=0.0):
    """Generate 4 joints for a single finger (MCP, PIP, DIP, TIP)."""
    if curled:
        return [
            [base_x, base_y, z_base],
            [base_x, base_y - 0.02, z_base - 0.02],
            [base_x + 0.005, base_y + 0.01, z_base - 0.03],
            [base_x + 0.01, base_y + 0.02, z_base - 0.02],
        ]
    elif extended:
        dx = (tip_x - base_x) / 3
        dy = (tip_y - base_y) / 3
        return [
            [base_x, base_y, z_base],
            [base_x + dx, base_y + dy, z_base - 0.01],
            [base_x + 2*dx, base_y + 2*dy, z_base - 0.015],
            [tip_x, tip_y, z_base - 0.02],
        ]
    else:  # half bent
        dx = (tip_x - base_x) / 3
        dy = (tip_y - base_y) / 6
        return [
            [base_x, base_y, z_base],
            [base_x + dx * 0.5, base_y + dy, z_base - 0.015],
            [base_x + dx * 0.3, base_y + dy * 0.5, z_base - 0.025],
            [base_x + dx * 0.2, base_y + dy * 0.3, z_base - 0.02],
        ]


def _thumb(extended=True, across=False, up=False):
    """Generate 4 joints for the thumb (CMC, MCP, IP, TIP)."""
    if across:
        return [
            [0.42, 0.75, 0.0],
            [0.44, 0.68, -0.02],
            [0.48, 0.62, -0.03],
            [0.50, 0.58, -0.035],
        ]
    elif up:
        return [
            [0.42, 0.75, 0.0],
            [0.40, 0.68, -0.02],
            [0.38, 0.58, -0.03],
            [0.37, 0.48, -0.035],
        ]
    elif extended:
        return [
            [0.42, 0.75, 0.0],
            [0.36, 0.70, -0.02],
            [0.30, 0.65, -0.03],
            [0.25, 0.60, -0.035],
        ]
    else:
        return [
            [0.42, 0.75, 0.0],
            [0.44, 0.72, -0.02],
            [0.46, 0.70, -0.035],
            [0.47, 0.68, -0.04],
        ]


def _build_hand(thumb_kw, index_ext, middle_ext, ring_ext, pinky_ext,
                index_curl=False, middle_curl=False, ring_curl=False, pinky_curl=False):
    """
    Build a 21-landmark hand from component specifications.
    Order follows MediaPipe convention:
      0: Wrist
      1-4: Thumb (CMC, MCP, IP, TIP)
      5-8: Index (MCP, PIP, DIP, TIP)
      9-12: Middle (MCP, PIP, DIP, TIP)
      13-16: Ring (MCP, PIP, DIP, TIP)
      17-20: Pinky (MCP, PIP, DIP, TIP)
    """
    landmarks = [_wrist()]
    landmarks.extend(_thumb(**thumb_kw))
    landmarks.extend(_make_finger(0.44, 0.65, 0.44, 0.35,
                                   extended=index_ext, curled=index_curl))
    landmarks.extend(_make_finger(0.50, 0.63, 0.50, 0.33,
                                   extended=middle_ext, curled=middle_curl))
    landmarks.extend(_make_finger(0.56, 0.65, 0.56, 0.35,
                                   extended=ring_ext, curled=ring_curl))
    landmarks.extend(_make_finger(0.62, 0.68, 0.62, 0.42,
                                   extended=pinky_ext, curled=pinky_curl))
    return np.array(landmarks, dtype=np.float32)


# ── ASL A–Z pose definitions ──────────────────────────────────────────────────
# Each letter is defined by which fingers are extended/curled and thumb position.
# This creates distinct geometric patterns the LSTM can learn to differentiate.

ASL_TEMPLATES: dict[str, np.ndarray] = {}

def _register_templates():
    """Register all 26 ASL letter templates."""
    global ASL_TEMPLATES

    # A: Fist with thumb alongside
    ASL_TEMPLATES["A"] = _build_hand(
        thumb_kw={"extended": False}, index_ext=False, middle_ext=False,
        ring_ext=False, pinky_ext=False,
        index_curl=True, middle_curl=True, ring_curl=True, pinky_curl=True,
    )

    # B: Flat hand, fingers up, thumb across palm
    ASL_TEMPLATES["B"] = _build_hand(
        thumb_kw={"across": True}, index_ext=True, middle_ext=True,
        ring_ext=True, pinky_ext=True,
    )

    # C: Curved hand (C shape)
    ASL_TEMPLATES["C"] = _build_hand(
        thumb_kw={"extended": True}, index_ext=False, middle_ext=False,
        ring_ext=False, pinky_ext=False,
    )

    # D: Index up, others curled, thumb touching middle
    ASL_TEMPLATES["D"] = _build_hand(
        thumb_kw={"across": True}, index_ext=True, middle_ext=False,
        ring_ext=False, pinky_ext=False,
        middle_curl=True, ring_curl=True, pinky_curl=True,
    )

    # E: All fingers curled, thumb across
    ASL_TEMPLATES["E"] = _build_hand(
        thumb_kw={"across": True}, index_ext=False, middle_ext=False,
        ring_ext=False, pinky_ext=False,
        index_curl=True, middle_curl=True, ring_curl=True, pinky_curl=True,
    )

    # F: OK sign – index & thumb touching, others extended
    ASL_TEMPLATES["F"] = _build_hand(
        thumb_kw={"across": True}, index_ext=False, middle_ext=True,
        ring_ext=True, pinky_ext=True,
    )

    # G: Index pointing sideways, thumb parallel
    t = _build_hand(
        thumb_kw={"extended": True}, index_ext=True, middle_ext=False,
        ring_ext=False, pinky_ext=False,
        middle_curl=True, ring_curl=True, pinky_curl=True,
    )
    # Rotate hand 90° to point sideways
    t[:, 0], t[:, 1] = t[:, 1].copy(), 1.0 - t[:, 0].copy()
    ASL_TEMPLATES["G"] = t

    # H: Index + middle pointing sideways
    t = _build_hand(
        thumb_kw={"across": True}, index_ext=True, middle_ext=True,
        ring_ext=False, pinky_ext=False,
        ring_curl=True, pinky_curl=True,
    )
    t[:, 0], t[:, 1] = t[:, 1].copy(), 1.0 - t[:, 0].copy()
    ASL_TEMPLATES["H"] = t

    # I: Pinky up only
    ASL_TEMPLATES["I"] = _build_hand(
        thumb_kw={"extended": False}, index_ext=False, middle_ext=False,
        ring_ext=False, pinky_ext=True,
        index_curl=True, middle_curl=True, ring_curl=True,
    )

    # J: Pinky up with J-motion (represented as I with slight offset)
    j = ASL_TEMPLATES["I"].copy()
    j[:, 0] += 0.03
    j[:, 2] -= 0.02
    ASL_TEMPLATES["J"] = j

    # K: Index + middle up, spread, thumb between
    ASL_TEMPLATES["K"] = _build_hand(
        thumb_kw={"up": True}, index_ext=True, middle_ext=True,
        ring_ext=False, pinky_ext=False,
        ring_curl=True, pinky_curl=True,
    )

    # L: L shape – index up, thumb out
    ASL_TEMPLATES["L"] = _build_hand(
        thumb_kw={"extended": True}, index_ext=True, middle_ext=False,
        ring_ext=False, pinky_ext=False,
        middle_curl=True, ring_curl=True, pinky_curl=True,
    )

    # M: Three fingers over thumb
    ASL_TEMPLATES["M"] = _build_hand(
        thumb_kw={"extended": False}, index_ext=False, middle_ext=False,
        ring_ext=False, pinky_ext=False,
        index_curl=True, middle_curl=True, ring_curl=True, pinky_curl=True,
    )
    ASL_TEMPLATES["M"][1:5, 2] -= 0.05  # thumb tucked deeper

    # N: Two fingers over thumb
    n = ASL_TEMPLATES["M"].copy()
    n[13:17, 1] += 0.03  # ring slightly different
    ASL_TEMPLATES["N"] = n

    # O: All fingers curved to meet thumb (O shape)
    ASL_TEMPLATES["O"] = _build_hand(
        thumb_kw={"across": True}, index_ext=False, middle_ext=False,
        ring_ext=False, pinky_ext=False,
    )
    ASL_TEMPLATES["O"][5:21, 2] -= 0.03  # fingers curve forward

    # P: K but pointing down
    p = ASL_TEMPLATES["K"].copy()
    p[:, 1] = 1.0 - p[:, 1]  # flip vertically
    ASL_TEMPLATES["P"] = p

    # Q: G but pointing down
    q = ASL_TEMPLATES["G"].copy()
    q[:, 1] = 1.0 - q[:, 1]
    ASL_TEMPLATES["Q"] = q

    # R: Index + middle crossed
    ASL_TEMPLATES["R"] = _build_hand(
        thumb_kw={"across": True}, index_ext=True, middle_ext=True,
        ring_ext=False, pinky_ext=False,
        ring_curl=True, pinky_curl=True,
    )
    # Cross index over middle
    ASL_TEMPLATES["R"][5:9, 0] += 0.02
    ASL_TEMPLATES["R"][9:13, 0] -= 0.02

    # S: Fist with thumb in front
    ASL_TEMPLATES["S"] = _build_hand(
        thumb_kw={"across": True}, index_ext=False, middle_ext=False,
        ring_ext=False, pinky_ext=False,
        index_curl=True, middle_curl=True, ring_curl=True, pinky_curl=True,
    )

    # T: Fist, thumb between index and middle
    ASL_TEMPLATES["T"] = _build_hand(
        thumb_kw={"up": True}, index_ext=False, middle_ext=False,
        ring_ext=False, pinky_ext=False,
        index_curl=True, middle_curl=True, ring_curl=True, pinky_curl=True,
    )
    ASL_TEMPLATES["T"][1:5, 0] = 0.47  # thumb between index/middle

    # U: Index + middle up together
    ASL_TEMPLATES["U"] = _build_hand(
        thumb_kw={"extended": False}, index_ext=True, middle_ext=True,
        ring_ext=False, pinky_ext=False,
        ring_curl=True, pinky_curl=True,
    )

    # V: Peace sign
    ASL_TEMPLATES["V"] = _build_hand(
        thumb_kw={"extended": False}, index_ext=True, middle_ext=True,
        ring_ext=False, pinky_ext=False,
        ring_curl=True, pinky_curl=True,
    )
    # Spread index and middle apart
    ASL_TEMPLATES["V"][5:9, 0] -= 0.04
    ASL_TEMPLATES["V"][9:13, 0] += 0.04

    # W: Three fingers up, spread
    ASL_TEMPLATES["W"] = _build_hand(
        thumb_kw={"extended": False}, index_ext=True, middle_ext=True,
        ring_ext=True, pinky_ext=False,
        pinky_curl=True,
    )
    ASL_TEMPLATES["W"][5:9, 0] -= 0.03
    ASL_TEMPLATES["W"][13:17, 0] += 0.03

    # X: Index hooked/bent
    ASL_TEMPLATES["X"] = _build_hand(
        thumb_kw={"extended": False}, index_ext=False, middle_ext=False,
        ring_ext=False, pinky_ext=False,
        middle_curl=True, ring_curl=True, pinky_curl=True,
    )

    # Y: Thumb + pinky out (hang loose)
    ASL_TEMPLATES["Y"] = _build_hand(
        thumb_kw={"extended": True}, index_ext=False, middle_ext=False,
        ring_ext=False, pinky_ext=True,
        index_curl=True, middle_curl=True, ring_curl=True,
    )

    # Z: Index traces Z (static = index pointing)
    ASL_TEMPLATES["Z"] = _build_hand(
        thumb_kw={"extended": False}, index_ext=True, middle_ext=False,
        ring_ext=False, pinky_ext=False,
        middle_curl=True, ring_curl=True, pinky_curl=True,
    )
    ASL_TEMPLATES["Z"][5:9, 2] -= 0.04  # index angled differently

_register_templates()


# ══════════════════════════════════════════════════════════════════════════════
# Synthetic Dataset Generation
# ══════════════════════════════════════════════════════════════════════════════

def augment_landmarks(template: np.ndarray, noise_std: float = 0.015,
                       scale_range: tuple = (0.85, 1.15),
                       shift_range: float = 0.08) -> np.ndarray:
    """Apply random noise, scaling, and translation to a template."""
    aug = template.copy()
    # Random noise
    aug += np.random.normal(0, noise_std, aug.shape).astype(np.float32)
    # Random scale
    scale = np.random.uniform(*scale_range)
    center = aug.mean(axis=0)
    aug = (aug - center) * scale + center
    # Random translation
    shift = np.random.uniform(-shift_range, shift_range, size=3).astype(np.float32)
    aug += shift
    return aug


def generate_sequence(template: np.ndarray, seq_len: int = 30,
                       noise_std: float = 0.012) -> np.ndarray:
    """Generate a sequence of frames from a template with per-frame jitter."""
    base = augment_landmarks(template, noise_std=0.015)
    frames = []
    for _ in range(seq_len):
        frame = base.copy()
        # Per-frame micro-jitter (simulates hand tremor)
        frame += np.random.normal(0, noise_std * 0.3, frame.shape).astype(np.float32)
        frames.append(frame.flatten())  # → (63,)
    return np.array(frames, dtype=np.float32)  # → (seq_len, 63)


class SyntheticSignDataset(Dataset):
    """Generates synthetic sign language landmark sequences on the fly."""

    def __init__(self, templates: dict[str, np.ndarray], samples_per_class: int = 500,
                 seq_len: int = 30):
        self.samples: list[tuple[np.ndarray, int]] = []
        self.labels = {i: letter for i, letter in enumerate(sorted(templates.keys()))}
        inv_labels = {v: k for k, v in self.labels.items()}

        for letter, template in templates.items():
            label_idx = inv_labels[letter]
            for _ in range(samples_per_class):
                seq = generate_sequence(template, seq_len)
                self.samples.append((seq, label_idx))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        seq, label = self.samples[idx]
        return torch.from_numpy(seq), label


# ══════════════════════════════════════════════════════════════════════════════
# Training
# ══════════════════════════════════════════════════════════════════════════════

def train_pretrained(
    samples_per_class: int = 600,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-3,
    seq_len: int = 30,
) -> None:
    """Train the BiLSTM on synthetic data and save the model + labels."""
    model_out  = ROOT / "models" / "sign_classifier.pth"
    labels_out = ROOT / "data" / "processed" / "labels.json"
    model_out.parent.mkdir(parents=True, exist_ok=True)
    labels_out.parent.mkdir(parents=True, exist_ok=True)

    print("═" * 60)
    print("  TalkLens – Pretrained Sign Language Model Bootstrap")
    print("═" * 60)

    # Build dataset
    print(f"\n📦 Generating synthetic dataset ({samples_per_class} samples × 26 classes)…")
    dataset = SyntheticSignDataset(ASL_TEMPLATES, samples_per_class, seq_len)
    n_val   = max(1, int(0.15 * len(dataset)))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)

    # Save labels
    labels = dataset.labels
    with open(labels_out, "w") as f:
        json.dump(labels, f, indent=2)
    print(f"📝 Labels saved → {labels_out}")
    print(f"   Classes: {list(labels.values())}")

    # Build model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = LandmarkLSTM(input_size=63, num_classes=26).to(device)
    print(f"\n🧠 Model: BiLSTM  |  Device: {device}")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_val_acc = 0.0
    print(f"\n🏋️ Training for {epochs} epochs…\n")

    for epoch in range(1, epochs + 1):
        # ── Train ──
        model.train()
        train_loss, correct, total = 0.0, 0, 0
        for seqs, targets in tqdm(train_dl, desc=f"Epoch {epoch:>3}/{epochs} [train]",
                                   leave=False):
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

        # ── Validate ──
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for seqs, targets in val_dl:
                seqs, targets = seqs.to(device), targets.to(device)
                logits      = model(seqs)
                val_loss   += criterion(logits, targets).item() * len(seqs)
                val_correct += (logits.argmax(1) == targets).sum().item()
                val_total  += len(seqs)
        val_acc = val_correct / val_total

        if epoch % 5 == 0 or epoch == 1 or val_acc > best_val_acc:
            print(
                f"  Epoch {epoch:>3}: "
                f"loss={train_loss/total:.4f}  acc={train_acc:.3f}  "
                f"val_loss={val_loss/val_total:.4f}  val_acc={val_acc:.3f}"
            )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), str(model_out))
            print(f"  ✓ Best model saved → val_acc={best_val_acc:.3f}")

    print(f"\n{'═' * 60}")
    print(f"  ✅ Training complete!")
    print(f"  Best validation accuracy: {best_val_acc:.3f}")
    print(f"  Model saved at: {model_out}")
    print(f"  Labels saved at: {labels_out}")
    print(f"{'═' * 60}")


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Bootstrap pretrained sign language model.")
    p.add_argument("--samples", type=int, default=600, help="Samples per ASL letter class")
    p.add_argument("--epochs",  type=int, default=50,  help="Training epochs")
    p.add_argument("--batch",   type=int, default=64,  help="Batch size")
    p.add_argument("--lr",      type=float, default=1e-3, help="Learning rate")
    args = p.parse_args()
    train_pretrained(
        samples_per_class=args.samples,
        epochs=args.epochs,
        batch_size=args.batch,
        lr=args.lr,
    )
