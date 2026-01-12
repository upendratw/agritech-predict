#!/usr/bin/env python3
"""
Train script for the single-stage detector.

Usage example:
python -m src.train \
  --annotations /path/to/train.json \
  --images_root /path/to/images \
  --epochs 200 \
  --batch_size 4 \
  --image_size 512 \
  --device auto \
  --lr 1e-4
"""
import argparse
import json
import os
import time
from pathlib import Path
from typing import Optional

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

# replace with your actual dataset implementation (CocoDataset in your repo)
# It should accept (annotations_json, images_root, image_size or target_size, transforms)
try:
    from src.data.coco_dataset import CocoDataset
except Exception:
    raise ImportError("Please ensure src.data.coco_dataset.CocoDataset is available")

from src.models.single_stage_model import SingleStageDetector
from src.models.single_stage_loss import SingleStageLoss


def choose_device(device_str: str) -> torch.device:
    if device_str == "auto":
        # prefer MPS on Apple Silicon, then CUDA, else CPU
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    try:
        return torch.device(device_str)
    except Exception:
        return torch.device("cpu")


def make_dataloaders(
    annotations_json: str,
    images_root: str,
    image_size: int,
    batch_size: int,
    val_fraction: float = 0.2,
):
    ds = CocoDataset(annotations_json=annotations_json, images_root=images_root, image_size=image_size, transforms=None)
    total = len(ds)
    if total == 0:
        raise RuntimeError("Dataset is empty. Check annotations and images_root.")
    val_size = max(1, int(val_fraction * total))
    train_size = total - val_size
    ds_train, ds_val = random_split(ds, [train_size, val_size])
    train_loader = DataLoader(
        ds_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        collate_fn=getattr(ds, "collate_fn", None),
    )
    val_loader = DataLoader(
        ds_val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=getattr(ds, "collate_fn", None),
    )
    return ds, train_loader, val_loader


def safe_save(obj, path: Path):
    try:
        torch.save(obj, str(path))
        print(f"Saved: {path}")
        return True
    except Exception as e:
        print(f"Failed to save {path}: {e}")
        return False


def train_loop(
    annotations_json: str,
    images_root: str,
    epochs: int = 30,
    batch_size: int = 8,
    lr: float = 1e-4,
    step_size: int = 10,
    gamma: float = 0.5,
    image_size: int = 512,
    device_str: str = "cpu",
    checkpoint_dir: str = "checkpoints",
    save_every_epoch: bool = True,
):
    device = choose_device(device_str)
    print(f"Using device: {device} (requested: {device_str})")

    print("Loading dataset and creating dataloaders...")
    ds_full, train_loader, val_loader = make_dataloaders(
        annotations_json=annotations_json, images_root=images_root, image_size=image_size, batch_size=batch_size
    )
    print(f"Dataset sizes — train: {len(train_loader.dataset)}, val: {len(val_loader.dataset)}")

    # infer number of classes from dataset (CocoDataset should expose attribute)
    num_classes = getattr(ds_full, "num_classes", None)
    if not num_classes or num_classes < 1:
        print("Warning: dataset reports zero classes — defaulting to 1 foreground class")
        num_classes = 1
    print(f"Detected number of classes (excl. background): {num_classes}")

    # model expects total classes incl background -> we keep detector num_classes = num_foreground
    # our SingleStageDetector uses num_classes (foreground count), the decoding and loss code handle shapes
    model = SingleStageDetector(num_classes=num_classes, image_size=image_size)
    model.to(device)

    criterion = SingleStageLoss(image_size=image_size, strides=model.strides)
    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    best_path = None

    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")
        model.train()
        running = {"total": 0.0, "box": 0.0, "obj": 0.0, "cls": 0.0}
        t0 = time.time()
        for imgs, targets in tqdm(train_loader, desc="train", leave=False):
            imgs = imgs.to(device)
            # Ensure targets is a list of dicts (boxes, labels) and tensors on device
            if isinstance(targets, (list, tuple)):
                for t in targets:
                    for k, v in list(t.items()):
                        if torch.is_tensor(v):
                            t[k] = v.to(device)
            optimizer.zero_grad()
            preds, _strides = model(imgs)  # preds: (B, N, 5+C)
            loss_dict = criterion(preds, targets)
            # standardize
            if isinstance(loss_dict, dict):
                total_loss = loss_dict.get("loss", loss_dict.get("total_loss", None))
            else:
                total_loss = loss_dict
            if torch.is_tensor(total_loss):
                total_loss.backward()
                optimizer.step()
                running["total"] += float(total_loss.detach().cpu().item())
            else:
                # wrap numeric value
                total_loss = torch.tensor(float(total_loss), device=device, requires_grad=True)
                total_loss.backward()
                optimizer.step()
                running["total"] += float(total_loss.detach().cpu().item())

            if isinstance(loss_dict, dict):
                running["box"] += float(loss_dict.get("box_loss", 0.0))
                running["obj"] += float(loss_dict.get("obj_loss", 0.0))
                running["cls"] += float(loss_dict.get("cls_loss", 0.0))

        elapsed = time.time() - t0
        train_batches = max(1, len(train_loader))
        print(
            f"Epoch {epoch}/{epochs} train — total_loss: {running['total'] / train_batches:.4f}, "
            f"box: {running['box'] / train_batches:.4f}, obj: {running['obj'] / train_batches:.4f}, cls: {running['cls'] / train_batches:.4f} — time: {elapsed:.1f}s"
        )

        # validation
        model.eval()
        val_running = {"total": 0.0}
        with torch.no_grad():
            for imgs, targets in tqdm(val_loader, desc="val", leave=False):
                imgs = imgs.to(device)
                if isinstance(targets, (list, tuple)):
                    for t in targets:
                        for k, v in list(t.items()):
                            if torch.is_tensor(v):
                                t[k] = v.to(device)
                preds, _ = model(imgs)
                loss_dict = criterion(preds, targets)
                if isinstance(loss_dict, dict):
                    v = loss_dict.get("loss", loss_dict.get("total_loss", None))
                else:
                    v = loss_dict
                if torch.is_tensor(v):
                    val_running["total"] += float(v.detach().cpu().item())
                else:
                    val_running["total"] += float(v)

        val_batches = max(1, len(val_loader))
        val_total = val_running["total"] / val_batches
        print(f"Epoch {epoch}/{epochs} val   — total_loss: {val_total:.4f}")
        scheduler.step()

        # optional checkpointing
        if save_every_epoch:
            ckpt_path = ckpt_dir / f"epoch_{epoch:04d}.pth"
            checkpoint_obj = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "val_loss": val_total,
            }
            safe_save(checkpoint_obj, ckpt_path)

        if val_total < best_val_loss:
            best_val_loss = val_total
            best_path = ckpt_dir / "best_model.pth"
            try:
                torch.save(model.state_dict(), str(best_path))
                print(f"Saved best model state dict to: {best_path}")
            except Exception as e:
                print(f"Failed to save best model: {e}")

    # final save
    final_model = ckpt_dir / "final_model.pth"
    try:
        torch.save(model.state_dict(), str(final_model))
        print(f"Saved final model state dict to: {final_model}")
    except Exception as e:
        print(f"Failed to save final model: {e}")

    final_ckpt = ckpt_dir / "final_checkpoint.pth"
    try:
        torch.save(
            {
                "epoch": epochs,
                "model_state": model.state_dict(),
                "val_loss": val_total,
            },
            str(final_ckpt),
        )
        print(f"Saved final checkpoint to: {final_ckpt}")
    except Exception as e:
        print(f"Failed to save final checkpoint: {e}")

    print(f"Training complete. Best val loss: {best_val_loss:.6f}. Best path: {best_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train single-stage detector")
    parser.add_argument("--annotations", required=True, help="Path to annotations JSON")
    parser.add_argument("--images_root", required=True, help="Images root folder")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--step_size", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--device", type=str, default="cpu", help="cpu | mps | cuda:0 | auto")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--no_epoch_ckpt", action="store_true", help="Don't save epoch-by-epoch checkpoints")
    args = parser.parse_args()

    train_loop(
        annotations_json=args.annotations,
        images_root=args.images_root,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        step_size=args.step_size,
        gamma=args.gamma,
        image_size=args.image_size,
        device_str=args.device,
        checkpoint_dir=args.checkpoint_dir,
        save_every_epoch=(not args.no_epoch_ckpt),
    )