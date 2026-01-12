# src/eval.py
import os
import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# internal imports (use package-style imports if running as `python -m src.eval`)
from src.data.coco_dataset import CocoDataset
from src.models.ssd import SSD
from src.models import loss as loss_module


def infer_num_classes_from_state(state_dict: dict):
    """
    Attempt to infer num_classes (including background) from pred_heads.* weights
    in the checkpoint state_dict. Returns int or None on failure.
    """
    head_indices = set()
    for k in state_dict.keys():
        if k.startswith("pred_heads."):
            parts = k.split(".")
            if len(parts) >= 3:
                try:
                    idx = int(parts[1])
                    head_indices.add(idx)
                except Exception:
                    continue
    head_indices = sorted(head_indices)
    if not head_indices:
        return None

    inferred = []
    for i in head_indices:
        loc_key = f"pred_heads.{i}.loc_conv.weight"
        cls_key = f"pred_heads.{i}.cls_conv.weight"
        if loc_key not in state_dict or cls_key not in state_dict:
            return None
        loc_out = state_dict[loc_key].shape[0]  # anchors*4
        cls_out = state_dict[cls_key].shape[0]  # anchors * num_classes
        if loc_out % 4 != 0:
            return None
        anchors = loc_out // 4
        if anchors <= 0:
            return None
        if cls_out % anchors != 0:
            return None
        classes = cls_out // anchors
        inferred.append((i, anchors, classes))

    classes_set = set(c for (_, _, c) in inferred)
    if len(classes_set) == 1:
        return inferred[0][2]
    return None


def load_checkpoint_and_build_model(ckpt_path: str, device: torch.device, num_classes_arg: int = None):
    """
    Load checkpoint and return (model, state_info). The function will:
     - load checkpoint
     - try to infer num_classes from checkpoint pred_heads
     - instantiate SSD(num_classes=...) accordingly
     - load state_dict (strict if possible; else fallback to non-strict)
    Returns model (moved to device) and the raw loaded checkpoint dict.
    """
    ckpt_obj = torch.load(ckpt_path, map_location="cpu")
    # ckpt may be either pure state_dict or dict with "model_state" key (or other top-level metadata)
    if isinstance(ckpt_obj, dict) and "model_state" in ckpt_obj:
        state_dict = ckpt_obj["model_state"]
    else:
        # If it's a dict with many keys that look like tensors, treat as state_dict
        # Otherwise if it's a metadata dict that wraps state_dict under 'state_dict' or similar, try common keys
        if isinstance(ckpt_obj, dict) and any(k.startswith("pred_heads.") for k in ckpt_obj.keys()):
            state_dict = ckpt_obj
        elif isinstance(ckpt_obj, dict) and "state_dict" in ckpt_obj:
            state_dict = ckpt_obj["state_dict"]
        else:
            # fallback: if it's not obviously a state dict, try to find nested model_state keys
            state_dict = ckpt_obj

    inferred = infer_num_classes_from_state(state_dict if isinstance(state_dict, dict) else {})
    if inferred is not None:
        num_classes = int(inferred)
        print(f"Inferred num_classes (including background) from checkpoint: {num_classes}")
    elif num_classes_arg:
        num_classes = int(num_classes_arg)
        print(f"Using provided --num_classes={num_classes}")
    else:
        # final fallback: assume 2 (background + 1 class)
        num_classes = 2
        print("Could not infer num_classes and --num_classes not provided. Falling back to num_classes=2 (bg+1).")

    model = SSD(num_classes=num_classes)
    # Try strict load first
    try:
        model.load_state_dict(state_dict, strict=True)
        print("Loaded checkpoint into model with strict=True.")
    except Exception as e:
        print("Strict load failed:", e)
        print("Attempting non-strict load (this will skip unmatched params).")
        model.load_state_dict(state_dict, strict=False)

    model.to(device)
    model.eval()
    return model, ckpt_obj


def evaluate(args):
    device = torch.device(args.device)
    print(f"Using device: {device}")

    print("Loading dataset...")
    ds_val = CocoDataset(
        args.annotations,
        images_root=args.images_root,
        target_size=(args.image_size, args.image_size),
        transforms=None,
    )

    val_loader = DataLoader(
        ds_val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=getattr(ds_val, "collate_fn", None),
    )

    print(f"Dataset sizes — val: {len(ds_val)}")

    # Load model + checkpoint
    print("Loading checkpoint and model...")
    model, raw_ckpt = load_checkpoint_and_build_model(args.checkpoint, device, num_classes_arg=args.num_classes)

    # Build loss (try to instantiate SSDLoss with the model's num_classes)
    num_classes_for_loss = None
    try:
        # try to derive num_classes from model attribute if set
        num_classes_for_loss = getattr(model, "num_classes", None)
    except Exception:
        num_classes_for_loss = None

    if not num_classes_for_loss:
        # fallback: use provided CLI or 2
        num_classes_for_loss = args.num_classes or 2

    print(f"Initializing loss with num_classes={num_classes_for_loss}")
    criterion = loss_module.SSDLoss(num_classes=int(num_classes_for_loss))

    # Run evaluation loop — compute average loss across validation set
    model.eval()
    val_losses = {"total_loss": 0.0, "loc_loss": 0.0, "cls_loss": 0.0}
    n_batches = 0

    with torch.no_grad():
        for imgs, targets in tqdm(val_loader, desc="Validation"):
            # imgs: tensor
            imgs = imgs.to(device)
            # Move target tensors to device if dataset returns list/dicts etc.
            if isinstance(targets, (list, tuple)):
                for t in targets:
                    if isinstance(t, dict):
                        for k, v in list(t.items()):
                            if torch.is_tensor(v):
                                t[k] = v.to(device)
            else:
                # If dataset returned a single tensor or other structure, leave it as-is
                pass

            outputs = model(imgs)

            # Attempt to call criterion in several expected signatures
            loss_dict = None
            # prefer (locs, cls, anchors, targets) -> some SSD implementations provide anchors
            if isinstance(outputs, (tuple, list)):
                try:
                    if len(outputs) == 3:
                        loc_preds, cls_preds, anchors = outputs
                        loss_dict = criterion(loc_preds, cls_preds, anchors, targets)
                    elif len(outputs) == 2:
                        loc_preds, cls_preds = outputs
                        loss_dict = criterion(loc_preds, cls_preds, targets)
                    else:
                        loss_dict = criterion(outputs, targets)
                except TypeError:
                    # try alternative ordering via a helper-like approach
                    try:
                        loss_dict = criterion(*outputs, targets)
                    except Exception as e:
                        # as last resort, try calling with outputs only
                        try:
                            loss_dict = criterion(outputs, targets)
                        except Exception as ee:
                            raise RuntimeError(f"Loss call failed with outputs shape/format. Errors: {e} / {ee}")
            else:
                loss_dict = criterion(outputs, targets)

            # Normalize loss_dict to scalar/tensor and aggregated dict form
            if isinstance(loss_dict, dict):
                total_loss = (
                    loss_dict.get("total_loss")
                    or loss_dict.get("loss")
                    or (loss_dict.get("loc_loss", 0) + loss_dict.get("cls_loss", 0))
                )
            else:
                total_loss = loss_dict

            # ensure it's a Python float
            if torch.is_tensor(total_loss):
                total_loss_val = float(total_loss.detach().cpu().item())
            else:
                try:
                    total_loss_val = float(total_loss)
                except Exception:
                    total_loss_val = 0.0

            val_losses["total_loss"] += total_loss_val
            if isinstance(loss_dict, dict):
                val_losses["loc_loss"] += float(loss_dict.get("loc_loss", 0))
                val_losses["cls_loss"] += float(loss_dict.get("cls_loss", 0))

            n_batches += 1

    if n_batches > 0:
        avg_total = val_losses["total_loss"] / n_batches
        avg_loc = val_losses["loc_loss"] / n_batches
        avg_cls = val_losses["cls_loss"] / n_batches
    else:
        avg_total = avg_loc = avg_cls = 0.0

    print(f"Validation summary — batches: {n_batches}")
    print(f"Avg total_loss: {avg_total:.6f}, loc: {avg_loc:.6f}, cls: {avg_cls:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate SSD model on validation set")
    parser.add_argument("--annotations", required=True, help="Path to validation annotations JSON (COCO-like)")
    parser.add_argument("--images_root", required=True, help="Root folder containing images (contains train/val/test folders or images/...)")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint (.pth)")
    parser.add_argument("--image_size", type=int, default=512, help="Resize images to this size (square)")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--num_classes", type=int, default=None, help="(optional) force num_classes (including background). If omitted the script will try to infer from checkpoint.")
    args = parser.parse_args()

    evaluate(args)