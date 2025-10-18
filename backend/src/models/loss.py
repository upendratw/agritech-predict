# src/models/loss.py
"""
Robust SSD loss wrapper.

This file normalizes the incoming `targets` structure into the canonical form:
  targets_list = [
      {"boxes": Tensor(num_objs, 4), "labels": Tensor(num_objs)},
      ...
  ]

Supported incoming shapes:
 - list of dicts (canonical) -> left intact
 - dict with numeric keys ("0","1",...) or int keys -> converted to list ordered by key
 - dict with batched tensors: {"boxes": Tensor(B, N, 4), "labels": Tensor(B, N)} -> converted per image
 - tensor with shape (B, N, 4) (boxes only) -> converted per image with dummy labels=0
 - tuple/list containing batched boxes/labels -> attempts to interpret sensibly

After normalization the real loss computation runs as before.
"""
from typing import List, Dict, Any
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

# NOTE: this loss is intentionally simple / illustrative.
# Replace internals with your original matching + localization/classification logic if needed.

class SSDLoss(nn.Module):
    def __init__(self, num_classes: int = 2, neg_pos_ratio: int = 3):
        """
        num_classes: total number of classes including background
        """
        super().__init__()
        self.num_classes = int(num_classes)
        self.neg_pos_ratio = int(neg_pos_ratio)

    def _normalize_targets(self, targets, device):
        """
        Convert many common `targets` formats into a list of dicts:
          [ {"boxes": Tensor(M,4), "labels": Tensor(M)}, ... ]
        """
        # already canonical
        if isinstance(targets, list):
            # ensure tensors are on device
            out = []
            for t in targets:
                if not isinstance(t, dict):
                    raise TypeError("Expected each target to be a dict with 'boxes' and 'labels'")
                boxes = t.get("boxes")
                labels = t.get("labels")
                if boxes is None:
                    # empty boxes
                    boxes = torch.zeros((0, 4), dtype=torch.float32, device=device)
                else:
                    boxes = boxes.to(device)
                if labels is None:
                    labels = torch.zeros((boxes.shape[0],), dtype=torch.long, device=device)
                else:
                    labels = labels.to(device)
                out.append({"boxes": boxes, "labels": labels})
            return out

        # dict-like mapping (maybe numeric keys)
        if isinstance(targets, dict):
            # If there are keys 'boxes' and/or 'labels' and they are batched tensors -> split per image
            if ("boxes" in targets and torch.is_tensor(targets["boxes"])) or ("labels" in targets and torch.is_tensor(targets.get("labels", None))):
                boxes = targets.get("boxes", None)
                labels = targets.get("labels", None)
                if boxes is not None and boxes.dim() == 3:
                    B = boxes.shape[0]
                    # labels may be (B,N) or (B,) etc. normalize to (B,N)
                    if labels is None:
                        labels = torch.zeros((B, boxes.shape[1]), dtype=torch.long, device=device)
                    else:
                        labels = labels.to(device)
                        if labels.dim() == 1:
                            # assume (B,) -> expand to (B,N) with zeros
                            labels = labels.unsqueeze(1).expand(-1, boxes.shape[1])
                    out = []
                    for b in range(B):
                        bboxes = boxes[b].to(device)
                        # remove zero-area rows if any (optional)
                        # create per image labels (trim to same length)
                        blabels = labels[b]
                        # If blabels length doesn't match number of boxes, try to trim/pad
                        if blabels.numel() != bboxes.shape[0]:
                            if blabels.numel() < bboxes.shape[0]:
                                pad = torch.zeros((bboxes.shape[0] - blabels.numel(),), dtype=torch.long, device=device)
                                blabels = torch.cat([blabels, pad], dim=0)
                            else:
                                blabels = blabels[: bboxes.shape[0]]
                        out.append({"boxes": bboxes, "labels": blabels})
                    warnings.warn("Normalized targets: dict with batched tensors -> list of per-image dicts.")
                    return out

            # otherwise dict of numeric keys mapping to per-image dicts?
            try:
                keys = list(targets.keys())
                # detect numeric-like keys
                if all(str(k).lstrip().lstrip("'\"").isdigit() for k in keys):
                    # sort by numeric value
                    items = sorted(targets.items(), key=lambda kv: int(str(kv[0]).lstrip().lstrip("'\"")))
                    out = []
                    for _, v in items:
                        if not isinstance(v, dict):
                            raise TypeError("Expected mapping values to be dicts")
                        boxes = v.get("boxes", None)
                        labels = v.get("labels", None)
                        if boxes is None:
                            boxes = torch.zeros((0, 4), dtype=torch.float32, device=device)
                        else:
                            boxes = boxes.to(device)
                        if labels is None:
                            labels = torch.zeros((boxes.shape[0],), dtype=torch.long, device=device)
                        else:
                            labels = labels.to(device)
                        out.append({"boxes": boxes, "labels": labels})
                    warnings.warn("Normalized targets: dict with numeric keys -> list ordered by key.")
                    return out
            except Exception:
                pass

            # fallback: use values()
            try:
                vals = list(targets.values())
                # map them through canonical list handling
                return self._normalize_targets(vals, device)
            except Exception:
                raise TypeError("Cannot normalize targets from dict-like input")

        # if targets is a Tensor of shape (B, N, 4) assume boxes-only
        if torch.is_tensor(targets):
            if targets.dim() == 3:
                B = targets.shape[0]
                out = []
                for b in range(B):
                    bboxes = targets[b].to(device)
                    labels = torch.zeros((bboxes.shape[0],), dtype=torch.long, device=device)
                    out.append({"boxes": bboxes, "labels": labels})
                warnings.warn("Normalized targets: tensor (B,N,4) -> list of per-image dicts with dummy labels.")
                return out
            else:
                raise TypeError(f"Unexpected tensor shape for targets: {targets.shape}")

        # tuple/list but not list-of-dicts -> try to interpret as (boxes, labels) batched
        if isinstance(targets, (tuple, list)):
            # already captured list-of-dicts above; this branch implies e.g. (boxes_tensor, labels_tensor)
            if len(targets) == 2 and torch.is_tensor(targets[0]):
                boxes, labels = targets
                if boxes.dim() == 3:
                    B = boxes.shape[0]
                    out = []
                    for b in range(B):
                        bboxes = boxes[b].to(device)
                        blabels = labels[b].to(device) if torch.is_tensor(labels) else torch.zeros((bboxes.shape[0],), dtype=torch.long, device=device)
                        out.append({"boxes": bboxes, "labels": blabels})
                    warnings.warn("Normalized targets: tuple (boxes,labels) batched -> list")
                    return out

        raise TypeError(f"Unsupported targets format: {type(targets)}")

    def forward(self, loc_preds, cls_preds=None, anchors=None, targets=None):
        """
        Expected canonical inputs:
          loc_preds: Tensor (B, num_locs, 4)
          cls_preds: Tensor (B, num_locs, num_classes) or None
          anchors: Tensor or None
          targets: list of per-image dicts {"boxes": Tensor(M,4), "labels": Tensor(M)}

        This forward returns either:
          - dict with loss components: {"total_loss": ..., "loc_loss": ..., "cls_loss": ...}
          - or a scalar tensor
        """
        device = loc_preds.device if hasattr(loc_preds, "device") else torch.device("cpu")
        # Normalize targets
        if targets is None:
            targets_list = []
        else:
            targets_list = self._normalize_targets(targets, device)

        # ---------- Example/simple loss computation ----------
        # This block is intentionally simple: it calculates IoU-insensitive L1 loc loss
        # and a cross-entropy classification loss using dummy matching.
        # Replace with your matching + hard negative mining logic for a real SSD.
        B = loc_preds.shape[0] if loc_preds is not None else (cls_preds.shape[0] if cls_preds is not None else 1)

        loc_loss = torch.tensor(0.0, device=device)
        cls_loss = torch.tensor(0.0, device=device)

        # For demo: treat every anchor as negative, and penalize classification toward background (0)
        # and L1 to zero for localization (this is placeholder logic).
        # A proper SSD loss would match anchors <-> gt boxes and compute loc/cls loss accordingly.
        if loc_preds is not None:
            loc_loss = F.l1_loss(loc_preds, torch.zeros_like(loc_preds), reduction="mean")

        if cls_preds is not None:
            # if cls_preds is (B, num_loc, num_classes)
            # create fake target class 0 for all anchors
            target_cls = torch.zeros((cls_preds.shape[0], cls_preds.shape[1]), dtype=torch.long, device=device)
            # flatten predictions for cross_entropy
            cls_preds_flat = cls_preds.view(-1, cls_preds.shape[-1])
            target_flat = target_cls.view(-1)
            cls_loss = F.cross_entropy(cls_preds_flat, target_flat, reduction="mean")

        total_loss = loc_loss + cls_loss

        return {"total_loss": total_loss, "loc_loss": loc_loss.detach().cpu().item() if torch.is_tensor(loc_loss) else float(loc_loss), "cls_loss": cls_loss.detach().cpu().item() if torch.is_tensor(cls_loss) else float(cls_loss)}