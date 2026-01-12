"""
single_stage_loss.py

Robust loss wrapper used by the training script.

- Accepts flexible constructor args (image_size, strides...) to be compatible
  with different train.py variations.
- Normalizes targets into canonical format:
    [ {"boxes": Tensor(M,4), "labels": Tensor(M)}, ... ]
- Returns a dict with total_loss/loc_loss/cls_loss for easy logging.
"""

from typing import List, Dict, Any
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F


class SingleStageLoss(nn.Module):
    def __init__(self, num_classes: int = 2, neg_pos_ratio: int = 3, image_size: int = 512, strides=None, **kwargs):
        super().__init__()
        # store values; keep signature tolerant to extra args
        self.num_classes = int(num_classes)
        self.neg_pos_ratio = int(neg_pos_ratio)
        self.image_size = int(image_size)
        self.strides = strides

    def _normalize_targets(self, targets, device):
        """
        Normalize many possible targets formats into canonical list-of-dicts:
           [ {"boxes": Tensor(M,4), "labels": Tensor(M,)} , ... ]
        Accepts:
           - list of dicts -> returned as-is (converted to device)
           - dict with 'boxes'/'labels' batched tensors (B,N,4) -> split per-image
           - tensor of shape (B,N,4) -> converted to list
           - dict of numeric keys mapping to per-image dicts -> ordered list
           - tuple/list like (boxes_tensor, labels_tensor) batched -> split
        """
        if targets is None:
            return []

        # already list-of-dicts
        if isinstance(targets, list):
            out = []
            for t in targets:
                if not isinstance(t, dict):
                    raise TypeError("Expected each target to be a dict with 'boxes' and 'labels'")
                boxes = t.get("boxes", None)
                labels = t.get("labels", None)
                if boxes is None:
                    boxes = torch.zeros((0, 4), dtype=torch.float32, device=device)
                else:
                    boxes = boxes.to(device)
                if labels is None:
                    labels = torch.zeros((boxes.shape[0],), dtype=torch.long, device=device)
                else:
                    labels = labels.to(device)
                out.append({"boxes": boxes, "labels": labels})
            return out

        # dict with batched tensors
        if isinstance(targets, dict):
            # if boxes is batched (B,N,4)
            if "boxes" in targets and torch.is_tensor(targets["boxes"]) and targets["boxes"].dim() == 3:
                boxes = targets["boxes"]
                labels = targets.get("labels", None)
                B = boxes.shape[0]
                if labels is None:
                    labels = torch.zeros((B, boxes.shape[1]), dtype=torch.long, device=device)
                # ensure labels is tensor and shape (B,N)
                if torch.is_tensor(labels) and labels.dim() == 1:
                    labels = labels.unsqueeze(1).expand(-1, boxes.shape[1])
                out = []
                for b in range(B):
                    bboxes = boxes[b].to(device)
                    blabels = labels[b].to(device) if torch.is_tensor(labels) else torch.zeros((bboxes.shape[0],), dtype=torch.long, device=device)
                    out.append({"boxes": bboxes, "labels": blabels})
                warnings.warn("Normalized targets: dict with batched 'boxes' -> list of per-image dicts.")
                return out

            # dict with numeric keys (e.g., {"0": {...}, "1": {...}})
            keys = list(targets.keys())
            try:
                if all(str(k).lstrip().lstrip("'\"").isdigit() for k in keys):
                    items = sorted(targets.items(), key=lambda kv: int(str(kv[0]).lstrip().lstrip("'\"")))
                    out = []
                    for _, v in items:
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

            # fallback: use values
            try:
                vals = list(targets.values())
                return self._normalize_targets(vals, device)
            except Exception:
                raise TypeError("Cannot normalize targets from dict-like input")

        # tensor inputs
        if torch.is_tensor(targets):
            if targets.dim() == 3:
                B = targets.shape[0]
                out = []
                for b in range(B):
                    bboxes = targets[b].to(device)
                    blabels = torch.zeros((bboxes.shape[0],), dtype=torch.long, device=device)
                    out.append({"boxes": bboxes, "labels": blabels})
                warnings.warn("Normalized targets: tensor (B,N,4) -> list with dummy labels.")
                return out
            else:
                raise TypeError(f"Unsupported tensor shape for targets: {targets.shape}")

        # tuple/list but not list-of-dicts -> try interpret as (boxes, labels) batched
        if isinstance(targets, (tuple, list)):
            if len(targets) == 2 and torch.is_tensor(targets[0]):
                boxes, labels = targets
                if boxes.dim() == 3:
                    B = boxes.shape[0]
                    out = []
                    for b in range(B):
                        bboxes = boxes[b].to(device)
                        blabels = labels[b].to(device) if torch.is_tensor(labels) else torch.zeros((bboxes.shape[0],), dtype=torch.long, device=device)
                        out.append({"boxes": bboxes, "labels": blabels})
                    warnings.warn("Normalized targets: tuple (boxes, labels) batched -> list")
                    return out

        raise TypeError(f"Unsupported targets format: {type(targets)}")

    def forward(self, preds, targets=None):
        """
        Expected canonical preds:
          - either (loc_preds, conf_preds, anchors) or model-specific outputs that you decode in loss.
        This example computes a *placeholder* location L1 loss (against zeros) and a
        cross-entropy classification loss where all anchors are treated as background.
        Replace with real matching + hard-negative mining logic for production.
        Returns dict {"total_loss","loc_loss","cls_loss"}.
        """
        device = preds[0].device if isinstance(preds, (tuple, list)) else (preds.device if torch.is_tensor(preds) else torch.device("cpu"))
        targets_list = self._normalize_targets(targets, device)

        # Example simple losses (placeholders)
        if isinstance(preds, (tuple, list)) and len(preds) >= 2:
            loc_preds = preds[0]  # (B,N,4)
            conf_preds = preds[1]  # (B,N,C)
        elif torch.is_tensor(preds):
            # some models may output single tensor; we cannot handle that generically here
            raise RuntimeError("Unexpected preds format for loss wrapper: please pass (loc_preds, conf_preds, anchors) or update loss implementation.")
        else:
            loc_preds = None
            conf_preds = None

        loc_loss = torch.tensor(0.0, device=device)
        cls_loss = torch.tensor(0.0, device=device)

        if loc_preds is not None:
            loc_loss = F.l1_loss(loc_preds, torch.zeros_like(loc_preds), reduction="mean")

        if conf_preds is not None:
            # treat all anchors as background (class 0) as placeholder
            target_cls = torch.zeros((conf_preds.shape[0], conf_preds.shape[1]), dtype=torch.long, device=device)
            cls_preds_flat = conf_preds.view(-1, conf_preds.shape[-1])
            target_flat = target_cls.view(-1)
            cls_loss = F.cross_entropy(cls_preds_flat, target_flat, reduction="mean")

        total_loss = loc_loss + cls_loss

        # return dict so training script can read keys safely
        return {"total_loss": total_loss, "loc_loss": float(loc_loss.detach().cpu().item()), "cls_loss": float(cls_loss.detach().cpu().item())}