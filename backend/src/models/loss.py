# src/models/loss.py
"""
SSD loss with anchor<->gt matching, encoding/decoding, and hard-negative mining.
Intended to replace the simpler placeholder so training learns meaningful matches.

Assumptions:
 - anchors are passed in as (N,4) in xyxy coordinates in the SAME "reference" pixel
   space that loc_preds correspond to. In the repo the SSD.generate_anchors()
   produces anchors in pixel coords relative to model.image_size; this loss will
   align gt boxes to that same reference using a scale step if necessary.
 - cls_preds are raw logits (B, N, num_classes) where class 0 is BACKGROUND.
 - targets may be given in many formats (list-of-dicts, batched dict, etc.)
   — _normalize_targets handles many forms (kept from your previous file).
"""

from typing import List, Dict, Any, Tuple
import warnings
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _IoU(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    IoU between two sets of boxes (xyxy)
      boxes1: (M,4), boxes2: (N,4)
    returns (M, N)
    """
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return torch.zeros((boxes1.shape[0], boxes2.shape[0]), device=boxes1.device)

    x11, y11, x12, y12 = boxes1[:, 0].unsqueeze(1), boxes1[:, 1].unsqueeze(1), boxes1[:, 2].unsqueeze(1), boxes1[:, 3].unsqueeze(1)
    x21, y21, x22, y22 = boxes2[:, 0].unsqueeze(0), boxes2[:, 1].unsqueeze(0), boxes2[:, 2].unsqueeze(0), boxes2[:, 3].unsqueeze(0)

    xx1 = torch.max(x11, x21)
    yy1 = torch.max(y11, y21)
    xx2 = torch.min(x12, x22)
    yy2 = torch.min(y12, y22)

    inter_w = (xx2 - xx1).clamp(min=0)
    inter_h = (yy2 - yy1).clamp(min=0)
    inter = inter_w * inter_h

    area1 = (x12 - x11).clamp(min=0) * (y12 - y11).clamp(min=0)
    area2 = (x22 - x21).clamp(min=0) * (y22 - y21).clamp(min=0)

    union = area1 + area2 - inter
    iou = torch.zeros_like(inter)
    nonzero = union > 0
    iou[nonzero] = inter[nonzero] / union[nonzero]
    return iou


class SSDLoss(nn.Module):
    def __init__(self, num_classes: int = 2, neg_pos_ratio: int = 3, match_iou_thresh: float = 0.5, variances=(0.1, 0.2)):
        """
        num_classes: total classes including background (so background index = 0)
        """
        super().__init__()
        self.num_classes = int(num_classes)
        assert self.num_classes >= 2, "Expect at least background + 1 class"
        self.neg_pos_ratio = int(neg_pos_ratio)
        self.match_iou_thresh = float(match_iou_thresh)
        self.variances = tuple(variances)

    # ---------------------------
    # target normalization (kept + extended)
    # ---------------------------
    def _normalize_targets(self, targets, device):
        # copied/adapted from your robust normalizer, simplified slightly
        if targets is None:
            return []

        if isinstance(targets, list):
            out = []
            for t in targets:
                if not isinstance(t, dict):
                    raise TypeError("Expected each target to be dict{'boxes','labels'}")
                boxes = t.get("boxes")
                labels = t.get("labels")
                if boxes is None:
                    boxes = torch.zeros((0, 4), dtype=torch.float32, device=device)
                else:
                    boxes = boxes.to(device).float()
                if labels is None:
                    labels = torch.zeros((boxes.shape[0],), dtype=torch.long, device=device)
                else:
                    labels = labels.to(device).long()
                out.append({"boxes": boxes, "labels": labels})
            return out

        if isinstance(targets, dict):
            # batched tensors? -> split per image
            if "boxes" in targets and torch.is_tensor(targets["boxes"]):
                boxes = targets["boxes"]
                labels = targets.get("labels", None)
                if boxes.dim() == 3:
                    B = boxes.shape[0]
                    out = []
                    if labels is None:
                        labels = torch.zeros((B, boxes.shape[1]), dtype=torch.long, device=device)
                    labels = labels.to(device).long()
                    for b in range(B):
                        bboxes = boxes[b].to(device).float()
                        blabels = labels[b]
                        out.append({"boxes": bboxes, "labels": blabels})
                    return out
            # otherwise try numeric keys
            keys = list(targets.keys())
            if all(str(k).lstrip().lstrip("'\"").isdigit() for k in keys):
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
                        boxes = boxes.to(device).float()
                    if labels is None:
                        labels = torch.zeros((boxes.shape[0],), dtype=torch.long, device=device)
                    else:
                        labels = labels.to(device).long()
                    out.append({"boxes": boxes, "labels": labels})
                return out

            # final fallback: values -> try to normalize them as list
            try:
                vals = list(targets.values())
                return self._normalize_targets(vals, device)
            except Exception:
                raise TypeError("Cannot normalize dict-like targets")

        if torch.is_tensor(targets):
            if targets.dim() == 3:
                B = targets.shape[0]
                out = []
                for b in range(B):
                    bboxes = targets[b].to(device).float()
                    out.append({"boxes": bboxes, "labels": torch.zeros((bboxes.shape[0],), dtype=torch.long, device=device)})
                return out
            else:
                raise TypeError("Unsupported tensor targets shape")

        if isinstance(targets, (tuple, list)):
            # if (boxes_tensor, labels_tensor)
            if len(targets) == 2 and torch.is_tensor(targets[0]):
                boxes, labels = targets
                if boxes.dim() == 3:
                    B = boxes.shape[0]
                    out = []
                    for b in range(B):
                        bboxes = boxes[b].to(device).float()
                        blabels = labels[b].to(device).long() if torch.is_tensor(labels) else torch.zeros((bboxes.shape[0],), dtype=torch.long, device=device)
                        out.append({"boxes": bboxes, "labels": blabels})
                    return out

        raise TypeError(f"Unsupported targets format: {type(targets)}")

    # ---------------------------
    # box encoding / decoding helpers
    # ---------------------------
    def _xyxy_to_cxcywh(self, boxes: torch.Tensor) -> torch.Tensor:
        """
        boxes: (N,4) xyxy -> (N,4) cx,cy,w,h
        """
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        w = (x2 - x1).clamp(min=1e-6)
        h = (y2 - y1).clamp(min=1e-6)
        return torch.stack([cx, cy, w, h], dim=1)

    def encode_boxes(self, gt_boxes: torch.Tensor, anchors_xyxy: torch.Tensor) -> torch.Tensor:
        """
        Encode gt_boxes -> location targets in the same order as anchors.
        gt_boxes: (K,4) xyxy
        anchors_xyxy: (N,4) xyxy
        Returns: loc_targets (N,4) where unmatched anchors get zeros (but we'll mask)
        Encoding (SSD-style):
           tx = (gcx - acx) / (aw * var_cxcy)
           ty = (gcy - acy) / (ah * var_cxcy)
           tw = log(gw / aw) / var_wh
           th = log(gh / ah) / var_wh
        """
        var_cxcy, var_wh = self.variances

        a_cxcywh = self._xyxy_to_cxcywh(anchors_xyxy)  # (N,4)
        # We will compute per-anchor matched gt in matching routine, so this helper expects
        # matched_gt per anchor to be provided externally. For convenience we'll still implement
        # general formula if arrays match shape N.
        raise RuntimeError("encode_boxes() is not intended to be called standalone without matched gt per anchor.")

    # ---------------------------
    # main forward
    # ---------------------------
    def forward(self, loc_preds: torch.Tensor, cls_preds: torch.Tensor = None, anchors: torch.Tensor = None, targets=None) -> Dict[str, Any]:
        """
        Expects:
          loc_preds: (B, N, 4) predicted offsets (dx,dy,dw,dh)
          cls_preds: (B, N, C) logits (C includes background at index 0)
          anchors:   (N,4) xyxy anchor boxes in the reference pixel coords
          targets:   list-of-dicts or other forms (normalized by helper)
        Returns dict: { total_loss, loc_loss, cls_loss }
        """

        device = loc_preds.device if hasattr(loc_preds, "device") else torch.device("cpu")

        # canonical targets list
        targets_list = self._normalize_targets(targets, device)

        B = loc_preds.shape[0]
        N = loc_preds.shape[1]
        # anchors should be (N,4) — move to device and float
        anchors = anchors.to(device).float()

        # If no targets (empty dataset) -> return zero loss
        if len(targets_list) == 0:
            # fallback: small regularizer losses
            loc_loss = F.l1_loss(loc_preds, torch.zeros_like(loc_preds), reduction="mean")
            if cls_preds is not None:
                target_cls = torch.zeros((cls_preds.shape[0], cls_preds.shape[1]), dtype=torch.long, device=device)
                cls_loss = F.cross_entropy(cls_preds.view(-1, cls_preds.shape[-1]), target_cls.view(-1), reduction="mean")
            else:
                cls_loss = torch.tensor(0.0, device=device)
            total_loss = loc_loss + cls_loss
            return {"total_loss": total_loss, "loc_loss": float(loc_loss.detach().cpu().item()), "cls_loss": float(cls_loss.detach().cpu().item())}

        # We'll build per-batch anchor targets: loc_t (B,N,4) and cls_t (B,N)
        loc_targets = torch.zeros_like(loc_preds, device=device)
        cls_targets = torch.zeros((B, N), dtype=torch.long, device=device)  # background default 0

        pos_mask = torch.zeros((B, N), dtype=torch.bool, device=device)

        # Determine scale relationship between gt boxes and anchors:
        # if GT coords much larger than anchors range, we'll scale GT to anchors' range.
        anchors_max = anchors.abs().max().item()

        # For each image, perform matching:
        for b in range(B):
            gt = targets_list[b] if b < len(targets_list) else {"boxes": torch.zeros((0, 4), device=device), "labels": torch.zeros((0,), dtype=torch.long, device=device)}
            gt_boxes = gt.get("boxes", torch.zeros((0, 4), device=device)).to(device).float()
            gt_labels = gt.get("labels", torch.zeros((gt_boxes.shape[0],), dtype=torch.long, device=device)).to(device).long()

            if gt_boxes.numel() == 0:
                # no objects -> all anchors background
                continue

            # if GT coordinates appear in original image scale and anchors in smaller reference,
            # rescale GT to anchor coordinate reference (anchors_max is roughly reference image_size)
            gt_max = float(gt_boxes.abs().max().item()) if gt_boxes.numel() > 0 else 0.0
            if gt_max > 0 and gt_max > anchors_max * 1.1:
                scale = anchors_max / gt_max
                gt_boxes = gt_boxes * scale

            # compute IoU between anchors (N,4) and gt_boxes (K,4) -> (N,K) (anchor x gt)
            ious = _IoU(anchors, gt_boxes)  # N x K
            if ious.numel() == 0:
                continue

            # For each anchor find best gt and IoU
            best_iou_per_anchor, best_gt_idx = torch.max(ious, dim=1)  # (N,)
            # For each gt, ensure it gets at least one anchor (best anchor for that gt)
            best_anchor_for_gt_iou, best_anchor_for_gt = torch.max(ious, dim=0)  # per-gt best anchor index

            # Force-match: anchor best for each gt -> positive
            for gt_idx, anchor_idx in enumerate(best_anchor_for_gt.tolist()):
                pos_mask[b, int(anchor_idx)] = True
                cls_targets[b, int(anchor_idx)] = (int(gt_labels[gt_idx]) + 1) if gt_labels.numel() > 0 else 1  # shift label +1 so 0 is background
                # compute loc target for this anchor -> do below in vectorized way

            # Positives by IoU threshold
            pos_by_thresh = best_iou_per_anchor >= self.match_iou_thresh
            pos_mask[b] = pos_mask[b] | pos_by_thresh

            # assign cls_targets for anchors selected by threshold
            pos_idx = torch.nonzero(pos_mask[b], as_tuple=False).squeeze(1) if pos_mask[b].any() else torch.tensor([], dtype=torch.long, device=device)
            if pos_idx.numel() > 0:
                # for those anchors pick the corresponding gt label via best_gt_idx
                sel_gt_idx = best_gt_idx[pos_idx]
                # map to labels (add 1 so background=0)
                mapped_labels = (gt_labels[sel_gt_idx] + 1).to(torch.long)
                cls_targets[b, pos_idx] = mapped_labels

            # Compute loc targets for matched anchors only
            if pos_idx.numel() > 0:
                # anchors_cxcywh
                a_cxcywh = self._xyxy_to_cxcywh(anchors[pos_idx])
                g_boxes_for_anchors = gt_boxes[best_gt_idx[pos_idx]]
                g_cxcywh = self._xyxy_to_cxcywh(g_boxes_for_anchors)

                var_cxcy, var_wh = self.variances
                tx = (g_cxcywh[:, 0] - a_cxcywh[:, 0]) / (a_cxcywh[:, 2] * var_cxcy)
                ty = (g_cxcywh[:, 1] - a_cxcywh[:, 1]) / (a_cxcywh[:, 3] * var_cxcy)
                tw = torch.log(g_cxcywh[:, 2] / a_cxcywh[:, 2]) / var_wh
                th = torch.log(g_cxcywh[:, 3] / a_cxcywh[:, 3]) / var_wh

                loc_targets[b, pos_idx, 0] = tx
                loc_targets[b, pos_idx, 1] = ty
                loc_targets[b, pos_idx, 2] = tw
                loc_targets[b, pos_idx, 3] = th

        # At this point, pos_mask indicates positives per image and loc_targets/cls_targets are filled.

        # Compute localization loss: Smooth L1 over positives only
        pos_mask_float = pos_mask.float()
        num_pos = pos_mask.sum().item()
        if num_pos == 0:
            # if no positives, still compute small regularizer to keep training stable
            loc_loss = F.smooth_l1_loss(loc_preds, torch.zeros_like(loc_preds), reduction="mean")
        else:
            # Use only positive anchors in loc loss
            pos_mask_exp = pos_mask.unsqueeze(-1).expand_as(loc_preds)  # (B,N,4)
            loc_pred_pos = loc_preds[pos_mask_exp].view(-1, 4)
            loc_t_pos = loc_targets[pos_mask_exp].view(-1, 4)
            loc_loss = F.smooth_l1_loss(loc_pred_pos, loc_t_pos, reduction="sum")  # sum over positives
            loc_loss = loc_loss / max(1.0, float(num_pos))  # normalize by num positives

        # Classification loss with hard negative mining
        if cls_preds is None:
            cls_loss = torch.tensor(0.0, device=device)
        else:
            B, N, C = cls_preds.shape
            # compute per-anchor classification loss (cross_entropy) but we need per-anchor scores
            cls_preds_flat = cls_preds.view(-1, C)
            cls_targets_flat = cls_targets.view(-1)

            # compute per-anchor CE loss (no reduction)
            ce = F.cross_entropy(cls_preds_flat, cls_targets_flat, reduction="none")  # (B*N,)
            ce = ce.view(B, N)

            cls_loss = 0.0
            for b in range(B):
                pos_mask_b = pos_mask[b]  # bool
                num_pos_b = pos_mask_b.sum().item()
                num_neg_b = int(min(self.neg_pos_ratio * max(1, num_pos_b), N - int(num_pos_b)))

                # positive loss contributions:
                pos_loss = ce[b][pos_mask_b].sum() if num_pos_b > 0 else torch.tensor(0.0, device=device)

                # negative mining: choose highest CE among negatives
                if num_neg_b > 0:
                    neg_mask_b = ~pos_mask_b
                    neg_losses = ce[b][neg_mask_b]
                    if neg_losses.numel() > 0:
                        topk = min(num_neg_b, neg_losses.numel())
                        topk_vals, _ = torch.topk(neg_losses, topk)
                        neg_loss = topk_vals.sum()
                    else:
                        neg_loss = torch.tensor(0.0, device=device)
                else:
                    neg_loss = torch.tensor(0.0, device=device)

                cls_loss += (pos_loss + neg_loss) / max(1.0, float(max(1, num_pos_b) + num_neg_b))

            cls_loss = cls_loss / float(B)

        total_loss = loc_loss + cls_loss

        # return numeric summaries similar to your previous return style
        return {
            "total_loss": total_loss,
            "loc_loss": float(loc_loss.detach().cpu().item()) if torch.is_tensor(loc_loss) else float(loc_loss),
            "cls_loss": float(cls_loss.detach().cpu().item()) if torch.is_tensor(cls_loss) else float(cls_loss),
        }