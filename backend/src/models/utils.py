# src/models/utils.py
import torch
import torch.nn.functional as F
from torchvision.ops import nms
from typing import Tuple, List, Optional

# decode_predictions(...)
#
# Expected input:
#   outputs: either (loc_preds, cls_preds, anchors) OR a single tensor depending on your model.
#     - loc_preds: Tensor (B, N, 4)  (dx, dy, dw, dh) usually
#     - cls_preds: Tensor (B, N, C)  (raw logits)
#     - anchors:   Tensor (N, 4)     (cx, cy, w, h) -- either normalized (0..1) or absolute
#
# Returns: (boxes, labels, scores)
#   boxes:  Tensor (M, 4) in pixel coords as [x1, y1, x2, y2]
#   labels: Tensor (M,) class indices (int)
#   scores: Tensor (M,) confidence scores (float)
#
def decode_predictions(
    outputs,
    score_threshold: float = 0.3,
    topk: int = 200,
    iou_threshold: float = 0.45,
    max_detections: int = 100,
    image_size: Optional[int] = None,
    variances=(0.1, 0.2),
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generic SSD-style decoder. Adjust variances if your model used different scaling.
    If outputs is not a (locs, cls, anchors) tuple, modify infer to pass those three items.
    """
    if not isinstance(outputs, (list, tuple)) or len(outputs) < 3:
        raise ValueError("decode_predictions expects (loc_preds, cls_preds, anchors) as outputs")

    loc_preds, cls_preds, anchors = outputs  # loc: (B,N,4), cls: (B,N,C), anchors: (N,4)
    device = loc_preds.device

    # Support batch size > 1, but infer/predict typically handles single image
    B = loc_preds.shape[0]
    if B != 1:
        # we'll decode only the first batch item for simplicity; adapt if you want batch decoding
        loc = loc_preds[0]
        cls = cls_preds[0]
    else:
        loc = loc_preds[0]  # (N,4)
        cls = cls_preds[0]  # (N,C)

    # anchors: maybe (N,4) or (1,N,4) — normalize shape
    if anchors.dim() == 3 and anchors.shape[0] == 1:
        anchors = anchors[0]
    anchors = anchors.to(device).float()  # (N,4)

    # Basic checks
    if loc.shape[0] != anchors.shape[0]:
        # try to reshape loc if it's (4, N) or (N*4,) etc.
        raise RuntimeError(f"Mismatch between loc_preds ({loc.shape}) and anchors ({anchors.shape})")

    # convert cls logits -> probs
    probs = F.softmax(cls, dim=-1)  # (N, C)
    num_classes = probs.shape[-1]

    # decide whether anchors are normalized (0..1) or absolute pixel coords
    anchors_max = anchors.abs().max().item()
    anchors_are_normalized = anchors_max <= 1.0001

    # decode loc to boxes (cx,cy,w,h)
    # SSD decoding: cx = anchor_cx + loc_dx * anchor_w * var[0]
    #               cy = anchor_cy + loc_dy * anchor_h * var[0]
    #               w = anchor_w * exp(loc_dw * var[1])
    #               h = anchor_h * exp(loc_dh * var[1])
    acx = anchors[:, 0]
    acy = anchors[:, 1]
    aw = anchors[:, 2]
    ah = anchors[:, 3]

    dx = loc[:, 0]
    dy = loc[:, 1]
    dw = loc[:, 2]
    dh = loc[:, 3]

    var_cx_cy, var_wh = variances

    cx = acx + dx * aw * var_cx_cy
    cy = acy + dy * ah * var_cx_cy
    w = aw * torch.exp(dw * var_wh)
    h = ah * torch.exp(dh * var_wh)

    # convert to corner format
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    boxes = torch.stack([x1, y1, x2, y2], dim=1)  # (N,4)

    # If anchors were normalized and user provided image_size, scale to pixels
    if anchors_are_normalized and image_size is not None:
        boxes = boxes * float(image_size)
    elif anchors_are_normalized and image_size is None:
        # still return normalized boxes 0..1
        pass
    else:
        # anchors appear absolute (pixel coords) => boxes already in pixels
        pass

    boxes = boxes.clamp(min=0.0)

    final_boxes = []
    final_labels = []
    final_scores = []

    # For each class (skip background index 0)
    for cls_idx in range(1, num_classes):
        scores = probs[:, cls_idx]  # (N,)
        keep = scores > score_threshold
        if not keep.any():
            continue
        cls_scores = scores[keep]
        cls_boxes = boxes[keep]
        # apply topk pre-filter
        if cls_scores.numel() > topk:
            topk_scores, topk_idx = torch.topk(cls_scores, topk)
            cls_scores = topk_scores
            cls_boxes = cls_boxes[topk_idx]

        # NMS
        keep_idx = nms(cls_boxes, cls_scores, iou_threshold)
        if keep_idx.numel() == 0:
            continue

        kept_boxes = cls_boxes[keep_idx]
        kept_scores = cls_scores[keep_idx]
        kept_labels = torch.full((kept_scores.shape[0],), cls_idx, dtype=torch.int64, device=device)

        final_boxes.append(kept_boxes)
        final_labels.append(kept_labels)
        final_scores.append(kept_scores)

    if len(final_boxes) == 0:
        return torch.zeros((0, 4), device=device), torch.zeros((0,), dtype=torch.int64, device=device), torch.zeros((0,), device=device)

    final_boxes = torch.cat(final_boxes, dim=0)
    final_labels = torch.cat(final_labels, dim=0)
    final_scores = torch.cat(final_scores, dim=0)

    # overall topk / clamp to max_detections
    if final_scores.numel() > max_detections:
        topk_scores, topk_idx = torch.topk(final_scores, max_detections)
        final_scores = topk_scores
        final_boxes = final_boxes[topk_idx]
        final_labels = final_labels[topk_idx]

    # move to cpu and return
    return final_boxes.cpu(), final_labels.cpu(), final_scores.cpu()