# src/models/single_stage_utils.py
import torch
from torchvision.ops import nms

def nms_postprocess(boxes: torch.Tensor, scores: torch.Tensor, labels: torch.Tensor, iou_thresh: float = 0.45, max_dets: int = 100):
    """
    Per-class NMS + top-K selection.
    boxes: (M,4) cpu tensor
    scores: (M,) cpu tensor
    labels: (M,) cpu tensor
    """
    if boxes.numel() == 0:
        return boxes, scores, labels
    keep_all = []
    unique_labels = labels.unique()
    for lab in unique_labels:
        idx = (labels == lab).nonzero(as_tuple=False).squeeze(1)
        b = boxes[idx]
        s = scores[idx]
        if b.numel() == 0:
            continue
        keep = nms(b, s, iou_thresh)
        if keep.numel() > 0:
            keep_all.append(idx[keep])
    if not keep_all:
        return torch.zeros((0, 4)), torch.zeros((0,)), torch.zeros((0,), dtype=torch.long)
    keep_idx = torch.cat(keep_all, dim=0)
    selected_scores = scores[keep_idx]
    topk = min(selected_scores.shape[0], max_dets)
    topk_scores, topk_idx = torch.topk(selected_scores, topk)
    final_idx = keep_idx[topk_idx]
    return boxes[final_idx], scores[final_idx], labels[final_idx]