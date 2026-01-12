# src/models/single_stage_model.py
import math
from typing import List, Tuple

import torch
import torch.nn as nn

# Try to reuse backbone & extras from your ssd module
try:
    from src.models.ssd import SmallBackbone, ExtraLayers
except Exception:
    # if import path differs, adjust accordingly in your project
    from src.models.ssd import SmallBackbone, ExtraLayers


class SingleStageHead(nn.Module):
    """
    Head that predicts per-cell:
      - 4 bbox deltas (tx,ty,tw,th)
      - 1 objectness logit
      - C class logits
    Outputs shape per feature map: (B, H*W, 5 + num_classes)
    """

    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        out_ch = 5 + num_classes
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_ch, kernel_size=1, padding=0),
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,C,H,W) -> out: (B, out_ch, H, W) -> (B, H*W, out_ch)
        out = self.conv(x)
        B, OC, H, W = out.shape
        out = out.permute(0, 2, 3, 1).contiguous()
        out = out.view(B, -1, OC)
        return out


class SingleStageDetector(nn.Module):
    """
    Single-stage detector using a small backbone and extra layers.
    Outputs: preds (B, N, 5+C), strides list for decoding
    """

    def __init__(self, num_classes: int = 1, image_size: int = 512):
        super().__init__()
        self.num_classes = num_classes
        self.image_size = image_size

        # reuse small backbone & extras from ssd.py
        self.backbone = SmallBackbone(in_channels=3)
        # backbone_out_channels expected: [64, 128]
        backbone_out_channels = [64, 128]
        self.extras = ExtraLayers(in_channels_list=backbone_out_channels)

        # feature channels used for heads: [64, 128, 128] (f2, f3, extra0)
        feat_channels = [backbone_out_channels[0], backbone_out_channels[1], backbone_out_channels[0] * 2]
        self.heads = nn.ModuleList([SingleStageHead(ch, num_classes) for ch in feat_channels])

        # approximate strides for these feature maps relative to input
        self.strides = [4, 8, 16]

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[int]]:
        B, C, H, W = x.shape
        f2, f3 = self.backbone(x)
        extra_feats = self.extras([f2, f3])
        feature_maps = [f2, f3] + extra_feats
        feature_maps = feature_maps[:3]

        outs = []
        for fmap, head in zip(feature_maps, self.heads):
            outs.append(head(fmap))  # (B, H*W, 5+C)

        preds = torch.cat(outs, dim=1)  # (B, N, 5+C)
        return preds, self.strides


def decode_predictions(preds: torch.Tensor, strides: List[int], image_size: int, conf_thresh: float = 0.25):
    """
    Decode model output preds (B=1) to boxes, scores, labels.
    preds: (B, N, 5+C)
    strides: list of ints matching heads order
    Returns: boxes (M,4) in pixel coords, scores (M,), labels (M,)
    """
    if preds.shape[0] != 1:
        raise ValueError("decode_predictions supports batch size 1")

    pred = preds[0]  # (N, 5+C)
    D = pred.shape[1]
    num_classes = D - 5

    boxes_list = []
    scores_list = []
    labels_list = []

    offset = 0
    for s in strides:
        fh = math.ceil(image_size / s)
        fw = math.ceil(image_size / s)
        fmap_n = fh * fw
        sub = pred[offset : offset + fmap_n]
        offset += fmap_n

        if sub.shape[0] == 0:
            continue

        tx = sub[:, 0]
        ty = sub[:, 1]
        tw = sub[:, 2]
        th = sub[:, 3]
        to = torch.sigmoid(sub[:, 4])
        if num_classes > 0:
            tcls = torch.sigmoid(sub[:, 5:])
            cls_scores, cls_idx = torch.max(tcls, dim=1)
        else:
            cls_scores = torch.zeros_like(to)
            cls_idx = torch.zeros_like(cls_scores, dtype=torch.long)

        # build grid centers
        grid_y = torch.arange(fh, dtype=torch.float32)
        grid_x = torch.arange(fw, dtype=torch.float32)
        gy, gx = torch.meshgrid(grid_y, grid_x, indexing="ij")
        gx = gx.reshape(-1)
        gy = gy.reshape(-1)
        cx = (gx + 0.5) * s
        cy = (gy + 0.5) * s

        bx = cx.to(tx.device) + tx * s
        by = cy.to(ty.device) + ty * s
        bw = torch.exp(tw) * s
        bh = torch.exp(th) * s

        x1 = bx - bw / 2.0
        y1 = by - bh / 2.0
        x2 = bx + bw / 2.0
        y2 = by + bh / 2.0

        scores = to * cls_scores
        keep = scores > conf_thresh
        if keep.any():
            boxes_list.append(torch.stack([x1[keep], y1[keep], x2[keep], y2[keep]], dim=1))
            scores_list.append(scores[keep])
            labels_list.append(cls_idx[keep])

    if not boxes_list:
        return torch.zeros((0, 4)), torch.zeros((0,)), torch.zeros((0,), dtype=torch.long)

    boxes = torch.cat(boxes_list, dim=0).cpu()
    scores = torch.cat(scores_list, dim=0).cpu()
    labels = torch.cat(labels_list, dim=0).cpu()
    return boxes, scores, labels