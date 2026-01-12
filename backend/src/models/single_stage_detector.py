# src/models/single_stage_detector.py
"""
Lightweight single-stage object detector composed of:
- Backbone
- Feature projection tower
- Prediction head (boxes + logits)
No external libraries required. No references to YOLO.
"""

import torch
import torch.nn as nn
import math


# ---------------------------------------------------
# Basic Convolutional Block
# ---------------------------------------------------
def conv_block(c_in, c_out, k=3, s=1, p=1):
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, k, s, p),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
    )


# ---------------------------------------------------
# Backbone network (simple conv tower)
# ---------------------------------------------------
class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.stage1 = nn.Sequential(
            conv_block(3, 32, k=3, s=2),
            conv_block(32, 32)
        )
        self.stage2 = nn.Sequential(
            conv_block(32, 64, k=3, s=2),
            conv_block(64, 64)
        )
        self.stage3 = nn.Sequential(
            conv_block(64, 128, k=3, s=2),
            conv_block(128, 128)
        )

    def forward(self, x):
        f1 = self.stage1(x)   # /2
        f2 = self.stage2(f1)  # /4
        f3 = self.stage3(f2)  # /8
        return f2, f3


# ---------------------------------------------------
# Feature projection layer
# ---------------------------------------------------
class FeatureProjector(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.block = nn.Sequential(
            conv_block(c_in, c_out, k=3, s=2),
            conv_block(c_out, c_out)
        )

    def forward(self, x):
        return self.block(x)


# ---------------------------------------------------
# Prediction head (box + class logits)
# ---------------------------------------------------
class DetectionHead(nn.Module):
    def __init__(self, c_in, num_anchors, num_classes):
        super().__init__()
        self.loc = nn.Conv2d(c_in, num_anchors * 4, 3, padding=1)
        self.cls = nn.Conv2d(c_in, num_anchors * num_classes, 3, padding=1)

    def forward(self, x):
        B = x.size(0)
        H, W = x.size(2), x.size(3)

        loc = self.loc(x).permute(0, 2, 3, 1).reshape(B, -1, 4)
        cls = self.cls(x).permute(0, 2, 3, 1).reshape(B, -1, -1)

        return loc, cls


# ---------------------------------------------------
# Build anchors
# ---------------------------------------------------
def generate_anchor_grid(fh, fw, stride, scales, ratios, img_size):
    anchors = []
    for i in range(fh):
        for j in range(fw):
            cy = (i + 0.5) * stride
            cx = (j + 0.5) * stride
            for s in scales:
                for ar in ratios:
                    w = s * math.sqrt(ar) * img_size
                    h = s / math.sqrt(ar) * img_size
                    anchors.append([
                        cx - w / 2, cy - h / 2,
                        cx + w / 2, cy + h / 2
                    ])
    return torch.tensor(anchors, dtype=torch.float32)


# ---------------------------------------------------
# The final detector
# ---------------------------------------------------
class SingleStageDetector(nn.Module):
    def __init__(
        self,
        num_classes=2,
        image_size=512,
        ratios=(0.5, 1.0, 2.0),
        scales=(0.10, 0.20, 0.40),
    ):
        super().__init__()

        self.num_classes = num_classes
        self.image_size = image_size
        self.ratios = list(ratios)
        self.scales = list(scales)

        self.backbone = Backbone()

        # project backbone features
        self.p3 = FeatureProjector(64, 128)
        self.p4 = FeatureProjector(128, 256)

        self.num_anchors = len(self.ratios) * len(self.scales)

        # heads
        self.head3 = DetectionHead(128, self.num_anchors, num_classes)
        self.head4 = DetectionHead(256, self.num_anchors, num_classes)
        self.head5 = DetectionHead(256, self.num_anchors, num_classes)

    def forward(self, x):
        f2, f3 = self.backbone(x)

        p3 = self.p3(f2)
        p4 = self.p4(f3)

        # For simplicity use three maps: p3, p4, f3
        fmaps = [p3, p4, f3]

        locs_all = []
        cls_all = []
        anchors_all = []

        for fmap in fmaps:
            B, C, H, W = fmap.shape

            loc, cls = DetectionHead(C, self.num_anchors, self.num_classes)(fmap)

            locs_all.append(loc)
            cls_all.append(cls)

            stride = self.image_size / H
            anchors = generate_anchor_grid(H, W, stride, self.scales, self.ratios, self.image_size)
            anchors_all.append(anchors)

        locs = torch.cat(locs_all, dim=1)
        cls = torch.cat(cls_all, dim=1)
        anchors = torch.cat(anchors_all, dim=0)

        return locs, cls, anchors