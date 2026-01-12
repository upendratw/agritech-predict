# single_stage_detector.py

import torch
import torch.nn as nn
import math


def conv_block(in_ch, out_ch, k=3, s=1, p=1):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class FeatureExtractor(nn.Module):
    """
    Small backbone producing feature maps at 1/4 and 1/8 resolution.
    """

    def __init__(self):
        super().__init__()
        self.stage1 = nn.Sequential(
            conv_block(3, 32, 3, 2, 1),
            conv_block(32, 32),
        )
        self.stage2 = nn.Sequential(
            conv_block(32, 64, 3, 2, 1),
            conv_block(64, 64),
        )
        self.stage3 = nn.Sequential(
            conv_block(64, 128, 3, 2, 1),
            conv_block(128, 128),
        )

    def forward(self, x):
        x = self.stage1(x)
        f2 = self.stage2(x)
        f3 = self.stage3(f2)
        return f2, f3


class DetectionHead(nn.Module):
    """
    Head predicting bounding box regression + class logits
    """

    def __init__(self, in_channels, num_anchors, num_classes):
        super().__init__()
        self.num_anchors = num_anchors
        self.num_classes = num_classes

        self.loc = nn.Conv2d(in_channels, num_anchors * 4, 3, padding=1)
        self.cls = nn.Conv2d(in_channels, num_anchors * num_classes, 3, padding=1)

        nn.init.normal_(self.loc.weight, std=0.01)
        nn.init.constant_(self.loc.bias, 0)
        nn.init.normal_(self.cls.weight, std=0.01)
        nn.init.constant_(self.cls.bias, 0)

    def forward(self, x):
        B = x.size(0)
        loc = self.loc(x).permute(0, 2, 3, 1).reshape(B, -1, 4)
        cls = self.cls(x).permute(0, 2, 3, 1).reshape(B, -1, self.num_classes)
        return loc, cls


class SingleStageDetector(nn.Module):

    def __init__(self, num_classes=2, num_anchors=3, image_size=512):
        super().__init__()
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        self.image_size = image_size

        self.backbone = FeatureExtractor()

        # HEADS MUST BE IN __init__ — NOT INSIDE forward()
        self.head2 = DetectionHead(64, num_anchors, num_classes)
        self.head3 = DetectionHead(128, num_anchors, num_classes)

    def forward(self, x):
        f2, f3 = self.backbone(x)

        loc2, cls2 = self.head2(f2)
        loc3, cls3 = self.head3(f3)

        loc = torch.cat([loc2, loc3], dim=1)
        cls = torch.cat([cls2, cls3], dim=1)

        # Dummy anchors to avoid crash — your project computes them separately
        anchors = torch.zeros(loc.size(1), 4, device=x.device)

        return loc, cls, anchors