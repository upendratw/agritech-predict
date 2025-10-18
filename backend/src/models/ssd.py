# src/models/ssd.py
"""
Small SSD-like detector (from-scratch, no pretrained weights).

- Lightweight backbone (conv blocks)
- Extra feature layers for multi-scale detection
- Prediction heads (localization + classification) per feature map
- Simple anchor/default-box generator (grid anchors with multiple aspect ratios)

Model outputs:
  locs: (B, num_anchors, 4)  - predicted deltas (SSD encoding)
  confs: (B, num_anchors, num_classes) - class logits (no softmax)
  anchors: (num_anchors, 4) anchors in xyxy image coordinates
"""
from typing import List, Tuple
import math

import torch
import torch.nn as nn

# Optional utils import (may be None depending on your project layout)
try:
    from models import utils as box_utils
except Exception:
    box_utils = None


def _make_conv_block(in_ch, out_ch, kernel=3, stride=1, padding=1):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=kernel, stride=stride, padding=padding),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class SmallBackbone(nn.Module):
    """
    Very small backbone producing feature maps at different scales.
    Produces feature maps f2 (/4) and f3 (/8) with channels 64 and 128 respectively.
    """

    def __init__(self, in_channels=3):
        super().__init__()
        self.conv1 = nn.Sequential(
            _make_conv_block(in_channels, 32, kernel=3, stride=2, padding=1),  # /2
            _make_conv_block(32, 32, kernel=3, stride=1, padding=1),
        )
        self.conv2 = nn.Sequential(
            _make_conv_block(32, 64, kernel=3, stride=2, padding=1),  # /4
            _make_conv_block(64, 64, kernel=3, stride=1, padding=1),
        )
        self.conv3 = nn.Sequential(
            _make_conv_block(64, 128, kernel=3, stride=2, padding=1),  # /8
            _make_conv_block(128, 128, kernel=3, stride=1, padding=1),
        )

    def forward(self, x):
        f1 = self.conv1(x)  # /2 (not used for heads)
        f2 = self.conv2(f1)  # /4 -> channels 64
        f3 = self.conv3(f2)  # /8 -> channels 128
        return f2, f3


class ExtraLayers(nn.Module):
    """
    Extra layers producing smaller spatial maps.
    Constructed with an 'in_channels_list' that indicates which backbone channels
    they are conceptually attached to. Each extra layer outputs out_ch = in_ch * 2
    in this simple design.
    """

    def __init__(self, in_channels_list: List[int]):
        super().__init__()
        self.extras = nn.ModuleList()
        for in_ch in in_channels_list:
            out_ch = in_ch * 2
            self.extras.append(
                nn.Sequential(
                    _make_conv_block(in_ch, out_ch, kernel=3, stride=2, padding=1),
                    _make_conv_block(out_ch, out_ch, kernel=3, stride=1, padding=1),
                )
            )

    def forward(self, feats: List[torch.Tensor]):
        out = []
        for f, layer in zip(feats, self.extras):
            out.append(layer(f))
        return out


class PredictionHead(nn.Module):
    """
    Prediction heads producing localization (4) and classification (num_classes) per anchor.
    """

    def __init__(self, in_channels: int, num_anchors: int, num_classes: int):
        super().__init__()
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        # localization head: 4 values per anchor
        self.loc_conv = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=3, padding=1)
        # classification head: num_classes logits per anchor
        self.cls_conv = nn.Conv2d(in_channels, num_anchors * num_classes, kernel_size=3, padding=1)

        # init
        nn.init.normal_(self.loc_conv.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.loc_conv.bias, 0.0)
        nn.init.normal_(self.cls_conv.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.cls_conv.bias, 0.0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return:
          loc: (B, H*W*num_anchors, 4)
          cls: (B, H*W*num_anchors, num_classes)
        """
        B, C, H, W = x.shape
        loc = self.loc_conv(x)  # (B, num_anchors*4, H, W)
        cls = self.cls_conv(x)  # (B, num_anchors*num_classes, H, W)

        loc = loc.permute(0, 2, 3, 1).contiguous()  # B,H,W,4*num_anchors
        loc = loc.view(B, -1, 4)

        cls = cls.permute(0, 2, 3, 1).contiguous()  # B,H,W,num_anchors*num_classes
        cls = cls.view(B, -1, self.num_classes)

        return loc, cls


class SSD(nn.Module):
    """
    Simple SSD-like detector.
    """

    def __init__(
        self,
        num_classes: int,
        image_size: int = 512,
        aspect_ratios: List[float] = (0.5, 1.0, 2.0),
        channels: List[int] = None,
    ):
        """
        Args:
          num_classes: number of target classes (NOT including background).
          image_size: used to size anchors (reference space).
        """
        super().__init__()
        self.num_classes = num_classes
        self.image_size = image_size

        # backbone and extras configuration
        self.backbone = SmallBackbone(in_channels=3)
        # these are the conceptual channel sizes of f2 and f3 produced by backbone
        backbone_out_channels = [64, 128]  # f2 channels, f3 channels

        # extras will be created from these backbone_out_channels and will output out_ch = in_ch*2
        self.extras = ExtraLayers(in_channels_list=backbone_out_channels)

        # Compute the feature-map channels we will actually use for heads.
        # Forward produces: feature_maps = [f2 (64), f3 (128), extra0(f2->128), extra1(f3->256)]
        # We take the first 3 feature maps to keep model compact: channels -> [64, 128, 128]
        # So derive feat_channels accordingly so heads get correct in_channels.
        if channels is None:
            # derive from backbone_out_channels: [f2, f3, extra0_out]
            self.feat_channels = [backbone_out_channels[0], backbone_out_channels[1], backbone_out_channels[0] * 2]
        else:
            self.feat_channels = channels

        # anchors setup
        self.aspect_ratios = list(aspect_ratios)
        self.num_anchors_per_loc = len(self.aspect_ratios)

        # create prediction heads using the derived feat_channels
        self.pred_heads = nn.ModuleList()
        for ch in self.feat_channels:
            self.pred_heads.append(PredictionHead(in_channels=ch, num_anchors=self.num_anchors_per_loc, num_classes=self.num_classes))

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def _generate_anchors_for_feature_map(self, fmap_h: int, fmap_w: int, scale: float) -> torch.Tensor:
        anchors = []
        step_y = self.image_size / fmap_h
        step_x = self.image_size / fmap_w

        for i in range(fmap_h):
            for j in range(fmap_w):
                cy = (i + 0.5) * step_y
                cx = (j + 0.5) * step_x
                for ar in self.aspect_ratios:
                    w = scale * math.sqrt(ar) * self.image_size
                    h = scale / math.sqrt(ar) * self.image_size
                    x1 = cx - w / 2.0
                    y1 = cy - h / 2.0
                    x2 = cx + w / 2.0
                    y2 = cy + h / 2.0
                    anchors.append([x1, y1, x2, y2])
        if len(anchors) == 0:
            return torch.zeros((0, 4), dtype=torch.float32)
        return torch.tensor(anchors, dtype=torch.float32)

    def generate_anchors(self, feature_maps: List[torch.Tensor]) -> torch.Tensor:
        anchors_all = []
        scales = [0.08, 0.16, 0.32]  # adjust as needed
        for fmap, s in zip(feature_maps, scales):
            _, _, fh, fw = fmap.shape
            a = self._generate_anchors_for_feature_map(fh, fw, s)
            anchors_all.append(a)
        if len(anchors_all) == 0:
            return torch.zeros((0, 4), dtype=torch.float32)
        return torch.cat(anchors_all, dim=0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
          locs: (B, num_anchors, 4)
          confs: (B, num_anchors, num_classes)
          anchors: (num_anchors, 4) -- floats in image pixel coords (reference image_size)
        """
        B, C, H, W = x.shape

        # backbone
        f2, f3 = self.backbone(x)  # f2: /4 (C=64), f3: /8 (C=128)
        # extras applied to the same inputs (zipped)
        extra_feats = self.extras([f2, f3])  # yields [extra0_out (C=128), extra1_out (C=256)]

        # build feature maps list and pick first 3 (keeps model small)
        feature_maps = [f2, f3] + extra_feats
        feature_maps = feature_maps[:3]  # channels will be [64,128,128]

        locs_list = []
        confs_list = []
        for fmap, head in zip(feature_maps, self.pred_heads):
            l_out, c_out = head(fmap)  # l_out: (B, n, 4), c_out: (B, n, num_classes)
            locs_list.append(l_out)
            confs_list.append(c_out)

        locs = torch.cat(locs_list, dim=1) if locs_list else torch.zeros((B, 0, 4), device=x.device)
        confs = torch.cat(confs_list, dim=1) if confs_list else torch.zeros((B, 0, self.num_classes), device=x.device)

        anchors = self.generate_anchors(feature_maps)  # (num_anchors, 4) on CPU by default

        return locs, confs, anchors