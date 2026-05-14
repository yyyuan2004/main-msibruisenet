"""PIDNet (CVPR 2023): A Real-time Semantic Segmentation Network Inspired by PID Controllers.

Pure PyTorch implementation — no mmcv / mmseg dependencies.
Variants: Small (~7.6M), Medium (~23M).

Note: This integration uses only the main P-branch output and applies
the project-wide CE+Dice loss for fair comparison with other baselines.
The original PIDNet boundary loss and auxiliary heads are intentionally
removed.

Reference: Xu et al., "PIDNet: A Real-time Semantic Segmentation Network
Inspired from PID Controllers", CVPR 2023.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F


bn_mom = 0.1


# ---------------------------------------------------------------------------
# building blocks
# ---------------------------------------------------------------------------

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 no_relu=False):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=bn_mom)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=bn_mom)
        self.downsample = downsample
        self.no_relu = no_relu

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        out = out + residual
        if self.no_relu:
            return out
        return self.relu(out)


class Bottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 no_relu=True):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=bn_mom)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=bn_mom)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion, momentum=bn_mom)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.no_relu = no_relu

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        out = out + residual
        if self.no_relu:
            return out
        return self.relu(out)


class SegmentHead(nn.Module):
    def __init__(self, inplanes, interplanes, outplanes, scale_factor=None):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(inplanes, momentum=bn_mom)
        self.conv1 = nn.Conv2d(inplanes, interplanes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(interplanes, momentum=bn_mom)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(interplanes, outplanes, 1, bias=True)
        self.scale_factor = scale_factor

    def forward(self, x):
        x = self.conv1(self.relu(self.bn1(x)))
        out = self.conv2(self.relu(self.bn2(x)))
        if self.scale_factor is not None:
            out = F.interpolate(out,
                                scale_factor=self.scale_factor,
                                mode='bilinear', align_corners=False)
        return out


# ---------------------------------------------------------------------------
# multi-scale pooling modules
# ---------------------------------------------------------------------------

class PAPPM(nn.Module):
    def __init__(self, inplanes, branch_planes, outplanes):
        super().__init__()
        self.scale1 = nn.Sequential(
            nn.AvgPool2d(5, 2, 2),
            nn.BatchNorm2d(inplanes, momentum=bn_mom), nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, branch_planes, 1, bias=False))
        self.scale2 = nn.Sequential(
            nn.AvgPool2d(9, 4, 4),
            nn.BatchNorm2d(inplanes, momentum=bn_mom), nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, branch_planes, 1, bias=False))
        self.scale3 = nn.Sequential(
            nn.AvgPool2d(17, 8, 8),
            nn.BatchNorm2d(inplanes, momentum=bn_mom), nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, branch_planes, 1, bias=False))
        self.scale4 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.BatchNorm2d(inplanes, momentum=bn_mom), nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, branch_planes, 1, bias=False))
        self.scale0 = nn.Sequential(
            nn.BatchNorm2d(inplanes, momentum=bn_mom), nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, branch_planes, 1, bias=False))
        self.scale_process = nn.Sequential(
            nn.BatchNorm2d(branch_planes * 4, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_planes * 4, branch_planes * 4, 3, 1, 1,
                      groups=4, bias=False))
        self.compression = nn.Sequential(
            nn.BatchNorm2d(branch_planes * 5, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_planes * 5, outplanes, 1, bias=False))
        self.shortcut = nn.Sequential(
            nn.BatchNorm2d(inplanes, momentum=bn_mom), nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, outplanes, 1, bias=False))

    def forward(self, x):
        H, W = x.shape[2:]
        x_ = self.scale0(x)
        s = [F.interpolate(self.scale1(x), (H, W), mode='bilinear',
                           align_corners=False) + x_,
             F.interpolate(self.scale2(x), (H, W), mode='bilinear',
                           align_corners=False) + x_,
             F.interpolate(self.scale3(x), (H, W), mode='bilinear',
                           align_corners=False) + x_,
             F.interpolate(self.scale4(x), (H, W), mode='bilinear',
                           align_corners=False) + x_]
        out = self.compression(
            torch.cat([x_, self.scale_process(torch.cat(s, 1))], 1)
        ) + self.shortcut(x)
        return out


# ---------------------------------------------------------------------------
# cross-branch fusion modules
# ---------------------------------------------------------------------------

class PagFM(nn.Module):
    def __init__(self, in_channels, mid_channels):
        super().__init__()
        self.f_x = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 1, bias=False),
            nn.BatchNorm2d(mid_channels))
        self.f_y = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 1, bias=False),
            nn.BatchNorm2d(mid_channels))

    def forward(self, x, y):
        H, W = x.shape[2:]
        y_q = F.interpolate(self.f_y(y), (H, W), mode='bilinear',
                            align_corners=False)
        x_k = self.f_x(x)
        sim = torch.sigmoid(torch.sum(x_k * y_q, dim=1, keepdim=True))
        y_up = F.interpolate(y, (H, W), mode='bilinear', align_corners=False)
        return (1 - sim) * x + sim * y_up


class LightBag(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_p = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels))
        self.conv_i = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels))

    def forward(self, p, i, d):
        edge = torch.sigmoid(d)
        return self.conv_p((1 - edge) * i + p) + self.conv_i(i + edge * p)


# ---------------------------------------------------------------------------
# PIDNet
# ---------------------------------------------------------------------------

class PIDNetModel(nn.Module):
    def __init__(self, m, n, planes, ppm_planes, head_planes,
                 num_classes=2, in_channels=3):
        super().__init__()
        self.relu = nn.ReLU(inplace=False)

        # stem
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, planes, 3, 2, 1, bias=False),
            nn.BatchNorm2d(planes, momentum=bn_mom), nn.ReLU(inplace=True),
            nn.Conv2d(planes, planes, 3, 2, 1, bias=False),
            nn.BatchNorm2d(planes, momentum=bn_mom), nn.ReLU(inplace=True))

        # shared
        self.layer1 = self._make_layer(BasicBlock, planes, planes, m)
        self.layer2 = self._make_layer(BasicBlock, planes, planes * 2, m,
                                       stride=2)

        # I-branch
        self.layer3 = self._make_layer(BasicBlock, planes * 2, planes * 4, n,
                                       stride=2)
        self.layer4 = self._make_layer(BasicBlock, planes * 4, planes * 8, n,
                                       stride=2)
        self.layer5 = self._make_layer(Bottleneck, planes * 8, planes * 8, 2,
                                       stride=2)

        # P-branch
        self.compression3 = nn.Sequential(
            nn.Conv2d(planes * 4, planes * 2, 1, bias=False),
            nn.BatchNorm2d(planes * 2, momentum=bn_mom))
        self.compression4 = nn.Sequential(
            nn.Conv2d(planes * 8, planes * 2, 1, bias=False),
            nn.BatchNorm2d(planes * 2, momentum=bn_mom))
        self.pag3 = PagFM(planes * 2, planes)
        self.pag4 = PagFM(planes * 2, planes)
        self.layer3_ = self._make_layer(BasicBlock, planes * 2, planes * 2, m)
        self.layer4_ = self._make_layer(BasicBlock, planes * 2, planes * 2, m)
        self.layer5_ = self._make_layer(Bottleneck, planes * 2, planes * 2, 1)

        # D-branch (m==2 path for S/M)
        self.layer3_d = self._make_single_layer(BasicBlock, planes * 2,
                                                planes)
        self.layer4_d = self._make_layer(Bottleneck, planes, planes, 1)
        self.diff3 = nn.Sequential(
            nn.Conv2d(planes * 4, planes, 3, 1, 1, bias=False),
            nn.BatchNorm2d(planes, momentum=bn_mom))
        self.diff4 = nn.Sequential(
            nn.Conv2d(planes * 8, planes * 2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(planes * 2, momentum=bn_mom))
        self.spp = PAPPM(planes * 16, ppm_planes, planes * 4)
        self.dfm = LightBag(planes * 4, planes * 4)
        self.layer5_d = self._make_layer(Bottleneck, planes * 2, planes * 2,
                                         1)

        # prediction head
        self.final_layer = SegmentHead(planes * 4, head_planes, num_classes)

        self._init_weights()

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion, 1, stride,
                          bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=bn_mom))
        layers = [block(inplanes, planes, stride, downsample)]
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            no_relu = (i == blocks - 1)
            layers.append(block(inplanes, planes, no_relu=no_relu))
        return nn.Sequential(*layers)

    def _make_single_layer(self, block, inplanes, planes, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion, 1, stride,
                          bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=bn_mom))
        return block(inplanes, planes, stride, downsample, no_relu=True)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        input_size = x.shape[2:]
        h_out = x.shape[2] // 8
        w_out = x.shape[3] // 8

        # shared stem + layers
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.relu(self.layer2(self.relu(x)))

        # three branches diverge
        x_ = self.layer3_(x)
        x_d = self.layer3_d(x)

        x = self.relu(self.layer3(x))
        x_ = self.pag3(x_, self.compression3(x))
        x_d = x_d + F.interpolate(self.diff3(x), (h_out, w_out),
                                   mode='bilinear', align_corners=False)

        x = self.relu(self.layer4(x))
        x_ = self.layer4_(self.relu(x_))
        x_d = self.layer4_d(self.relu(x_d))

        x_ = self.pag4(x_, self.compression4(x))
        x_d = x_d + F.interpolate(self.diff4(x), (h_out, w_out),
                                   mode='bilinear', align_corners=False)

        x_ = self.layer5_(self.relu(x_))
        x_d = self.layer5_d(self.relu(x_d))
        x = F.interpolate(self.spp(self.layer5(x)), (h_out, w_out),
                          mode='bilinear', align_corners=False)

        logits = self.final_layer(self.dfm(x_, x, x_d))
        return F.interpolate(logits, input_size, mode='bilinear',
                             align_corners=False)


# ---------------------------------------------------------------------------
# variant configs
# ---------------------------------------------------------------------------

PIDNET_VARIANTS = {
    'small':  dict(m=2, n=3, planes=32,  ppm_planes=96,  head_planes=128),
    'medium': dict(m=2, n=3, planes=64,  ppm_planes=96,  head_planes=128),
}


def _adapt_conv_weight(weight, target_in_ch):
    if weight.shape[1] == target_in_ch:
        return weight
    out_ch = weight.shape[0]
    new_w = torch.zeros(out_ch, target_in_ch, *weight.shape[2:])
    copy_ch = min(target_in_ch, 3)
    new_w[:, :copy_ch] = weight[:, :copy_ch]
    if target_in_ch > 3:
        nn.init.kaiming_normal_(new_w[:, 3:], mode='fan_out', nonlinearity='relu')
    return new_w


def build_pidnet(cfg):
    model_cfg = cfg['model']
    variant = model_cfg.get('variant', 'small')
    vcfg = PIDNET_VARIANTS[variant]

    in_ch = cfg['data'].get('num_channels', 4)
    num_classes = model_cfg.get('num_classes', 2)

    model = PIDNetModel(
        m=vcfg['m'], n=vcfg['n'], planes=vcfg['planes'],
        ppm_planes=vcfg['ppm_planes'], head_planes=vcfg['head_planes'],
        num_classes=num_classes, in_channels=in_ch)

    pretrained = model_cfg.get('pretrained_path', '')
    if pretrained and os.path.isfile(pretrained):
        ckpt = torch.load(pretrained, map_location='cpu', weights_only=False)
        sd = ckpt.get('state_dict', ckpt)
        # adapt 3ch stem to in_ch
        stem_key = 'conv1.0.weight'
        if in_ch != 3 and stem_key in sd:
            sd[stem_key] = _adapt_conv_weight(sd[stem_key], in_ch)
        # filter mismatched shapes
        model_sd = model.state_dict()
        sd = {k: v for k, v in sd.items()
              if k in model_sd and v.shape == model_sd[k].shape}
        missing = set(model_sd) - set(sd)
        model_sd.update(sd)
        model.load_state_dict(model_sd, strict=False)
        print(f"[PIDNet-{variant}] Loaded {len(sd)} params from {pretrained} "
              f"({len(missing)} missing)")
    else:
        print(f"[PIDNet-{variant}] Training from scratch (no pretrained at "
              f"'{pretrained}')")

    return model
