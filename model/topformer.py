"""TopFormer (CVPR 2022): Token Pyramid Transformer for Mobile Semantic Segmentation.

Pure PyTorch implementation — no mmcv / mmseg dependencies.
Variants: Tiny (~1.4M), Small (~3.1M), Base (~5.1M).

Reference: Zhang et al., "TopFormer: Token Pyramid Transformer for Mobile
Semantic Segmentation", CVPR 2022.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class DropPath(nn.Module):
    def __init__(self, p=0.):
        super().__init__()
        self.p = p

    def forward(self, x):
        if self.p == 0. or not self.training:
            return x
        keep = 1 - self.p
        mask = x.new_empty(x.shape[0], *((1,) * (x.ndim - 1))).bernoulli_(keep)
        return x.div(keep) * mask


class Conv2dBN(nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1.0):
        super().__init__()
        self.add_module('c', nn.Conv2d(a, b, ks, stride, pad, dilation,
                                       groups, bias=False))
        bn = nn.BatchNorm2d(b)
        nn.init.constant_(bn.weight, bn_weight_init)
        nn.init.constant_(bn.bias, 0)
        self.add_module('bn', bn)


# ---------------------------------------------------------------------------
# CNN local branch
# ---------------------------------------------------------------------------

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, ks, stride, expand_ratio):
        super().__init__()
        self.use_res_connect = (stride == 1 and inp == oup)
        hidden = int(round(inp * expand_ratio))
        layers = []
        if expand_ratio != 1:
            layers += [Conv2dBN(inp, hidden, ks=1), nn.ReLU6(inplace=True)]
        layers += [
            Conv2dBN(hidden, hidden, ks=ks, stride=stride,
                     pad=ks // 2, groups=hidden),
            nn.ReLU6(inplace=True),
            Conv2dBN(hidden, oup, ks=1),
        ]
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        return self.conv(x)


class TokenPyramidModule(nn.Module):
    def __init__(self, cfgs, out_indices, in_channels=3, inp_channel=16):
        super().__init__()
        self.out_indices = out_indices
        self.stem = nn.Sequential(
            Conv2dBN(in_channels, inp_channel, 3, 2, 1), nn.ReLU6(inplace=True))
        self._layer_names = []
        ch = inp_channel
        for i, (k, t, c, s) in enumerate(cfgs):
            out_ch = _make_divisible(c, 8)
            name = f'layer{i + 1}'
            self.add_module(name, InvertedResidual(ch, out_ch, ks=k,
                                                   stride=s, expand_ratio=t))
            self._layer_names.append(name)
            ch = out_ch

    def forward(self, x):
        outs = []
        x = self.stem(x)
        for i, name in enumerate(self._layer_names):
            x = getattr(self, name)(x)
            if i in self.out_indices:
                outs.append(x)
        return outs


# ---------------------------------------------------------------------------
# Transformer global branch
# ---------------------------------------------------------------------------

class Mlp(nn.Module):
    def __init__(self, dim, hidden_dim, drop=0.):
        super().__init__()
        self.fc1 = Conv2dBN(dim, hidden_dim)
        self.dw = nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1,
                            groups=hidden_dim, bias=True)
        self.act = nn.ReLU6(inplace=True)
        self.fc2 = Conv2dBN(hidden_dim, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.act(self.dw(self.fc1(x)))
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, key_dim, num_heads, attn_ratio=2):
        super().__init__()
        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.d = int(attn_ratio * key_dim)
        self.to_q = Conv2dBN(dim, key_dim * num_heads)
        self.to_k = Conv2dBN(dim, key_dim * num_heads)
        self.to_v = Conv2dBN(dim, self.d * num_heads)
        self.proj = nn.Sequential(
            nn.ReLU6(inplace=True),
            Conv2dBN(self.d * num_heads, dim, bn_weight_init=0))

    def forward(self, x):
        B, _, H, W = x.shape
        N = H * W
        nh, kd, d = self.num_heads, self.key_dim, self.d
        q = self.to_q(x).reshape(B, nh, kd, N).permute(0, 1, 3, 2)
        k = self.to_k(x).reshape(B, nh, kd, N)
        v = self.to_v(x).reshape(B, nh, d, N).permute(0, 1, 3, 2)
        attn = (q @ k).mul_(self.scale).softmax(dim=-1)
        out = (attn @ v).permute(0, 1, 3, 2).reshape(B, d * nh, H, W)
        return self.proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, dim, key_dim, num_heads, mlp_ratio=2., attn_ratio=2.,
                 drop_path=0.):
        super().__init__()
        self.attn = Attention(dim, key_dim, num_heads, attn_ratio)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.mlp = Mlp(dim, int(dim * mlp_ratio))

    def forward(self, x):
        x = x + self.drop_path(self.attn(x))
        x = x + self.drop_path(self.mlp(x))
        return x


class BasicLayer(nn.Module):
    def __init__(self, depth, embed_dim, key_dim, num_heads,
                 mlp_ratio=2., attn_ratio=2., drop_path=None):
        super().__init__()
        if drop_path is None:
            drop_path = [0.] * depth
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, key_dim, num_heads, mlp_ratio,
                             attn_ratio, drop_path=drop_path[i])
            for i in range(depth)])

    def forward(self, x):
        for blk in self.transformer_blocks:
            x = blk(x)
        return x


# ---------------------------------------------------------------------------
# Token pooling & injection
# ---------------------------------------------------------------------------

class PyramidPoolAgg(nn.Module):
    def __init__(self, stride):
        super().__init__()
        self.stride = stride

    def forward(self, inputs):
        _, _, H, W = inputs[-1].shape
        tH = (H - 1) // self.stride + 1
        tW = (W - 1) // self.stride + 1
        return torch.cat([F.adaptive_avg_pool2d(t, (tH, tW)) for t in inputs],
                         dim=1)


class HSigmoid(nn.Module):
    def forward(self, x):
        return F.relu6(x + 3, inplace=True) / 6


class InjectionMultiSum(nn.Module):
    def __init__(self, inp, oup):
        super().__init__()
        self.local_embedding = nn.Sequential(
            nn.Conv2d(inp, oup, 1, bias=False), nn.BatchNorm2d(oup))
        self.global_embedding = nn.Sequential(
            nn.Conv2d(inp, oup, 1, bias=False), nn.BatchNorm2d(oup))
        self.global_act = nn.Sequential(
            nn.Conv2d(inp, oup, 1, bias=False), nn.BatchNorm2d(oup))
        self.act = HSigmoid()

    def forward(self, x_l, x_g):
        H, W = x_l.shape[2:]
        local_feat = self.local_embedding(x_l)
        g_act = F.interpolate(self.act(self.global_act(x_g)), (H, W),
                               mode='bilinear', align_corners=False)
        g_feat = F.interpolate(self.global_embedding(x_g), (H, W),
                               mode='bilinear', align_corners=False)
        return local_feat * g_act + g_feat


# ---------------------------------------------------------------------------
# Segmentation head
# ---------------------------------------------------------------------------

class SimpleHead(nn.Module):
    def __init__(self, channels, num_classes, is_dw=False, dropout=0.1):
        super().__init__()
        self.linear_fuse = nn.Sequential(
            nn.Conv2d(channels, channels, 1,
                      groups=channels if is_dw else 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True))
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.conv_seg = nn.Conv2d(channels, num_classes, 1)

    def forward(self, inputs):
        out = inputs[0]
        for feat in inputs[1:]:
            out = out + F.interpolate(feat, size=out.shape[2:],
                                      mode='bilinear', align_corners=False)
        return self.conv_seg(self.dropout(self.linear_fuse(out)))


# ---------------------------------------------------------------------------
# TopFormer full model
# ---------------------------------------------------------------------------

class TopFormerModel(nn.Module):
    def __init__(self, cfgs, channels, out_channels, embed_out_indice,
                 decode_out_indices, num_heads, num_classes=2,
                 in_channels=3, depths=4, key_dim=16, attn_ratio=2,
                 mlp_ratio=2, c2t_stride=2, drop_path_rate=0.1,
                 is_dw=False, head_dropout=0.1):
        super().__init__()
        self.channels = channels
        self.decode_out_indices = decode_out_indices

        self.tpm = TokenPyramidModule(cfgs, embed_out_indice, in_channels)
        self.ppa = PyramidPoolAgg(c2t_stride)

        embed_dim = sum(channels)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depths)]
        self.trans = BasicLayer(depths, embed_dim, key_dim, num_heads,
                                mlp_ratio, attn_ratio, dpr)

        self.SIM = nn.ModuleList()
        for i in range(len(channels)):
            if i in decode_out_indices:
                self.SIM.append(InjectionMultiSum(channels[i], out_channels[i]))
            else:
                self.SIM.append(nn.Identity())

        head_ch = out_channels[decode_out_indices[0]]
        self.head = SimpleHead(head_ch, num_classes, is_dw, head_dropout)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        input_size = x.shape[2:]
        tpm_outs = self.tpm(x)
        ppa_out = self.ppa(tpm_outs)
        trans_out = self.trans(ppa_out)
        tokens = trans_out.split(self.channels, dim=1)

        results = []
        for i in range(len(self.channels)):
            if i in self.decode_out_indices:
                results.append(self.SIM[i](tpm_outs[i], tokens[i]))

        logits = self.head(results)
        return F.interpolate(logits, input_size, mode='bilinear',
                             align_corners=False)


# ---------------------------------------------------------------------------
# variant configs
# ---------------------------------------------------------------------------

TOPFORMER_VARIANTS = {
    'tiny': dict(
        cfgs=[
            [3, 1, 16, 1], [3, 4, 16, 2], [3, 3, 16, 1],
            [5, 3, 32, 2], [5, 3, 32, 1],
            [3, 3, 64, 2], [3, 3, 64, 1],
            [5, 6, 96, 2], [5, 6, 96, 1],
        ],
        channels=[16, 32, 64, 96],
        out_channels=[None, 128, 128, 128],
        embed_out_indice=[2, 4, 6, 8],
        decode_out_indices=[1, 2, 3],
        num_heads=4, is_dw=True,
    ),
    'small': dict(
        cfgs=[
            [3, 1, 16, 1], [3, 4, 24, 2], [3, 3, 24, 1],
            [5, 3, 48, 2], [5, 3, 48, 1],
            [3, 3, 96, 2], [3, 3, 96, 1],
            [5, 6, 128, 2], [5, 6, 128, 1], [3, 6, 128, 1],
        ],
        channels=[24, 48, 96, 128],
        out_channels=[None, 192, 192, 192],
        embed_out_indice=[2, 4, 6, 9],
        decode_out_indices=[1, 2, 3],
        num_heads=6, is_dw=False,
    ),
    'base': dict(
        cfgs=[
            [3, 1, 16, 1], [3, 4, 32, 2], [3, 3, 32, 1],
            [5, 3, 64, 2], [5, 3, 64, 1],
            [3, 3, 128, 2], [3, 3, 128, 1],
            [5, 6, 160, 2], [5, 6, 160, 1], [3, 6, 160, 1],
        ],
        channels=[32, 64, 128, 160],
        out_channels=[None, 256, 256, 256],
        embed_out_indice=[2, 4, 6, 9],
        decode_out_indices=[1, 2, 3],
        num_heads=8, is_dw=False,
    ),
}


def _adapt_conv_weight(weight, target_in_ch):
    """Expand 3-channel conv weight to target_in_ch, Kaiming-init extra channels."""
    if weight.shape[1] == target_in_ch:
        return weight
    out_ch = weight.shape[0]
    new_w = torch.zeros(out_ch, target_in_ch, *weight.shape[2:])
    new_w[:, :3] = weight
    nn.init.kaiming_normal_(new_w[:, 3:], mode='fan_out', nonlinearity='relu')
    return new_w


def build_topformer(cfg):
    model_cfg = cfg['model']
    variant = model_cfg.get('variant', 'tiny')
    vcfg = TOPFORMER_VARIANTS[variant]

    in_ch = cfg['data'].get('num_channels', 4)
    num_classes = model_cfg.get('num_classes', 2)

    model = TopFormerModel(
        cfgs=vcfg['cfgs'],
        channels=vcfg['channels'],
        out_channels=vcfg['out_channels'],
        embed_out_indice=vcfg['embed_out_indice'],
        decode_out_indices=vcfg['decode_out_indices'],
        num_heads=vcfg['num_heads'],
        num_classes=num_classes,
        in_channels=in_ch,
        is_dw=vcfg.get('is_dw', False),
    )

    pretrained = model_cfg.get('pretrained_path', '')
    if pretrained and os.path.isfile(pretrained):
        ckpt = torch.load(pretrained, map_location='cpu', weights_only=False)
        sd = ckpt.get('state_dict_ema',
                       ckpt.get('state_dict', ckpt.get('model', ckpt)))
        if in_ch != 3 and 'tpm.stem.0.c.weight' in sd:
            sd['tpm.stem.0.c.weight'] = _adapt_conv_weight(
                sd['tpm.stem.0.c.weight'], in_ch)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"[TopFormer-{variant}] Loaded pretrained: {pretrained} "
              f"(missing={len(missing)}, unexpected={len(unexpected)})")
    else:
        print(f"[TopFormer-{variant}] Training from scratch (no pretrained at "
              f"'{pretrained}')")

    return model
