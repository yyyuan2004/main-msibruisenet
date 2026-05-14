"""SeaFormer (ICLR 2023): Squeeze-enhanced Axial Transformer for Mobile Segmentation.

Pure PyTorch implementation — no mmcv / mmseg dependencies.
Variants: Tiny (~1.7M), Small (~4M), Base (~8M).

Reference: Wan et al., "SeaFormer: Squeeze-Enhanced Axial Transformer for
Mobile Semantic Segmentation", ICLR 2023.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# helpers (shared with TopFormer)
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


class HSigmoid(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU6(inplace=True)

    def forward(self, x):
        return self.relu(x + 3) / 6


# ---------------------------------------------------------------------------
# MobileNetV2-style blocks
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


class StackedMV2Block(nn.Module):
    def __init__(self, cfgs, stem, inp_channel=16, in_channels=3):
        super().__init__()
        self.has_stem = stem
        if stem:
            self.stem_block = nn.Sequential(
                Conv2dBN(in_channels, inp_channel, 3, 2, 1),
                nn.ReLU6(inplace=True))
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
        if self.has_stem:
            x = self.stem_block(x)
        for name in self._layer_names:
            x = getattr(self, name)(x)
        return x


# ---------------------------------------------------------------------------
# Squeeze-enhanced Axial Attention
# ---------------------------------------------------------------------------

class SqueezeAxialPositionalEmbedding(nn.Module):
    def __init__(self, dim, shape=16):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.randn(1, dim, shape))

    def forward(self, x):
        B, C, N = x.shape
        return x + F.interpolate(self.pos_embed, size=(N,),
                                 mode='linear', align_corners=False)


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


class SeaAttention(nn.Module):
    def __init__(self, dim, key_dim, num_heads, attn_ratio=2):
        super().__init__()
        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.nh_kd = key_dim * num_heads
        self.d = int(attn_ratio * key_dim)
        self.dh = self.d * num_heads

        self.to_q = Conv2dBN(dim, self.nh_kd)
        self.to_k = Conv2dBN(dim, self.nh_kd)
        self.to_v = Conv2dBN(dim, self.dh)

        self.proj = nn.Sequential(
            nn.ReLU6(inplace=True),
            Conv2dBN(self.dh, dim, bn_weight_init=0))
        self.proj_encode_row = nn.Sequential(
            nn.ReLU6(inplace=True),
            Conv2dBN(self.dh, self.dh, bn_weight_init=0))
        self.proj_encode_column = nn.Sequential(
            nn.ReLU6(inplace=True),
            Conv2dBN(self.dh, self.dh, bn_weight_init=0))

        self.pos_emb_rowq = SqueezeAxialPositionalEmbedding(self.nh_kd, 16)
        self.pos_emb_rowk = SqueezeAxialPositionalEmbedding(self.nh_kd, 16)
        self.pos_emb_columnq = SqueezeAxialPositionalEmbedding(self.nh_kd, 16)
        self.pos_emb_columnk = SqueezeAxialPositionalEmbedding(self.nh_kd, 16)

        qkv_ch = 2 * self.nh_kd + self.dh
        self.dwconv = Conv2dBN(qkv_ch, qkv_ch, ks=3, stride=1, pad=1,
                               groups=qkv_ch)
        self.act = nn.ReLU6(inplace=True)
        self.pwconv = Conv2dBN(qkv_ch, dim)
        self.sigmoid = HSigmoid()

    def forward(self, x):
        B, C, H, W = x.shape
        nh, kd, d = self.num_heads, self.key_dim, self.d

        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        # detail-enhance branch
        qkv = self.pwconv(self.act(self.dwconv(torch.cat([q, k, v], dim=1))))

        # squeeze row attention
        qrow = self.pos_emb_rowq(q.mean(-1)).reshape(B, nh, kd, H).permute(0, 1, 3, 2)
        krow = self.pos_emb_rowk(k.mean(-1)).reshape(B, nh, kd, H)
        vrow = v.mean(-1).reshape(B, nh, d, H).permute(0, 1, 3, 2)
        attn_row = (qrow @ krow).mul_(self.scale).softmax(dim=-1)
        xx_row = self.proj_encode_row(
            (attn_row @ vrow).permute(0, 1, 3, 2).reshape(B, self.dh, H, 1))

        # squeeze column attention
        qcol = self.pos_emb_columnq(q.mean(-2)).reshape(B, nh, kd, W).permute(0, 1, 3, 2)
        kcol = self.pos_emb_columnk(k.mean(-2)).reshape(B, nh, kd, W)
        vcol = v.mean(-2).reshape(B, nh, d, W).permute(0, 1, 3, 2)
        attn_col = (qcol @ kcol).mul_(self.scale).softmax(dim=-1)
        xx_col = self.proj_encode_column(
            (attn_col @ vcol).permute(0, 1, 3, 2).reshape(B, self.dh, 1, W))

        xx = self.proj(v + xx_row + xx_col)
        return self.sigmoid(xx) * qkv


class TransformerBlock(nn.Module):
    def __init__(self, dim, key_dim, num_heads, mlp_ratio=4., attn_ratio=2.,
                 drop_path=0.):
        super().__init__()
        self.attn = SeaAttention(dim, key_dim, num_heads, attn_ratio)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.mlp = Mlp(dim, int(dim * mlp_ratio))

    def forward(self, x):
        x = x + self.drop_path(self.attn(x))
        x = x + self.drop_path(self.mlp(x))
        return x


class BasicLayer(nn.Module):
    def __init__(self, depth, embed_dim, key_dim, num_heads,
                 mlp_ratio=4., attn_ratio=2., drop_path=None):
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
# Segmentation head
# ---------------------------------------------------------------------------

class FusionBlock(nn.Module):
    def __init__(self, inp, oup, embed_dim):
        super().__init__()
        self.local_embedding = nn.Sequential(
            nn.Conv2d(inp, embed_dim, 1, bias=False), nn.BatchNorm2d(embed_dim))
        self.global_act = nn.Sequential(
            nn.Conv2d(oup, embed_dim, 1, bias=False), nn.BatchNorm2d(embed_dim))
        self.act = HSigmoid()

    def forward(self, x_l, x_g):
        H, W = x_l.shape[2:]
        local_feat = self.local_embedding(x_l)
        sig_act = F.interpolate(self.act(self.global_act(x_g)), (H, W),
                                mode='bilinear', align_corners=False)
        return local_feat * sig_act


class LightHead(nn.Module):
    def __init__(self, in_channels, head_channels, embed_dims,
                 num_classes, is_dw=True, dropout=0.1):
        super().__init__()
        self.embed_dims = embed_dims
        for i in range(len(embed_dims)):
            inp = in_channels[0] if i == 0 else embed_dims[i - 1]
            oup = in_channels[i + 1]
            setattr(self, f'fuse{i + 1}',
                    FusionBlock(inp, oup, embed_dims[i]))
        self.linear_fuse = nn.Sequential(
            nn.Conv2d(head_channels, head_channels, 1,
                      groups=head_channels if is_dw else 1, bias=False),
            nn.BatchNorm2d(head_channels),
            nn.ReLU(inplace=True))
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.conv_seg = nn.Conv2d(head_channels, num_classes, 1)

    def forward(self, inputs):
        x = inputs[0]
        for i in range(len(self.embed_dims)):
            fuse = getattr(self, f'fuse{i + 1}')
            x = fuse(x, inputs[i + 1])
        return self.conv_seg(self.dropout(self.linear_fuse(x)))


# ---------------------------------------------------------------------------
# Full SeaFormer model
# ---------------------------------------------------------------------------

class SeaFormerModel(nn.Module):
    def __init__(self, cfgs, channels, emb_dims, key_dims, depths,
                 num_heads, head_channels, head_embed_dims,
                 head_in_channels, num_classes=2, in_channels=3,
                 attn_ratio=2, mlp_ratios=None, drop_path_rate=0.1,
                 is_dw=True, head_dropout=0.1):
        super().__init__()
        self.num_smb = len(cfgs)
        self.num_trans = len(depths)
        if mlp_ratios is None:
            mlp_ratios = [2, 4]

        # MobileNetV2 stages
        for i in range(self.num_smb):
            smb = StackedMV2Block(cfgs[i], stem=(i == 0),
                                  inp_channel=channels[i],
                                  in_channels=in_channels)
            setattr(self, f'smb{i + 1}', smb)

        # Transformer stages
        for i in range(self.num_trans):
            dpr = [x.item() for x in
                   torch.linspace(0, drop_path_rate, depths[i])]
            trans = BasicLayer(depths[i], emb_dims[i], key_dims[i],
                               num_heads, mlp_ratios[i], attn_ratio, dpr)
            setattr(self, f'trans{i + 1}', trans)

        # Segmentation head
        self.head = LightHead(head_in_channels, head_channels,
                              head_embed_dims, num_classes, is_dw,
                              head_dropout)

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
        outputs = []
        for i in range(self.num_smb):
            x = getattr(self, f'smb{i + 1}')(x)
            if i == 1:
                outputs.append(x)
            if self.num_trans + i >= self.num_smb:
                ti = i + self.num_trans - self.num_smb + 1
                x = getattr(self, f'trans{ti}')(x)
                outputs.append(x)

        logits = self.head(outputs)
        return F.interpolate(logits, input_size, mode='bilinear',
                             align_corners=False)


# ---------------------------------------------------------------------------
# variant configs
# ---------------------------------------------------------------------------

SEAFORMER_VARIANTS = {
    'tiny': dict(
        cfgs=[
            [[3, 1, 16, 1], [3, 4, 16, 2], [3, 3, 16, 1]],
            [[5, 3, 32, 2], [5, 3, 32, 1]],
            [[3, 3, 64, 2], [3, 3, 64, 1]],
            [[5, 3, 128, 2]],
            [[3, 6, 160, 2]],
        ],
        channels=[16, 16, 32, 64, 128, 160],
        depths=[2, 2], emb_dims=[128, 160],
        key_dims=[16, 24], num_heads=4, mlp_ratios=[2, 4],
        head_in_channels=[32, 128, 160],
        head_channels=96, head_embed_dims=[64, 96], is_dw=True,
    ),
    'small': dict(
        cfgs=[
            [[3, 1, 16, 1], [3, 4, 24, 2], [3, 3, 24, 1]],
            [[5, 3, 48, 2], [5, 3, 48, 1]],
            [[3, 3, 96, 2], [3, 3, 96, 1]],
            [[5, 4, 160, 2]],
            [[3, 6, 192, 2]],
        ],
        channels=[16, 24, 48, 96, 160, 192],
        depths=[3, 3], emb_dims=[160, 192],
        key_dims=[16, 24], num_heads=6, mlp_ratios=[2, 4],
        head_in_channels=[48, 160, 192],
        head_channels=128, head_embed_dims=[96, 128], is_dw=True,
    ),
    'base': dict(
        cfgs=[
            [[3, 1, 16, 1], [3, 4, 32, 2], [3, 3, 32, 1]],
            [[5, 3, 64, 2], [5, 3, 64, 1]],
            [[3, 3, 128, 2], [3, 3, 128, 1]],
            [[5, 4, 192, 2]],
            [[3, 6, 256, 2]],
        ],
        channels=[16, 32, 64, 128, 192, 256],
        depths=[4, 4], emb_dims=[192, 256],
        key_dims=[16, 24], num_heads=8, mlp_ratios=[2, 4],
        head_in_channels=[64, 192, 256],
        head_channels=160, head_embed_dims=[128, 160], is_dw=True,
    ),
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


def build_seaformer(cfg):
    model_cfg = cfg['model']
    variant = model_cfg.get('variant', 'tiny')
    vcfg = SEAFORMER_VARIANTS[variant]

    in_ch = cfg['data'].get('num_channels', 4)
    num_classes = model_cfg.get('num_classes', 2)

    model = SeaFormerModel(
        cfgs=vcfg['cfgs'],
        channels=vcfg['channels'],
        emb_dims=vcfg['emb_dims'],
        key_dims=vcfg['key_dims'],
        depths=vcfg['depths'],
        num_heads=vcfg['num_heads'],
        head_channels=vcfg['head_channels'],
        head_embed_dims=vcfg['head_embed_dims'],
        head_in_channels=vcfg['head_in_channels'],
        num_classes=num_classes,
        in_channels=in_ch,
        mlp_ratios=vcfg.get('mlp_ratios'),
        is_dw=vcfg.get('is_dw', True),
    )

    pretrained = model_cfg.get('pretrained_path', '')
    if pretrained and os.path.isfile(pretrained):
        ckpt = torch.load(pretrained, map_location='cpu', weights_only=False)
        sd = ckpt.get('state_dict_ema',
                       ckpt.get('state_dict', ckpt.get('model', ckpt)))
        stem_key = 'smb1.stem_block.0.c.weight'
        if in_ch != 3 and stem_key in sd:
            sd[stem_key] = _adapt_conv_weight(sd[stem_key], in_ch)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"[SeaFormer-{variant}] Loaded pretrained: {pretrained} "
              f"(missing={len(missing)}, unexpected={len(unexpected)})")
    else:
        print(f"[SeaFormer-{variant}] Training from scratch (no pretrained at "
              f"'{pretrained}')")

    return model
