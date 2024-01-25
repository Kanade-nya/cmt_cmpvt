"""
pvtv2->改进
# MLP 改成 x + dwconv 的残差结构 (哪来的我也不知道)
# Patch-Embedding 改成 Overlap Patch-Embedding (pvtv2)
# Attention 改成加强版的Shunted-Attention
# Conv-stem 将 feature map 下采样到原来的 1/2 (CMT)

stem_channel, dim = 18, [36, 72, 144, 288]
num_heads = [2, 4, 8, 16]
depth = [1, 1, 3, 2]

# GFLOPs 0.59, Params 3.83M
# v2_shunted cuda:0 883.7398542239948 images/s @ batch size 128

----- update -----

-> v4_shunted
stem_channel, dim = 24, [48, 96, 192, 384]
num_heads = [2, 4, 8, 16]
depth = [1, 1, 3, 2]

# GFLOPs 1.01, Params 6.26M
# v4_shunted cuda:0 705.4723713069801 images/s @ batch size 128

another type:
stem_channel, dim = 16, [32, 64, 128, 384]
depth = [1, 2, 4, 1]

# GFLOPs 0.58, Params 3.64M

"""

import math
from functools import partial

import torch
import torch.nn as nn

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x


# 把原始MLP换成DW卷积MLP
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.act(x + self.dwconv(x, H, W))
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# PatchEmbed OverLap
class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, H, W


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1, layer=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        # layer
        self.layer = layer

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        B, _, _ = x.shape

        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))

        return x


"""Attention,按照shunted的方式修改"""


# class Attention(nn.Module):
#     def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
#         super().__init__()
#         assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
#
#         self.dim = dim
#         self.num_heads = num_heads
#         head_dim = dim // num_heads
#         self.scale = qk_scale or head_dim ** -0.5
#
#         self.q = nn.Linear(dim, dim, bias=qkv_bias)
#
#         self.attn_drop = nn.Dropout(attn_drop)
#         self.proj = nn.Linear(dim, dim)
#         self.proj_drop = nn.Dropout(proj_drop)
#
#         self.sr_ratio = sr_ratio
#
#         if sr_ratio > 1:
#             self.act = nn.GELU()
#             if sr_ratio == 8:
#                 self.sr1 = nn.Conv2d(dim, dim, kernel_size=8, stride=8)
#                 self.norm1 = nn.LayerNorm(dim)
#                 self.sr2 = nn.Conv2d(dim, dim, kernel_size=4, stride=4)
#                 self.norm2 = nn.LayerNorm(dim)
#             if sr_ratio == 4:
#                 self.sr1 = nn.Conv2d(dim, dim, kernel_size=4, stride=4)
#                 self.norm1 = nn.LayerNorm(dim)
#                 self.sr2 = nn.Conv2d(dim, dim, kernel_size=2, stride=2)
#                 self.norm2 = nn.LayerNorm(dim)
#             if sr_ratio == 2:
#                 self.sr1 = nn.Conv2d(dim, dim, kernel_size=2, stride=2)
#                 self.norm1 = nn.LayerNorm(dim)
#                 self.sr2 = nn.Conv2d(dim, dim, kernel_size=1, stride=1)
#                 self.norm2 = nn.LayerNorm(dim)
#             self.kv1 = nn.Linear(dim, dim, bias=qkv_bias)
#             self.kv2 = nn.Linear(dim, dim, bias=qkv_bias)
#             self.local_conv1 = nn.Conv2d(dim // 2, dim // 2, kernel_size=3, padding=1, stride=1, groups=dim // 2)
#             self.local_conv2 = nn.Conv2d(dim // 2, dim // 2, kernel_size=3, padding=1, stride=1, groups=dim // 2)
#         else:
#             self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
#             self.local_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, stride=1, groups=dim)
#         self.apply(self._init_weights)
#
#     #
#     def _init_weights(self, m):
#         if isinstance(m, nn.Linear):
#             trunc_normal_(m.weight, std=.02)
#             if isinstance(m, nn.Linear) and m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#         elif isinstance(m, nn.LayerNorm):
#             nn.init.constant_(m.bias, 0)
#             nn.init.constant_(m.weight, 1.0)
#         elif isinstance(m, nn.Conv2d):
#             fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#             fan_out //= m.groups
#             m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
#             if m.bias is not None:
#                 m.bias.data.zero_()
#
#     def forward(self, x, H, W):
#         B, N, C = x.shape
#         # [B,N , 8 , C // 8 ], permute [B,N,8,C//8] -> [B,8,N,C//8],q 的操作是一样的
#         q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
#         if self.sr_ratio > 1:
#             # permute -> [B,C,N] -> [B,C,H,W]
#             x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
#             # act=relu norm = layernorm ,sr1 = ker2,stride2的卷积，把大小变为一半,reshape,再permute->[B,N,C]
#             # print('test1,sr1shape = ',test.shape)  B,C,7,7,
#             # print('test2,sr2shape = ', test2.shape) B,C,14,14
#             x_1 = self.act(self.norm1(self.sr1(x_).reshape(B, C, -1).permute(0, 2, 1)))
#             x_2 = self.act(self.norm2(self.sr2(x_).reshape(B, C, -1).permute(0, 2, 1)))
#             kv1 = self.kv1(x_1).reshape(B, -1, 2, self.num_heads // 2, C // self.num_heads).permute(2, 0, 3, 1, 4)
#             kv2 = self.kv2(x_2).reshape(B, -1, 2, self.num_heads // 2, C // self.num_heads).permute(2, 0, 3, 1, 4)
#             k1, v1 = kv1[0], kv1[1]  # B head N C
#             k2, v2 = kv2[0], kv2[1]
#             attn1 = (q[:, :self.num_heads // 2] @ k1.transpose(-2, -1)) * self.scale
#             attn1 = attn1.softmax(dim=-1)
#             attn1 = self.attn_drop(attn1)
#             v1 = v1 + self.local_conv1(v1.transpose(1, 2).reshape(B, -1, C // 2).
#                                        transpose(1, 2).view(B, C // 2, H // self.sr_ratio, W // self.sr_ratio)). \
#                 view(B, C // 2, -1).view(B, self.num_heads // 2, C // self.num_heads, -1).transpose(-1, -2)
#             x1 = (attn1 @ v1).transpose(1, 2).reshape(B, N, C // 2)
#             attn2 = (q[:, self.num_heads // 2:] @ k2.transpose(-2, -1)) * self.scale
#             attn2 = attn2.softmax(dim=-1)
#             attn2 = self.attn_drop(attn2)
#             v2 = v2 + self.local_conv2(v2.transpose(1, 2).reshape(B, -1, C // 2).
#                                        transpose(1, 2).view(B, C // 2, H * 2 // self.sr_ratio, W * 2 // self.sr_ratio)). \
#                 view(B, C // 2, -1).view(B, self.num_heads // 2, C // self.num_heads, -1).transpose(-1, -2)
#             x2 = (attn2 @ v2).transpose(1, 2).reshape(B, N, C // 2)
#             x = torch.cat([x1, x2], dim=-1)
#         else:
#             kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
#             k, v = kv[0], kv[1]
#
#             attn = (q @ k.transpose(-2, -1)) * self.scale
#             attn = attn.softmax(dim=-1)
#             attn = self.attn_drop(attn)
#
#             x = (attn @ v).transpose(1, 2).reshape(B, N, C) + self.local_conv(v.transpose(1, 2).reshape(B, N, C).
#                                                                               transpose(1, 2).view(B, C, H, W)).view(B,
#                                                                                                                      C,
#                                                                                                                      N).transpose(
#                 1, 2)
#         x = self.proj(x)
#         x = self.proj_drop(x)
#
#         return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.origin_dim = dim
        self.sr_ratio = sr_ratio

        self.dim = self.origin_dim // 4 * 3
        self.conv_part_dim = self.origin_dim // 4

        # conv_part
        if self.sr_ratio == 8 or self.sr_ratio == 4:
            self.conv_part_2d = nn.Conv2d(in_channels=self.conv_part_dim,
                                          out_channels=self.conv_part_dim,
                                          kernel_size=5,
                                          padding=2)

        elif sr_ratio == 2 or sr_ratio == 1:
            self.conv_part_2d = nn.Conv2d(in_channels=self.conv_part_dim,
                                          out_channels=self.conv_part_dim,
                                          kernel_size=3,
                                          padding=1)
        self.convGELU = nn.GELU()
        self.convNorm = nn.LayerNorm(self.conv_part_dim)
        # 不变的几个dim
        self.proj = nn.Linear(self.origin_dim, self.origin_dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # attn part
        self.num_heads = num_heads
        head_dim = self.dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # 现在开始所有的dim改变

        self.q = nn.Linear(self.dim, self.dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)

        if sr_ratio > 1:
            self.act = nn.GELU()
            if sr_ratio == 8:
                self.sr1 = nn.Conv2d(self.dim, self.dim, kernel_size=8, stride=8, groups=self.dim)
                self.norm1 = nn.LayerNorm(self.dim)
                self.sr2 = nn.Conv2d(self.dim, self.dim, kernel_size=4, stride=4, groups=self.dim)
                self.norm2 = nn.LayerNorm(self.dim)
                # self.sr3 = nn.Conv2d(dim, dim, kernel_size=2, stride=2, groups=dim)
                # self.norm3 = nn.LayerNorm(dim)
                # self.sr4 = nn.Conv2d(dim, dim, kernel_size=1, stride=1, groups=dim)
                # self.norm4 = nn.LayerNorm(dim)

                self.kv1 = nn.Linear(self.dim, self.dim, bias=qkv_bias)
                self.kv2 = nn.Linear(self.dim, self.dim, bias=qkv_bias)
                # self.kv3 = nn.Linear(dim, dim // 2, bias=qkv_bias)
                # self.kv4 = nn.Linear(dim, dim // 2, bias=qkv_bias)
            if sr_ratio == 4:
                self.sr1 = nn.Conv2d(self.dim, self.dim, kernel_size=4, stride=4, groups=self.dim)
                self.norm1 = nn.LayerNorm(self.dim)
                self.sr2 = nn.Conv2d(self.dim, self.dim, kernel_size=2, stride=2, groups=self.dim)
                self.norm2 = nn.LayerNorm(self.dim)
                self.sr3 = nn.Conv2d(self.dim, self.dim, kernel_size=1, stride=1, groups=self.dim)
                self.norm3 = nn.LayerNorm(self.dim)
                self.sr4 = nn.Conv2d(self.dim, self.dim // 4, kernel_size=3, stride=1, padding=1)
                self.norm4 = nn.LayerNorm(self.dim // 4)

                self.kv1 = nn.Linear(self.dim, self.dim // 2, bias=qkv_bias)
                self.kv2 = nn.Linear(self.dim, self.dim // 2, bias=qkv_bias)
                self.kv3 = nn.Linear(self.dim, self.dim // 2, bias=qkv_bias)
                # self.kv4 = nn.Linear(dim, dim // 2, bias=qkv_bias)
            if sr_ratio == 2:
                self.sr1 = nn.Conv2d(self.dim, self.dim, kernel_size=2, stride=2, groups=self.dim)
                self.norm1 = nn.LayerNorm(self.dim)
                self.sr2 = nn.Conv2d(self.dim, self.dim, kernel_size=1, stride=1, groups=self.dim)
                self.norm2 = nn.LayerNorm(self.dim)
                self.sr3 = nn.Conv2d(self.dim, self.dim // 4, kernel_size=3, stride=1, padding=1)
                self.norm3 = nn.LayerNorm(self.dim // 4)
                self.sr4 = nn.Conv2d(self.dim, self.dim // 4, kernel_size=3, stride=1, padding=1, groups=self.dim // 4)
                self.norm4 = nn.LayerNorm(self.dim // 4)

                self.kv1 = nn.Linear(self.dim, self.dim // 2, bias=qkv_bias)
                self.kv2 = nn.Linear(self.dim, self.dim // 2, bias=qkv_bias)
                # self.kv3 = nn.Linear(dim, dim // 2, bias=qkv_bias)
                # self.kv4 = nn.Linear(dim, dim // 2, bias=qkv_bias)
            # self.kv1 = nn.Linear(dim, dim, bias=qkv_bias)
            # self.kv2 = nn.Linear(dim, dim, bias=qkv_bias)
            # self.local_conv1 = nn.Conv2d(dim // 2, dim // 2, kernel_size=3, padding=1, stride=1, groups=dim // 2)
            # self.local_conv2 = nn.Conv2d(dim // 2, dim // 2, kernel_size=3, padding=1, stride=1, groups=dim // 2)
        else:
            self.kv = nn.Linear(self.dim, self.dim * 2, bias=qkv_bias)
            # local conv 全部删掉
            # self.local_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, stride=1, groups=dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H=56, W=56):
        B, N, C = x.shape
        # print('x.shape: ', x.shape)  # 1,3636,36
        x_conv_part, x_attn_part = torch.split(x, [self.conv_part_dim, self.dim], dim=2)
        # print('part_shape: ', x_conv_part.shape, x_attn_part.shape)
        x_conv_part = x_conv_part.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        # print('conv_shape: ', x_conv_part.shape)
        x_conv_part = self.convGELU(
            self.convNorm(self.conv_part_2d(x_conv_part).reshape(B, self.conv_part_dim, -1).permute(0, 2, 1)))
        # print('after.shape', x_conv_part.shape)

        # 改变C 为原来的 3 / 4
        _, _, C = x_attn_part.shape  # 27  48，96，192，384   12，36
        q = self.q(x_attn_part).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        # print('?')  # pass
        if self.sr_ratio > 1:
            x_ = x_attn_part.permute(0, 2, 1).reshape(B, C, H, W)
            if self.sr_ratio == 8:
                x_1 = self.act(self.norm1(self.sr1(x_).reshape(B, C, -1).permute(0, 2, 1)))
                x_2 = self.act(self.norm2(self.sr2(x_).reshape(B, C, -1).permute(0, 2, 1)))

                kv1 = self.kv1(x_1).reshape(B, -1, 2, self.num_heads // 2, C // self.num_heads).permute(2, 0, 3, 1, 4)
                kv2 = self.kv2(x_2).reshape(B, -1, 2, self.num_heads // 2, C // self.num_heads).permute(2, 0, 3, 1, 4)

                k1, v1 = kv1[0], kv1[1]  # B head N C
                k2, v2 = kv2[0], kv2[1]

                attn1 = (q[:, :self.num_heads // 2] @ k1.transpose(-2, -1)) * self.scale
                attn1 = attn1.softmax(dim=-1)
                attn1 = self.attn_drop(attn1)
                x1 = (attn1 @ v1).transpose(1, 2).reshape(B, N, C // 2)
                # v1 = v1 + self.local_conv1(v1.transpose(1, 2).reshape(B, -1, C // 2).
                #                            transpose(1, 2).view(B, C // 2, H // self.sr_ratio, W // self.sr_ratio)). \
                #     view(B, C // 2, -1).view(B, self.num_heads // 2, C // self.num_heads, -1).transpose(-1, -2)

                # attn2:
                attn2 = (q[:, self.num_heads // 2:] @ k2.transpose(-2, -1)) * self.scale
                attn2 = attn2.softmax(dim=-1)
                attn2 = self.attn_drop(attn2)
                x2 = (attn2 @ v2).transpose(1, 2).reshape(B, N, C // 2)
                # v2 = v2 + self.local_conv2(v2.transpose(1, 2).reshape(B, -1, C // 2). transpose(1, 2).view(B, C // 2,
                # H * 2 // self.sr_ratio, W * 2 // self.sr_ratio)). \ view(B, C // 2, -1).view(B, self.num_heads // 2,
                # C // self.num_heads, -1).transpose(-1, -2)

                # print('x1.shape', x1.shape)
                # print('x2.shape', x2.shape)

                # # concat
                x = torch.cat([x1, x2], dim=-1)
                # print('x.shape', x.shape)

            elif self.sr_ratio == 4:
                x_1 = self.act(self.norm1(self.sr1(x_).reshape(B, C, -1).permute(0, 2, 1)))
                x_2 = self.act(self.norm2(self.sr2(x_).reshape(B, C, -1).permute(0, 2, 1)))
                x_3 = self.act(self.norm3(self.sr3(x_).reshape(B, C, -1).permute(0, 2, 1)))
                x_4 = self.act(self.norm4(self.sr4(x_).reshape(B, C // 4, -1).permute(0, 2, 1)))
                # print('x1.shape: ', x_1.shape)  # 56*56 , 64
                # print('x_2.shape: ', x_2.shape)  # 56*56 , 64
                # print('x_3.shape: ', x_3.shape)  # 56*56 , 64
                # print('x_4.shape: ', x_4.shape)  # 56 * 56 16

                kv1 = self.kv1(x_1).reshape(B, -1, 2, self.num_heads // 4, C // self.num_heads).permute(2, 0, 3, 1, 4)
                kv2 = self.kv2(x_2).reshape(B, -1, 2, self.num_heads // 4, C // self.num_heads).permute(2, 0, 3, 1, 4)
                kv3 = self.kv3(x_3).reshape(B, -1, 2, self.num_heads // 4, C // self.num_heads).permute(2, 0, 3, 1, 4)
                # kv4 = self.kv4(x_4).reshape(B, -1, 2, self.num_heads // 4, C // self.num_heads).permute(2, 0, 3, 1, 4)

                k1, v1 = kv1[0], kv1[1]  # B head N C
                k2, v2 = kv2[0], kv2[1]
                k3, v3 = kv3[0], kv3[1]  # B head N C
                # k4, v4 = kv4[0], kv4[1]
                # attn1:
                attn1 = (q[:, :self.num_heads // 4] @ k1.transpose(-2, -1)) * self.scale
                attn1 = attn1.softmax(dim=-1)
                attn1 = self.attn_drop(attn1)
                x1 = (attn1 @ v1).transpose(1, 2).reshape(B, N, C // 4)
                # v1 = v1 + self.local_conv1(v1.transpose(1, 2).reshape(B, -1, C // 2).
                #                            transpose(1, 2).view(B, C // 2, H // self.sr_ratio, W // self.sr_ratio)). \
                #     view(B, C // 2, -1).view(B, self.num_heads // 2, C // self.num_heads, -1).transpose(-1, -2)

                # attn2:
                attn2 = (q[:, self.num_heads // 4: self.num_heads // 2] @ k2.transpose(-2, -1)) * self.scale
                attn2 = attn2.softmax(dim=-1)
                attn2 = self.attn_drop(attn2)
                x2 = (attn2 @ v2).transpose(1, 2).reshape(B, N, C // 4)
                # v2 = v2 + self.local_conv2(v2.transpose(1, 2).reshape(B, -1, C // 2). transpose(1, 2).view(B, C // 2,
                # H * 2 // self.sr_ratio, W * 2 // self.sr_ratio)). \ view(B, C // 2, -1).view(B, self.num_heads // 2,
                # C // self.num_heads, -1).transpose(-1, -2)

                # attn3:
                attn3 = (q[:, self.num_heads // 2: self.num_heads // 4 * 3] @ k3.transpose(-2, -1)) * self.scale
                attn3 = attn3.softmax(dim=-1)
                attn3 = self.attn_drop(attn3)
                x3 = (attn3 @ v3).transpose(1, 2).reshape(B, N, C // 4)

                # attn4:
                # attn4 = (q[:, self.num_heads // 4 * 3: ] @ k4.transpose(-2, -1)) * self.scale
                # attn4 = attn4.softmax(dim=-1)
                # attn4 = self.attn_drop(attn4)
                # x4 = (attn4 @ v4).transpose(1, 2).reshape(B, N, C // 4)
                # print('x4.shape',x4.shape)
                # # concat
                x = torch.cat([x1, x2, x3, x_4], dim=-1)
                # print('x.shape', x.shape)

            elif self.sr_ratio == 2:
                x_1 = self.act(self.norm1(self.sr1(x_).reshape(B, C, -1).permute(0, 2, 1)))
                x_2 = self.act(self.norm2(self.sr2(x_).reshape(B, C, -1).permute(0, 2, 1)))
                x_3 = self.act(self.norm3(self.sr3(x_).reshape(B, C // 4, -1).permute(0, 2, 1)))
                x_4 = self.act(self.norm4(self.sr4(x_).reshape(B, C // 4, -1).permute(0, 2, 1)))
                # print('x1.shape: ', x_1.shape)  # 56*56 , 64
                # print('x_2.shape: ', x_2.shape)  # 56*56 , 64
                # print('x_3.shape: ', x_3.shape)  # 56*56 , 64
                # print('x_4.shape: ', x_4.shape)  # 56 * 56 16

                kv1 = self.kv1(x_1).reshape(B, -1, 2, self.num_heads // 4, C // self.num_heads).permute(2, 0, 3, 1, 4)
                kv2 = self.kv2(x_2).reshape(B, -1, 2, self.num_heads // 4, C // self.num_heads).permute(2, 0, 3, 1, 4)
                # kv3 = self.kv3(x_3).reshape(B, -1, 2, self.num_heads // 4, C // self.num_heads).permute(2, 0, 3, 1, 4)
                # kv4 = self.kv4(x_4).reshape(B, -1, 2, self.num_heads // 4, C // self.num_heads).permute(2, 0, 3, 1, 4)

                k1, v1 = kv1[0], kv1[1]  # B head N C
                k2, v2 = kv2[0], kv2[1]
                # k3, v3 = kv3[0], kv3[1]  # B head N C
                # k4, v4 = kv4[0], kv4[1]
                # attn1:
                attn1 = (q[:, :self.num_heads // 4] @ k1.transpose(-2, -1)) * self.scale
                attn1 = attn1.softmax(dim=-1)
                attn1 = self.attn_drop(attn1)
                x1 = (attn1 @ v1).transpose(1, 2).reshape(B, N, C // 4)
                # v1 = v1 + self.local_conv1(v1.transpose(1, 2).reshape(B, -1, C // 2).
                #                            transpose(1, 2).view(B, C // 2, H // self.sr_ratio, W // self.sr_ratio)). \
                #     view(B, C // 2, -1).view(B, self.num_heads // 2, C // self.num_heads, -1).transpose(-1, -2)

                # attn2:
                attn2 = (q[:, self.num_heads // 4: self.num_heads // 2] @ k2.transpose(-2, -1)) * self.scale
                attn2 = attn2.softmax(dim=-1)
                attn2 = self.attn_drop(attn2)
                x2 = (attn2 @ v2).transpose(1, 2).reshape(B, N, C // 4)
                # v2 = v2 + self.local_conv2(v2.transpose(1, 2).reshape(B, -1, C // 2). transpose(1, 2).view(B, C // 2,
                # H * 2 // self.sr_ratio, W * 2 // self.sr_ratio)). \ view(B, C // 2, -1).view(B, self.num_heads // 2,
                # C // self.num_heads, -1).transpose(-1, -2)

                # attn3:
                # attn3 = (q[:, self.num_heads // 2: self.num_heads // 4 * 3] @ k3.transpose(-2, -1)) * self.scale
                # attn3 = attn3.softmax(dim=-1)
                # attn3 = self.attn_drop(attn3)
                # x3 = (attn3 @ v3).transpose(1, 2).reshape(B, N, C // 4)

                # attn4:
                # attn4 = (q[:, self.num_heads // 4 * 3: ] @ k4.transpose(-2, -1)) * self.scale
                # attn4 = attn4.softmax(dim=-1)
                # attn4 = self.attn_drop(attn4)
                # x4 = (attn4 @ v4).transpose(1, 2).reshape(B, N, C // 4)
                # print('x4.shape',x4.shape)
                # # concat
                x = torch.cat([x1, x2, x_3, x_4], dim=-1)
                # print('x.shape', x.shape)
            else:
                x_1 = self.act(self.norm1(self.sr1(x_).reshape(B, C, -1).permute(0, 2, 1)))
                x_2 = self.act(self.norm2(self.sr2(x_).reshape(B, C, -1).permute(0, 2, 1)))
                kv1 = self.kv1(x_1).reshape(B, -1, 2, self.num_heads // 2, C // self.num_heads).permute(2, 0, 3, 1, 4)
                kv2 = self.kv2(x_2).reshape(B, -1, 2, self.num_heads // 2, C // self.num_heads).permute(2, 0, 3, 1, 4)
                k1, v1 = kv1[0], kv1[1]  # B head N C
                k2, v2 = kv2[0], kv2[1]
                # attn1:
                attn1 = (q[:, :self.num_heads // 2] @ k1.transpose(-2, -1)) * self.scale
                attn1 = attn1.softmax(dim=-1)
                attn1 = self.attn_drop(attn1)
                x1 = (attn1 @ v1).transpose(1, 2).reshape(B, N, C // 2)
                # v1 = v1 + self.local_conv1(v1.transpose(1, 2).reshape(B, -1, C // 2).
                #                            transpose(1, 2).view(B, C // 2, H // self.sr_ratio, W // self.sr_ratio)). \
                #     view(B, C // 2, -1).view(B, self.num_heads // 2, C // self.num_heads, -1).transpose(-1, -2)

                # attn2:
                attn2 = (q[:, self.num_heads // 2:] @ k2.transpose(-2, -1)) * self.scale
                attn2 = attn2.softmax(dim=-1)
                attn2 = self.attn_drop(attn2)
                x2 = (attn2 @ v2).transpose(1, 2).reshape(B, N, C // 2)
                # v2 = v2 + self.local_conv2(v2.transpose(1, 2).reshape(B, -1, C // 2). transpose(1, 2).view(B, C // 2,
                # H * 2 // self.sr_ratio, W * 2 // self.sr_ratio)). \ view(B, C // 2, -1).view(B, self.num_heads // 2,
                # C // self.num_heads, -1).transpose(-1, -2)

                # concat
                x = torch.cat([x1, x2], dim=-1)
        else:
            kv = self.kv(x_attn_part).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            k, v = kv[0], kv[1]

            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
            # x = (attn @ v).transpose(1, 2).reshape(B, N, C) + self.local_conv(v.transpose(1, 2).reshape(B, N,
            # C). transpose(1, 2).view(B, C, H, W)).view(B, C, N).transpose( 1, 2)
            # print('now,dim', x.shape)
        x = torch.cat([x, x_conv_part], dim=-1)
        # print('now,dim', x.shape)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class REPVTModel(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, num_classes=1000, embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None,
                 depths=[3, 4, 6, 3], num_stages=4, drop_path_rate=0., drop_rate=0., attn_drop_rate=0.,
                 norm_layer=nn.LayerNorm, sr_ratios=[8, 4, 2, 1], stem_channel=16):
        super().__init__()
        self.num_classes = num_classes  # 分类数量
        self.depths = depths  # 每层的block块数
        self.num_stages = num_stages  # 一共4个阶段
        # dpr,cur负责drop_path的
        self.dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        self.cur = 0

        # --conv-stem--
        self.conv_stem = nn.Sequential(
            # conv1 3,224,224 -> stem_c,112,112
            nn.Conv2d(3, stem_channel, kernel_size=3, stride=2, padding=1, bias=True),
            nn.GELU(),
            nn.BatchNorm2d(stem_channel),
            # conv2
            nn.Conv2d(stem_channel, stem_channel, kernel_size=3, stride=1, padding=1, bias=True),
            nn.GELU(),
            nn.BatchNorm2d(stem_channel),
            # conv3
            nn.Conv2d(stem_channel, stem_channel, kernel_size=3, stride=1, padding=1, bias=True),
            nn.GELU(),
            nn.BatchNorm2d(stem_channel),
        )
        # --stage 1--
        # self.patch_embed_1 = PatchEmbed(img_size=img_size,
        #                                 patch_size=patch_size,
        #                                 in_chans=in_channels,
        #                                 embed_dim=embed_dims[0])
        self.patch_embed_1 = OverlapPatchEmbed(img_size=img_size // 2,
                                               patch_size=3,
                                               stride=2,
                                               in_chans=stem_channel,
                                               embed_dim=embed_dims[0])
        # self.num_patches_1 = self.patch_embed_1.num_patches
        # self.pos_embed_1 = nn.Parameter(torch.zeros(1, self.num_patches_1, embed_dims[0]))
        # self.pos_drop_1 = nn.Dropout(p=drop_rate)

        self.block_1 = nn.ModuleList([Block(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias,
            qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=self.dpr[self.cur + j],
            norm_layer=norm_layer, sr_ratio=sr_ratios[0], layer=1)
            for j in range(depths[0])])
        # 移动cur
        self.cur += depths[0]

        # --stage 2--
        # self.patch_embed_2 = PatchEmbed(img_size=img_size // 4,
        #                                 patch_size=2,
        #                                 in_chans=embed_dims[0],
        #                                 embed_dim=embed_dims[1])
        self.patch_embed_2 = OverlapPatchEmbed(img_size=img_size // 4,
                                               patch_size=3,
                                               stride=2,
                                               in_chans=embed_dims[0],
                                               embed_dim=embed_dims[1])
        # self.num_patches_2 = self.patch_embed_2.num_patches
        # self.pos_embed_2 = nn.Parameter(torch.zeros(1, self.num_patches_2, embed_dims[1]))
        # self.pos_drop_2 = nn.Dropout(p=drop_rate)

        self.block_2 = nn.ModuleList([Block(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias,
            qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=self.dpr[self.cur + j],
            norm_layer=norm_layer, sr_ratio=sr_ratios[1], layer=2)
            for j in range(depths[1])])
        # 移动cur
        self.cur += depths[1]

        # --stage 3--
        # self.patch_embed_3 = PatchEmbed(img_size=img_size // 8,
        #                                 patch_size=2,
        #                                 in_chans=embed_dims[1],
        #                                 embed_dim=embed_dims[2])
        self.patch_embed_3 = OverlapPatchEmbed(img_size=img_size // 8,
                                               patch_size=3,
                                               stride=2,
                                               in_chans=embed_dims[1],
                                               embed_dim=embed_dims[2])

        # self.num_patches_3 = self.patch_embed_3.num_patches
        # self.pos_embed_3 = nn.Parameter(torch.zeros(1, self.num_patches_3, embed_dims[2]))
        # self.pos_drop_3 = nn.Dropout(p=drop_rate)

        self.block_3 = nn.ModuleList([Block(
            dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias,
            qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=self.dpr[self.cur + j],
            norm_layer=norm_layer, sr_ratio=sr_ratios[2], layer=3)
            for j in range(depths[2])])
        # 移动cur
        self.cur += depths[2]

        # --stage 4--
        # self.patch_embed_4 = PatchEmbed(img_size=img_size // 16,
        #                                 patch_size=2,
        #                                 in_chans=embed_dims[2],
        #                                 embed_dim=embed_dims[3])
        self.patch_embed_4 = OverlapPatchEmbed(img_size=img_size // 16,
                                               patch_size=3,
                                               stride=2,
                                               in_chans=embed_dims[2],
                                               embed_dim=embed_dims[3])

        # self.num_patches_4 = self.patch_embed_4.num_patches + 1
        # self.pos_embed_4 = nn.Parameter(torch.zeros(1, self.num_patches_4, embed_dims[3]))
        # self.pos_drop_4 = nn.Dropout(p=drop_rate)

        self.block_4 = nn.ModuleList([Block(
            dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias,
            qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=self.dpr[self.cur + j],
            norm_layer=norm_layer, sr_ratio=sr_ratios[3], layer=4)
            for j in range(depths[3])])
        # 移动cur
        self.cur += depths[3]

        # after stage 4
        self.last_norm = norm_layer(embed_dims[3])
        # self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dims[3]))
        # classification head
        self.head = nn.Linear(embed_dims[3], num_classes) if num_classes > 0 else nn.Identity()

        # init weights
        # trunc_normal_(self.pos_embed_1, std=.02)
        # trunc_normal_(self.pos_embed_2, std=.02)
        # trunc_normal_(self.pos_embed_3, std=.02)
        # trunc_normal_(self.pos_embed_4, std=.02)
        # trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def forward_features(self, x):
        B = x.shape[0]
        x = self.conv_stem(x)
        # stage 1
        x, H, W = self.patch_embed_1(x)  # patch embedding # B,3,224,224 -> B,3136,64 H,w = 56
        # x = self.pos_drop_1(x + self.pos_embed_1)  # 加上position embedding 和 drop out , pos 的张量 [1,3136,64] 会自动扩展
        # print('after postion:',x.shape) x.shape B ,3136,64
        for blk in self.block_1:
            x = blk(x, H, W)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        # print('after stage1:', x.shape)

        # stage 2
        x, H, W = self.patch_embed_2(x)  # // 2  # B,64,56,56 -> B,128,28,28(784) -> B,784,128
        # x = self.pos_drop_2(x + self.pos_embed_2)
        for blk in self.block_2:
            x = blk(x, H, W)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        # print('after stage2:', x.shape)

        # stage 3
        x, H, W = self.patch_embed_3(x)  # // 2  # B,128,28,28 -> B,320,14,14 -> B,196,320
        # x = self.pos_drop_3(x + self.pos_embed_3)
        for blk in self.block_3:
            x = blk(x, H, W)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        # print('after stage3:', x.shape)

        # stage 4
        x, H, W = self.patch_embed_4(x)  # // 2  # B,320,14,14 -> B,512,7,7

        # cls and pos embed
        # cls_tokens = self.cls_token.expand(B, -1, -1)
        # x = torch.cat((cls_tokens, x), dim=1)

        # x = self.pos_drop_4(x + self.pos_embed_4)
        for blk in self.block_4:
            x = blk(x, H, W)
        # print('after stage4:', x.shape)

        # head
        x = self.last_norm(x)
        return x.mean(dim=1)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        # print('last x :', x.shape)
        return x

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


# Computational complexity:       594.05 MMac
# Number of parameters:           3.83 M
# v2_shunted cuda:0 883.7398542239948 images/s @ batch size 128


@register_model
def v4_shunted(pretrained=False, **kwargs):
    # sr_ratios = 【8，4，2，1】 num_heads = 【1，2，5，8】
    model = REPVTModel(
        patch_size=4,
        embed_dims=[48, 96, 192, 384],
        # embed_dims = [64,128,256,512],
        num_heads=[2, 4, 8, 16],
        mlp_ratios=[8, 8, 4, 4],
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        depths=[1,1,3,2],
        sr_ratios=[8, 4, 2, 1],
        stem_channel=24,
        **kwargs)
    model.default_cfg = _cfg()

    return model

# 256 + 128 = 384
@register_model
def v4_shunted_xxs(pretrained=False, **kwargs):
    # sr_ratios = 【8，4，2，1】 num_heads = 【1，2，5，8】
    model = REPVTModel(
        patch_size=4,
        embed_dims=[32, 64, 128, 384],
        # embed_dims = [64,128,256,512],
        num_heads=[2, 4, 8, 16],
        mlp_ratios=[8, 8, 4, 4],
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        depths=[1,2,4,1],
        sr_ratios=[8, 4, 2, 1],
        stem_channel=16,
        **kwargs)
    model.default_cfg = _cfg()

    return model


