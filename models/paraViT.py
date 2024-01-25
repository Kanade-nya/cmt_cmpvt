"""
red 2, 在pyramid_vit_3 上面的改进
0. num_heads = 1,2,5,8 -> 2,4,8,16
1. 去掉了ViT的四个norm
2. patch-embeding后面的dwconv去掉了 (?)
3. x_1 = x_1 + self.dwConv_2(x) 直接进行残差链接，删除了dwConv
4. parallel conv  第二部分 改为 深度卷积 + 恢复
5. 在block 开始的时候加入了LPU（dw卷积的残差链接。）

-> cmt_pyvit，
small 参数改变
0. stem = 32, heads = 1,2,4,8, depth = 2,4,12,2

-> paraViT :
重新设计
1. patch embeding 改为原版
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


# class OverlapPatchEmbed(nn.Module):
#     """ Image to Patch Embedding
#     """
#
#     def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
#         super().__init__()
#         # imagesize and patchsize 224,patch_size = 7
#         img_size = to_2tuple(img_size)
#         patch_size = to_2tuple(patch_size)
#
#         self.img_size = img_size
#         self.patch_size = patch_size
#         self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
#         self.num_patches = self.H * self.W
#         # 比原版多了这个卷积
#         self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
#                               padding=(patch_size[0] // 2, patch_size[1] // 2))
#
#         self.norm = nn.LayerNorm(embed_dim)
#         self.apply(self._init_weights)
#
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
#     def forward(self, x):
#         x = self.proj(x)
#         _, _, H, W = x.shape
#         x = x.flatten(2).transpose(1, 2)
#         x = self.norm(x)
#
#         return x, H, W


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        # assert img_size[0] % patch_size[0] == 0 and img_size[1] % patch_size[1] == 0, \
        #     f"img_size {img_size} should be divided by patch_size {patch_size}."
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

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
        B, C, H, W = x.shape

        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        H, W = H // self.patch_size[0], W // self.patch_size[1]

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
        self.apply(self._init_weights)

        self.proj = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)
        self.combine = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)

    def forward(self, x, H, W, x_combine):
        # print('---into block---')
        B, N, C = x.shape
        x_LPU = x.permute(0, 2, 1).contiguous().reshape(B, C, H, W)
        x = self.proj(x_LPU) + x_LPU
        x = x.flatten(2).permute(0, 2, 1)
        # print('--before x_combine--: ',x_combine.shape)

        # print('com',x_combine.shape)
        # print('self ',self.combine(x_combine))
        x_combine = x_combine + self.combine(x_combine)
        x_combine = x_combine.permute(0, 2, 3, 1).contiguous().reshape(B, H * W, -1)
        # print('--x_combine.shape--: ', x_combine.shape)
        x = x + self.drop_path(self.attn(self.norm1(x), H, W) + x_combine)
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))

        # print('after mlp:' ,x.shape)
        # print('---end block---')

        return x

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


"""Attention"""


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)
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
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).contiguous().reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1).contiguous()
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class ParallelConv(nn.Module):
    def __init__(self, in_chans, embed_dim, layer=None):
        super().__init__()

        # if layer == 1:
        #     # self.conv = nn.Sequential(  # 池化下采样两次
        #     #     nn.Conv2d(in_chans, in_chans, kernel_size=3, padding=1),
        #     #     nn.BatchNorm2d(in_chans),
        #     #     nn.ReLU(inplace=True),
        #     #
        #     #     nn.MaxPool2d(kernel_size=2, stride=2),
        #     #     nn.ReLU(inplace=True),
        #     #
        #     #     nn.Conv2d(in_chans, in_chans, kernel_size=3, padding=1, groups=in_chans),
        #     #     nn.BatchNorm2d(in_chans),
        #     #     nn.ReLU(inplace=True),
        #     #
        #     #     nn.Conv2d(in_chans, embed_dim, kernel_size=1, stride=1, padding=0),
        #     #     nn.BatchNorm2d(embed_dim),
        #     #     nn.ReLU(inplace=True),
        #     #
        #     #     nn.MaxPool2d(kernel_size=2, stride=2),
        #     #     nn.ReLU(inplace=True),
        #     # )
        #
        #     self.conv = nn.Sequential(
        #         nn.Conv2d(in_channels=in_chans, out_channels=64, kernel_size=7, stride=2, padding=3),
        #         # 112*112 变成1/2
        #         nn.BatchNorm2d(64),
        #         nn.ReLU(inplace=True),
        #         nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1),  # 56*56
        #         nn.BatchNorm2d(64),
        #         nn.ReLU(inplace=True),
        #     )  # endregion
        #
        # else:
        self.conv = nn.Sequential(  # 池化下采样一次
            nn.Conv2d(in_chans, in_chans, kernel_size=3, padding=1, groups=in_chans),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(in_chans, eps=1e-5),

            nn.Conv2d(in_chans, embed_dim, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(embed_dim),

            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True),

            # 多加了一个conv 不直接池化下采样了
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=1, padding=1, groups=in_chans),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(embed_dim),

            # 恢复
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(embed_dim),
        )

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
        # print('--before conv--')
        return self.conv(x)


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
        self.patch_embed_1 = PatchEmbed(img_size=img_size // 2,
                                        patch_size=2,
                                        in_chans=stem_channel,
                                        embed_dim=embed_dims[0])

        self.parallel_1 = ParallelConv(in_chans=stem_channel,
                                       embed_dim=embed_dims[0],
                                       layer=1)

        self.block_1 = nn.ModuleList([Block(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias,
            qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=self.dpr[self.cur + j],
            norm_layer=norm_layer, sr_ratio=sr_ratios[0], layer=1)
            for j in range(depths[0])])
        # self.norm_1 = norm_layer(embed_dims[0])
        # 移动cur
        self.cur += depths[0]

        # --stage 2--
        self.patch_embed_2 = PatchEmbed(img_size=img_size // 4,
                                        patch_size=2,
                                        in_chans=embed_dims[0],
                                        embed_dim=embed_dims[1])

        self.parallel_2 = ParallelConv(in_chans=embed_dims[0],
                                       embed_dim=embed_dims[1],
                                       layer=2)

        # self.dwConv_2 = nn.Conv2d(in_channels=embed_dims[0], out_channels=embed_dims[0], kernel_size=1, padding=0)

        self.block_2 = nn.ModuleList([Block(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias,
            qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=self.dpr[self.cur + j],
            norm_layer=norm_layer, sr_ratio=sr_ratios[1], layer=2)
            for j in range(depths[1])])
        # self.norm_2 = norm_layer(embed_dims[1])
        # 移动cur
        self.cur += depths[1]

        # --stage 3--
        self.patch_embed_3 = PatchEmbed(img_size=img_size // 8,
                                        patch_size=2,
                                        in_chans=embed_dims[1],
                                        embed_dim=embed_dims[2])

        # self.dwConv_3 = nn.Conv2d(in_channels=embed_dims[1], out_channels=embed_dims[1], kernel_size=1, padding=0)

        self.parallel_3 = ParallelConv(in_chans=embed_dims[1],
                                       embed_dim=embed_dims[2],
                                       layer=3)

        self.block_3 = nn.ModuleList([Block(
            dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias,
            qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=self.dpr[self.cur + j],
            norm_layer=norm_layer, sr_ratio=sr_ratios[2], layer=3)
            for j in range(depths[2])])
        # self.norm_3 = norm_layer(embed_dims[2])
        # 移动cur
        self.cur += depths[2]

        # --stage 4--
        self.patch_embed_4 = PatchEmbed(img_size=img_size // 16,
                                        patch_size=2,
                                        in_chans=embed_dims[2],
                                        embed_dim=embed_dims[3])

        # self.dwConv_4 = nn.Conv2d(in_channels=embed_dims[2], out_channels=embed_dims[2], kernel_size=1, padding=0)

        self.parallel_4 = ParallelConv(in_chans=embed_dims[2],
                                       embed_dim=embed_dims[3],
                                       layer=4)

        self.block_4 = nn.ModuleList([Block(
            dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias,
            qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=self.dpr[self.cur + j],
            norm_layer=norm_layer, sr_ratio=sr_ratios[3], layer=4)
            for j in range(depths[3])])
        # self.norm_4 = norm_layer(embed_dims[3])
        # 移动cur
        self.cur += depths[3]

        self.end_conv = nn.Conv2d(embed_dims[3] * 2, embed_dims[3] * 2, kernel_size=3, padding=1)
        self.last_norm = norm_layer(embed_dims[3] * 2)
        # after stage 4
        # self.norm = norm_layer(embed_dims[3])
        # self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dims[3]))
        # classification head
        self.head = nn.Linear(embed_dims[3] * 2, num_classes) if num_classes > 0 else nn.Identity()

        # init weights
        self.apply(self._init_weights)

    def forward_features(self, x):
        B = x.shape[0]
        x = self.conv_stem(x)
        # stage 1
        # print('shape: ', x.shape)
        x_1 = self.parallel_1(x)  # x1 3,224,224 -> 64,56,56
        x, H, W = self.patch_embed_1(x)  # patch embedding # B,3,224,224 -> B,3136,64 H,w = 56

        # x = self.pos_drop_1(x + self.pos_embed_1)  # 加上position embedding 和 drop out , pos 的张量 [1,3136,64] 会自动扩展
        # print('after postion:',x.shape) x.shape B ,3136,64
        # print('after 1')
        # print('after postion:',x_1.shape) #x.shape B ,3136,64
        for blk in self.block_1:
            x = blk(x, H, W, x_1)
        # x = self.norm_1(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        # print('after stage1:', x.shape)

        x_1 = x_1 + x
        x_2 = self.parallel_2(x_1)
        # print('--after parallel2-- shape: ', x_2.shape)
        x, H, W = self.patch_embed_2(x)  # // 2  # B,64,56,56 -> B,128,28,28(784) -> B,784,128
        # x = self.pos_drop_2(x + self.pos_embed_2)

        # stage 2
        # x_1plus = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous() # 28 * 28
        # print('x1plus',x_1plus.shape)
        # print('x1',x_1.shape)

        # x_1 = x_1 + self.dwConv_2(x)
        # x_2 = self.parallel_2(x_1)

        for blk in self.block_2:
            x = blk(x, H, W, x_2)
        # x = self.norm_2(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        # print('after stage2:', x.shape)

        # stage 3
        # x_2plus = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x_2 = x_2 + x
        x_3 = self.parallel_3(x_2)

        x, H, W = self.patch_embed_3(x)  # // 2  # B,128,28,28 -> B,320,14,14 -> B,196,320
        # x = self.pos_drop_3(x + self.pos_embed_3)
        for blk in self.block_3:
            x = blk(x, H, W, x_3)
        # x = self.norm_3(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        # print('after stage3:', x.shape)

        # stage 4
        # x_3plus = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x_3 = x_3 + x
        x_4 = self.parallel_4(x_3)
        # print('x_4.shape', x_4.shape)
        # print('--after parallel4-- shape: ', x_4.shape)
        x, H, W = self.patch_embed_4(x)  # // 2  # B,320,14,14 -> B,512,7,7

        # cls and pos embed
        # cls_tokens = self.cls_token.expand(B, -1, -1)
        # x = torch.cat((cls_tokens, x), dim=1)

        # x = self.pos_drop_4(x + self.pos_embed_4)
        for blk in self.block_4:
            x = blk(x, H, W, x_4)
        # print('x.shape0', x.shape)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        # print('x.shape1', x.shape)
        x = torch.cat((x, x_4), dim=1)
        # print('x.shape2', x.shape)
        x = self.end_conv(x).flatten(2).transpose(1, 2)
        x = self.last_norm(x)
        # print('x.shape3', x.shape)
        # x = self.norm_4(x)
        # print('--after blk4-- shape: ',x.shape)
        # print('after stage4:', x.shape)

        # head
        # x = self.norm(x)
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
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def freeze_patch_emb(self):
        self.patch_embed1.requires_grad = False

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed1', 'pos_embed2', 'pos_embed3', 'pos_embed4', 'cls_token'}  # has pos_embed may be better

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()


# # 224*224*3 模拟图片
# image = torch.randn([64, 3, 224, 224])
#
# model = REPVTModel(patch_size=4, embed_dims=[64, 128, 320, 512], depths=[2, 2, 2, 2], num_heads=[1, 2, 5, 8],
#               mlp_ratios=[8, 8, 4, 4], qkv_bias=True)
#
# out = model(image)
# print("最后输出：", out.shape)


# block: [1,2,4,1], 11.67 2.21 (tiny)
# cmt_pyvit cuda:0 444.14517029038115 images/s @ batch size 128

# block : [2,4,10,2] 22.83   4.06         (small)
# cmt_pyvit cuda:0 247.55713691045648 images/s @ batch size 128


# @register_model
# def cmt_pyvit(pretrained=False, **kwargs):
#     model = REPVTModel(
#         patch_size=4,
#         embed_dims=[64, 128, 256, 512],
#         num_heads=[1, 2, 4, 8],
#         mlp_ratios=[8, 8, 4, 4],
#         qkv_bias=True,
#         norm_layer=partial(nn.LayerNorm, eps=1e-6),
#         depths=[2,4,10,2], # 2,4,10,2
#         sr_ratios=[8, 4, 2, 1],
#         stem_channel=32,
#         **kwargs)
#     model.default_cfg = _cfg()
#
#     return model

@register_model
def paraViT_xxs(pretrained=False, **kwargs):
    model = REPVTModel(
        patch_size=4,
        embed_dims=[32,64,128,256],
        num_heads=[2, 4, 8, 16],
        mlp_ratios=[8, 8, 4, 4],
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        depths=[1, 1, 3, 2],  # 2,4,10,2
        sr_ratios=[8, 4, 2, 1],
        stem_channel=16,
        **kwargs)
    model.default_cfg = _cfg()

    return model
