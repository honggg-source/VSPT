import logging
from functools import partial

import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from .attention import Cross_Attention, FocusedLinearCross_Attention_toV, FocusedLinearCross_Attention_toC
from timm.models.layers import DropPath
import torch.utils.checkpoint as cp
from .kernel_warehouse import Warehouse_Manager
import matplotlib.pyplot as plt

_logger = logging.getLogger(__name__)


# 2024/6/25
class MRM(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.conv = MultiDWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.conv(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# SE模块
class SEModule(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        x = self.avg_pool(input)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return input * x


class MultiDWConv(nn.Module):
    def __init__(self, dim=768, scales=4, use_se=True, norm_layer=True):
        super().__init__()
        dim1 = dim
        dim = dim // scales
        self.scales = scales
        if norm_layer:  # BN层
            norm_layer = nn.BatchNorm2d
        self.dwconv1 = nn.ModuleList([nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim) for _ in
                                      range(scales - 1)])
        self.bn1 = nn.ModuleList([norm_layer(dim) for _ in range(scales - 1)])

        self.dwconv2 = nn.ModuleList([nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim) for _ in
                                      range(scales - 1)])
        self.bn2 = nn.ModuleList([norm_layer(dim) for _ in range(scales - 1)])

        self.dwconv3 = nn.ModuleList([nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim) for _ in
                                      range(scales - 1)])
        self.bn3 = nn.ModuleList([norm_layer(dim) for _ in range(scales - 1)])

        self.act1 = nn.GELU()
        self.act2 = nn.GELU()
        self.act3 = nn.GELU()

        # if use_se:
        #     self.se = SEModule(dim1)

    def forward(self, x, H_z, W_z, H_x, W_x):
        z_split = x[:, :336, :].contiguous()
        x_split = x[:, 336:, :].contiguous()
        B_z, N_z, C_z = z_split.shape
        n_z = N_z // 21
        z_split1 = z_split[:, 0:16 * n_z, :].transpose(1, 2).view(B_z, C_z, H_z * 2, W_z * 2).contiguous()
        z_split2 = z_split[:, 16 * n_z:20 * n_z, :].transpose(1, 2).view(B_z, C_z, H_z, W_z).contiguous()
        z_split3 = z_split[:, 20 * n_z:, :].transpose(1, 2).view(B_z, C_z, H_z // 2, W_z // 2).contiguous()
        z_split1_xs = torch.chunk(z_split1, self.scales, 1)  # 将x分割成scales块
        z_split1_ys = []
        for s in range(self.scales):
            if s == 0:
                z_split1_ys.append(z_split1_xs[s])
            elif s == 1:
                z_split1_ys.append(self.act1(self.bn1[s - 1](self.dwconv1[s - 1](z_split1_xs[s]))))
            else:
                z_split1_ys.append(self.act1(self.bn1[s - 1](self.dwconv1[s - 1](z_split1_xs[s] + z_split1_ys[-1]))))
        z_split1 = torch.cat(z_split1_ys, dim=1)
        # 加入SE模块
        # if self.se is not None:
        #     z_split1 = self.se(z_split1)
        z_split1 = z_split1.flatten(2).transpose(1, 2)

        z_split2_xs = torch.chunk(z_split2, self.scales, 1)  # 将x分割成scales块
        z_split2_ys = []
        for s in range(self.scales):
            if s == 0:
                z_split2_ys.append(z_split2_xs[s])
            elif s == 1:
                z_split2_ys.append(self.act2(self.bn2[s - 1](self.dwconv2[s - 1](z_split2_xs[s]))))
            else:
                z_split2_ys.append(self.act2(self.bn2[s - 1](self.dwconv2[s - 1](z_split2_xs[s] + z_split2_ys[-1]))))
        z_split2 = torch.cat(z_split2_ys, dim=1)
        # 加入SE模块
        # if self.se is not None:
        #     z_split2 = self.se(z_split2)
        z_split2 = z_split2.flatten(2).transpose(1, 2)

        z_split3_xs = torch.chunk(z_split3, self.scales, 1)  # 将x分割成scales块
        z_split3_ys = []
        for s in range(self.scales):
            if s == 0:
                z_split3_ys.append(z_split3_xs[s])
            elif s == 1:
                z_split3_ys.append(self.act3(self.bn3[s - 1](self.dwconv3[s - 1](z_split3_xs[s]))))
            else:
                z_split3_ys.append(self.act3(self.bn3[s - 1](self.dwconv3[s - 1](z_split3_xs[s] + z_split3_ys[-1]))))
        z_split3 = torch.cat(z_split3_ys, dim=1)
        # 加入SE模块
        # if self.se is not None:
        #     z_split3 = self.se(z_split3)
        z_split3 = z_split3.flatten(2).transpose(1, 2)
        z_split = torch.cat([z_split1, z_split2, z_split3], dim=1)

        # z_split11, z_split12 = z_split1[:, :C_z // 2, :, :],z_split1[:, C_z // 2:, :, :]
        # z_split11 = self.dwconv1(z_split11)
        # z_split12 = self.dwconv2(z_split12)
        # z_split1 = torch.cat([z_split11, z_split12], dim=1)
        # z_split1 = self.act1(self.bn1(z_split1)).flatten(2).transpose(1, 2)
        #
        # z_split21, z_split22 = z_split2[:, :C_z // 2, :, :], z_split2[:, C_z // 2:, :, :]
        # z_split21 = self.dwconv3(z_split21)
        # z_split22 = self.dwconv4(z_split22)
        # z_split2 = torch.cat([z_split21, z_split22], dim=1)
        # z_split2 = self.act2(self.bn2(z_split2)).flatten(2).transpose(1, 2)
        #
        # z_split31, z_split32 = z_split3[:, :C_z // 2, :, :], z_split3[:, C_z // 2:, :, :]
        # z_split31 = self.dwconv5(z_split31)
        # z_split32 = self.dwconv6(z_split32)
        # z_split3 = torch.cat([z_split31, z_split32], dim=1)
        # z_split3 = self.act3(self.bn3(z_split3)).flatten(2).transpose(1, 2)
        # z_split = torch.cat([z_split1, z_split2, z_split3], dim=1)

        B_x, N_x, C_x = x_split.shape
        n_x = N_x // 21
        x_split1 = x_split[:, 0:16 * n_x, :].transpose(1, 2).view(B_x, C_x, H_x * 2, W_x * 2).contiguous()
        x_split2 = x_split[:, 16 * n_x:20 * n_x, :].transpose(1, 2).view(B_x, C_x, H_x, W_x).contiguous()
        x_split3 = x_split[:, 20 * n_x:, :].transpose(1, 2).view(B_x, C_x, H_x // 2, W_x // 2).contiguous()

        x_split1_xs = torch.chunk(x_split1, self.scales, 1)  # 将x分割成scales块
        x_split1_ys = []
        for s in range(self.scales):
            if s == 0:
                x_split1_ys.append(x_split1_xs[s])
            elif s == 1:
                x_split1_ys.append(self.act1(self.bn1[s - 1](self.dwconv1[s - 1](x_split1_xs[s]))))
            else:
                x_split1_ys.append(self.act1(self.bn1[s - 1](self.dwconv1[s - 1](x_split1_xs[s] + x_split1_ys[-1]))))
        x_split1 = torch.cat(x_split1_ys, dim=1)
        # 加入SE模块
        # if self.se is not None:
        #     x_split1 = self.se(x_split1)
        x_split1 = x_split1.flatten(2).transpose(1, 2)

        x_split2_xs = torch.chunk(x_split2, self.scales, 1)  # 将x分割成scales块
        x_split2_ys = []
        for s in range(self.scales):
            if s == 0:
                x_split2_ys.append(x_split2_xs[s])
            elif s == 1:
                x_split2_ys.append(self.act2(self.bn2[s - 1](self.dwconv2[s - 1](x_split2_xs[s]))))
            else:
                x_split2_ys.append(self.act2(self.bn2[s - 1](self.dwconv2[s - 1](x_split2_xs[s] + x_split2_ys[-1]))))
        x_split2 = torch.cat(x_split2_ys, dim=1)
        # 加入SE模块
        # if self.se is not None:
        #     x_split2 = self.se(x_split2)
        x_split2 = x_split2.flatten(2).transpose(1, 2)

        x_split3_xs = torch.chunk(x_split3, self.scales, 1)  # 将x分割成scales块
        x_split3_ys = []
        for s in range(self.scales):
            if s == 0:
                x_split3_ys.append(x_split3_xs[s])
            elif s == 1:
                x_split3_ys.append(self.act3(self.bn3[s - 1](self.dwconv3[s - 1](x_split3_xs[s]))))
            else:
                x_split3_ys.append(self.act3(self.bn3[s - 1](self.dwconv3[s - 1](x_split3_xs[s] + x_split3_ys[-1]))))
        x_split3 = torch.cat(x_split3_ys, dim=1)
        # 加入SE模块
        # if self.se is not None:
        #     x_split3 = self.se(x_split3)
        x_split3 = x_split3.flatten(2).transpose(1, 2)
        x_split = torch.cat([x_split1, x_split2, x_split3], dim=1)
        x = torch.cat((z_split, x_split), dim=1)

        # x_split11, x_split12 = x_split1[:, :C_x // 2, :, :], x_split1[:, C_x // 2:, :, :]
        # x_split11 = self.dwconv1(x_split11)
        # x_split12 = self.dwconv2(x_split12)
        # x_split1 = torch.cat([x_split11, x_split12], dim=1)
        # x_split1 = self.act1(self.bn1(x_split1)).flatten(2).transpose(1, 2)
        #
        # x_split21, x_split22 = x_split2[:, :C_x // 2, :, :], x_split2[:, C_x // 2:, :, :]
        # x_split21 = self.dwconv3(x_split21)
        # x_split22 = self.dwconv4(x_split22)
        # x_split2 = torch.cat([x_split21, x_split22], dim=1)
        # x_split2 = self.act2(self.bn2(x_split2)).flatten(2).transpose(1, 2)
        #
        # x_split31, x_split32 = x_split3[:, :C_x // 2, :, :], x_split3[:, C_x // 2:, :, :]
        # x_split31 = self.dwconv5(x_split31)
        # x_split32 = self.dwconv6(x_split32)
        # x_split3 = torch.cat([x_split31, x_split32], dim=1)
        # x_split3 = self.act3(self.bn3(x_split3)).flatten(2).transpose(1, 2)
        # x_split = torch.cat([x_split1, x_split2, x_split3], dim=1)
        # x = torch.cat((z_split, x_split), dim=1)
        return x


class ConvFFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H_z, W_z, H_x, W_x):
        x = self.fc1(x)
        x = self.dwconv(x, H_z, W_z, H_x, W_x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H_z, W_z, H_x, W_x):
        # 2024.7.5
        z_split = x[:, :336, :]
        x_split = x[:, 336:, :]
        B_z, N_z, C_z = z_split.shape
        n_z = N_z // 21
        z_split1 = z_split[:, 0:16 * n_z, :].transpose(1, 2).view(B_z, C_z, H_z * 2, W_z * 2)
        z_split2 = z_split[:, 16 * n_z:20 * n_z, :].transpose(1, 2).view(B_z, C_z, H_z, W_z)
        z_split3 = z_split[:, 20 * n_z:, :].transpose(1, 2).view(B_z, C_z, H_z // 2, W_z // 2)
        z_split1 = self.dwconv(z_split1).flatten(2).transpose(1, 2)
        z_split2 = self.dwconv(z_split2).flatten(2).transpose(1, 2)
        z_split3 = self.dwconv(z_split3).flatten(2).transpose(1, 2)
        z_split = torch.cat([z_split1, z_split2, z_split3], dim=1)
        B_x, N_x, C_x = x_split.shape
        n_x = N_x // 21
        x_split1 = x_split[:, 0:16 * n_x, :].transpose(1, 2).view(B_x, C_x, H_x * 2, W_x * 2)
        x_split2 = x_split[:, 16 * n_x:20 * n_x, :].transpose(1, 2).view(B_x, C_x, H_x, W_x)
        x_split3 = x_split[:, 20 * n_x:, :].transpose(1, 2).view(B_x, C_x, H_x // 2, W_x // 2)
        x_split1 = self.dwconv(x_split1).flatten(2).transpose(1, 2)
        x_split2 = self.dwconv(x_split2).flatten(2).transpose(1, 2)
        x_split3 = self.dwconv(x_split3).flatten(2).transpose(1, 2)
        x_split = torch.cat([x_split1, x_split2, x_split3], dim=1)
        x = torch.cat((z_split, x_split), dim=1)

        # 2024.7.6
        # B_z, N_z, C_z = x.shape
        # n_z = N_z // 21
        # z_split1 = x[:, 0:16 * n_z, :].transpose(1, 2).view(B_z, C_z, H_z * 2, W_z * 2).contiguous()
        # z_split2 = x[:, 16 * n_z:20 * n_z, :].transpose(1, 2).view(B_z, C_z, H_z, W_z).contiguous()
        # z_split3 = x[:, 20 * n_z:, :].transpose(1, 2).view(B_z, C_z, H_z // 2, W_z // 2).contiguous()
        # z_split1 = self.dwconv(z_split1).flatten(2).transpose(1, 2)
        # z_split2 = self.dwconv(z_split2).flatten(2).transpose(1, 2)
        # z_split3 = self.dwconv(z_split3).flatten(2).transpose(1, 2)
        # x = torch.cat([z_split1, z_split2, z_split3], dim=1)
        return x


class Extractor(nn.Module):
    def __init__(self, dim, hide_dim=8, num_heads=6, attn_drop=0., qkv_bias=False,
                 with_cffn=True, cffn_ratio=0.25, drop=0., drop_path=0.,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6), with_cp=False):
        super().__init__()
        self.query_norm = norm_layer(hide_dim)
        self.feat_norm = norm_layer(hide_dim)
        self.attn = Cross_Attention(hide_dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop,
                                    proj_drop=drop)
        self.with_cffn = with_cffn
        self.with_cp = with_cp

        # self.r = hide_dim
        # self.lora_A_q = nn.Parameter(torch.zeros((self.r, dim)))
        # self.lora_A_kv = nn.Parameter(torch.zeros((self.r, dim)))
        # self.lora_B_q = nn.Parameter(torch.zeros((dim, self.r)))
        # nn.init.kaiming_uniform_(self.lora_A_q, a=math.sqrt(5))
        # nn.init.kaiming_uniform_(self.lora_A_kv, a=math.sqrt(5))
        # nn.init.zeros_(self.lora_B_q)
        self.q_down = nn.Linear(dim, hide_dim)
        self.q_up = nn.Linear(hide_dim, dim)
        self.kv_down = nn.Linear(dim, hide_dim)
        if with_cffn:
            self.ffn = ConvFFN(in_features=dim, hidden_features=int(dim * cffn_ratio), drop=drop)
            self.ffn_norm = norm_layer(dim)
            self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, query, feat, H_z, W_z, H_x, W_x):

        def _inner_forward(query, feat, H_z, W_z, H_x, W_x):
            # 2024.7.5
            # zc_split = query[:, :336, :]
            # zc_select1, zc_select2, zc_select3 = zc_split[:, :H_z * W_z * 4, :], zc_split[:,H_z * W_z * 4:H_z * W_z * 4 + H_z * W_z,:] \
            #                                      , zc_split[:, H_z * W_z * 4 + H_z * W_z:, :]
            # zx_split = feat[:, :64, :]
            # zc_split2 = torch.cat([zx_split, zc_select2], dim=2)
            # zc_split2 = self.fusion(zc_split2)
            # zc_split1 = self.attn(self.query_norm(zc_select1), self.feat_norm(zc_split2))
            # zc_split3 = self.attn(self.query_norm(zc_select3), self.feat_norm(zc_split2))
            #
            # xc_split = query[:, 336:, :]
            # xc_select1, xc_select2, xc_select3 = xc_split[:, :H_x * W_x * 4, :], xc_split[:,H_x * W_x * 4:H_x * W_x * 4 + H_x * W_x,:] \
            #                                      , xc_split[:, H_x * W_x * 4 + H_x * W_x:, :]
            # xx_split = feat[:, 64:, :]
            # xc_split2 = torch.cat([xx_split, xc_select2], dim=2)
            # xc_split2 = self.fusion(xc_split2)
            # xc_split1 = self.attn(self.query_norm(xc_select1), self.feat_norm(xc_split2))
            # xc_split3 = self.attn(self.query_norm(xc_select3), self.feat_norm(xc_split2))
            # attn = torch.cat([zc_split1, zc_split2, zc_split3, xc_split1, xc_split2, xc_split3], dim=1)

            # q = query @ self.lora_A_q.T
            # kv = feat @ self.lora_A_kv.T
            # attn = self.attn(self.query_norm(q),
            #                  self.feat_norm(kv))
            # attn = attn @ self.lora_B_q.T
            q = self.q_down(query)
            kv = self.kv_down(feat)
            attn = self.attn(self.query_norm(q),
                             self.feat_norm(kv))
            attn = self.q_up(attn)

            query = query + attn

            if self.with_cffn:
                query = query + self.drop_path(self.ffn(self.ffn_norm(query), H_z, W_z, H_x, W_x))
            return query

        if self.with_cp and query.requires_grad:
            query = cp.checkpoint(_inner_forward, query, feat, H_z, W_z, H_x, W_x)
        else:
            query = _inner_forward(query, feat, H_z, W_z, H_x, W_x)

        return query


class Injector(nn.Module):
    def __init__(self, dim, hide_dim=8, num_heads=6, attn_drop=0., qkv_bias=False,
                 drop=0., norm_layer=partial(nn.LayerNorm, eps=1e-6), init_values=0., with_cp=False, cffn_ratio=0.25,
                 drop_path=0., ):
        super().__init__()
        self.with_cp = with_cp
        self.query_norm = norm_layer(hide_dim)
        self.feat_norm = norm_layer(hide_dim)
        self.attn = Cross_Attention(hide_dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop,
                                    proj_drop=drop)
        self.gamma = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)

        # self.r = hide_dim
        # self.lora_A_q = nn.Parameter(torch.zeros((self.r, dim)))
        # self.lora_A_kv = nn.Parameter(torch.zeros((self.r, dim)))
        # self.lora_B_q = nn.Parameter(torch.zeros((dim, self.r)))
        # nn.init.kaiming_uniform_(self.lora_A_q, a=math.sqrt(5))
        # nn.init.kaiming_uniform_(self.lora_A_kv, a=math.sqrt(5))
        # nn.init.zeros_(self.lora_B_q)

        self.q_down = nn.Linear(dim, hide_dim)
        self.q_up = nn.Linear(hide_dim, dim)
        self.kv_down = nn.Linear(dim, hide_dim)

    def forward(self, query, feat, H_z, W_z, H_x, W_x):

        def _inner_forward(query, feat, H_z, W_z, H_x, W_x):
            # 2024.7.5
            # zc_split = feat[:, :336, :]
            # zc_select1, zc_select2, zc_select3 = zc_split[:, :H_z * W_z * 4, :], zc_split[:,H_z * W_z * 4:H_z * W_z * 4 + H_z * W_z,:] \
            #                                      , zc_split[:, H_z * W_z * 4 + H_z * W_z:, :]
            # zx_split2 = query[:, :64, :]
            # zx_split2 = torch.cat([zx_split2, zc_select2], dim=2)
            # zx_split2 = self.fusion(zx_split2)
            # zx_split1 = self.attn(self.query_norm(zx_split2), self.feat_norm(zc_select1))
            # zx_split3 = self.attn(self.query_norm(zx_split2), self.feat_norm(zc_select3))
            # zx = zx_split1 + zx_split3 + zx_split2
            #
            # xc_split = feat[:, 336:, :]
            # xc_select1, xc_select2, xc_select3 = xc_split[:, :H_x * W_x * 4, :], xc_split[:,H_x * W_x * 4:H_x * W_x * 4 + H_x * W_x,:] \
            #                                     , xc_split[:, H_x * W_x * 4 + H_x * W_x:, :]
            # xx_split2 = query[:, 64:, :]
            # xx_split2 = torch.cat([xx_split2, xc_select2], dim=2)
            # xx_split2 = self.fusion(xx_split2)
            # xx_split1 = self.attn(self.query_norm(xx_split2), self.feat_norm(xc_select1))
            # xx_split3 = self.attn(self.query_norm(xx_split2), self.feat_norm(xc_select3))
            # xx = xx_split1 + xx_split3 + xx_split2
            # attn = torch.cat([zx, xx], dim=1)

            # q = query @ self.lora_A_q.T
            # kv = feat @ self.lora_A_kv.T
            # attn = self.attn(self.query_norm(q),
            #                  self.feat_norm(kv))
            # attn = attn @ self.lora_B_q.T

            q = self.q_down(query)
            kv = self.kv_down(feat)

            attn = self.attn(self.query_norm(q),
                             self.feat_norm(kv))
            attn = self.q_up(attn)
            return query + self.gamma * attn

        if self.with_cp and query.requires_grad:
            query = cp.checkpoint(_inner_forward, query, feat, H_z, W_z, H_x, W_x)
        else:
            query = _inner_forward(query, feat, H_z, W_z, H_x, W_x)

        return query


class ConvFFN_New(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv_New(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H_z, W_z, H_x, W_x):
        x = self.fc1(x)
        x = self.dwconv(x, H_z, W_z, H_x, W_x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DWConv_New(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H_z, W_z, H_x, W_x):
        # Split for template (z) and search (x) regions
        z_split = x[:, :208, :]  # Template tokens
        x_split = x[:, 208:, :]  # Search tokens

        B_z, N_z, C_z = z_split.shape
        n_z = N_z // 25  # 400/16=25 tokens per part

        # Template: Split into 4 parts (256+64+64+16 = 400 tokens)
        z_split1 = z_split[:, :64, :].transpose(1, 2).view(B_z, C_z, H_z, W_z)  # 256 tokens
        z_split2 = z_split[:, 64:128, :].transpose(1, 2).view(B_z, C_z, H_z, W_z)  # 64 tokens
        z_split3 = z_split[:, 128:192, :].transpose(1, 2).view(B_z, C_z, H_z, W_z)  # 64 tokens
        z_split4 = z_split[:, 192:208, :].transpose(1, 2).view(B_z, C_z, H_z // 2, W_z // 2)  # 16 tokens

        # Apply convolution to each part
        z_split1 = self.dwconv(z_split1).flatten(2).transpose(1, 2)
        z_split2 = self.dwconv(z_split2).flatten(2).transpose(1, 2)
        z_split3 = self.dwconv(z_split3).flatten(2).transpose(1, 2)
        z_split4 = self.dwconv(z_split4).flatten(2).transpose(1, 2)

        # Concatenate template parts
        z_split = torch.cat([z_split1, z_split2, z_split3, z_split4], dim=1)

        B_x, N_x, C_x = x_split.shape
        n_x = N_x // 25  # 1600/64=25 tokens per part

        # Search: Split into 4 parts (1024+256+256+64 = 1600 tokens)
        x_split1 = x_split[:, :256, :].transpose(1, 2).view(B_x, C_x, H_x, W_x)  # 1024 tokens
        x_split2 = x_split[:, 256:512, :].transpose(1, 2).view(B_x, C_x, H_x, W_x)  # 256 tokens
        x_split3 = x_split[:, 512:768, :].transpose(1, 2).view(B_x, C_x, H_x, W_x)  # 256 tokens
        x_split4 = x_split[:, 768:832, :].transpose(1, 2).view(B_x, C_x, H_x // 2, W_x // 2)  # 64 tokens

        # Apply convolution to each part
        x_split1 = self.dwconv(x_split1).flatten(2).transpose(1, 2)
        x_split2 = self.dwconv(x_split2).flatten(2).transpose(1, 2)
        x_split3 = self.dwconv(x_split3).flatten(2).transpose(1, 2)
        x_split4 = self.dwconv(x_split4).flatten(2).transpose(1, 2)

        # Concatenate search parts
        x_split = torch.cat([x_split1, x_split2, x_split3, x_split4], dim=1)

        # Combine template and search features
        x = torch.cat((z_split, x_split), dim=1)
        return x


class Extractor_New(nn.Module):
    def __init__(self, dim, hide_dim=8, num_heads=6, attn_drop=0., qkv_bias=False,
                 with_cffn=True, cffn_ratio=0.25, drop=0., drop_path=0.,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6), with_cp=False):
        super().__init__()
        self.query_norm = norm_layer(dim)
        self.feat_norm = norm_layer(dim)
        self.attn = Cross_Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop,
                                    proj_drop=drop)
        self.with_cffn = with_cffn
        self.with_cp = with_cp

        # Linear projections for dimension reduction and expansion
        self.q_down = nn.Linear(dim, hide_dim)
        self.q_up = nn.Linear(hide_dim, dim)
        self.kv_down = nn.Linear(dim, hide_dim)

        if with_cffn:
            self.ffn = ConvFFN_New(in_features=dim, hidden_features=int(dim * cffn_ratio), drop=drop)
            self.ffn_norm = norm_layer(dim)
            self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, query, feat, H_z, W_z, H_x, W_x):
        def _inner_forward(query, feat, H_z, W_z, H_x, W_x):
            # Project to lower dimension
            # q = self.q_down(query)
            # kv = self.kv_down(feat)

            # Apply cross attention
            attn = self.attn(self.query_norm(query),
                             self.feat_norm(feat))

            # Project back to original dimension
            # attn = self.q_up(attn)

            # Residual connection
            query = query + attn

            # Apply ConvFFN if enabled
            if self.with_cffn:
                query = query + self.drop_path(self.ffn(self.ffn_norm(query), H_z, W_z, H_x, W_x))
            return query

        if self.with_cp and query.requires_grad:
            query = cp.checkpoint(_inner_forward, query, feat, H_z, W_z, H_x, W_x)
        else:
            query = _inner_forward(query, feat, H_z, W_z, H_x, W_x)

        return query

class CNN(nn.Module):
    def __init__(self, inplanes=64, embed_dim=384, warehouse_manager=None):
        super(CNN, self).__init__()
        self.warehouse_manager = warehouse_manager or Warehouse_Manager()
        # warehouse_manager是一个模块管理类，负责模块的创建，缓存，复用和存储
        self.stem = nn.Sequential(*[
            nn.Conv2d(3, inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(inplanes),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(inplanes),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(inplanes),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        ])

        self.conv2 = nn.Sequential(
            self._kwconv3x3_2(inplanes, 2 * inplanes, stride=2, padding=1),
            nn.BatchNorm2d(2 * inplanes),
            nn.ReLU(inplace=True)
        )

        self.conv3 = nn.Sequential(
            self._kwconv3x3_3(2 * inplanes, 4 * inplanes, stride=2, padding=1),
            nn.BatchNorm2d(4 * inplanes),
            nn.ReLU(inplace=True)
        )

        self.conv4 = nn.Sequential(
            self._kwconv3x3_4(4 * inplanes, 4 * inplanes, stride=2, padding=1),
            nn.BatchNorm2d(4 * inplanes),
            nn.ReLU(inplace=True)
        )

        self.fc1 = nn.Conv2d(inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.fc2 = nn.Conv2d(2 * inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.fc3 = nn.Conv2d(4 * inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.fc4 = nn.Conv2d(4 * inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)

        self.warehouse_manager.store()
        self.warehouse_manager.allocate(self)

    def _kwconv3x3_2(self, in_planes, out_planes, stride=1, padding=1):
        warehouse_name = 'conv3x3_layer2'
        return self.warehouse_manager.reserve(
            in_planes, out_planes, kernel_size=3, stride=stride, padding=padding,
            warehouse_name=warehouse_name, enabled=True, bias=False)

    def _kwconv3x3_3(self, in_planes, out_planes, stride=1, padding=1):
        warehouse_name = 'conv3x3_layer3'
        return self.warehouse_manager.reserve(
            in_planes, out_planes, kernel_size=3, stride=stride, padding=padding,
            warehouse_name=warehouse_name, enabled=True, bias=False)

    def _kwconv3x3_4(self, in_planes, out_planes, stride=1, padding=1):
        warehouse_name = 'conv3x3_layer4'
        return self.warehouse_manager.reserve(
            in_planes, out_planes, kernel_size=3, stride=stride, padding=padding,
            warehouse_name=warehouse_name, enabled=True, bias=False)

    def forward(self, x):
        B, C, H, W = x.shape
        c1 = self.stem(x)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        c4 = self.conv4(c3)
        c1 = self.fc1(c1)
        c2 = self.fc2(c2)
        c3 = self.fc3(c3)
        c4 = self.fc4(c4)
        bs, dim, _, _ = c1.shape
        # c1 = c1.view(bs, dim, -1).transpose(1, 2)  # 4s
        # c11 = c1.cpu()
        # c11 = c11.view(1, H * W // 16, 768).mean(dim=2).squeeze().view(H // 4, W // 4)
        # # 绘制图像并设置值范围和插值方法
        # plt.imshow(c11, cmap='viridis', interpolation='bilinear')
        # # 添加颜色条
        # plt.colorbar()
        # # 显示图像
        # plt.show()

        c2 = c2.view(bs, dim, -1).transpose(1, 2)  # 8s

        # c21 = c2.cpu()
        # c21 = c21.view(1, H * W // 64, 768).mean(dim=2).squeeze().view(H // 8, W // 8)
        # # 绘制图像并设置值范围和插值方法
        # plt.imshow(c21, cmap='viridis', interpolation='bilinear')
        # # 添加颜色条
        # plt.colorbar()
        # # 显示图像
        # plt.show()

        c3 = c3.view(bs, dim, -1).transpose(1, 2)  # 16s

        # c31 = c3.cpu()
        # c31 = c31.view(1, H * W // 256, 768).mean(dim=2).squeeze().view(H // 16, W // 16)
        # # 绘制图像并设置值范围和插值方法
        # plt.imshow(c31, cmap='viridis', interpolation='bilinear')
        # # 添加颜色条
        # plt.colorbar()
        # # 显示图像
        # plt.show()

        c4 = c4.view(bs, dim, -1).transpose(1, 2)  # 32s

        # c41 = c4.cpu()
        # c41 = c41.view(1, H * W // 1024, 768).mean(dim=2).squeeze().view(H // 32, W // 32)
        # # 绘制图像并设置值范围和插值方法
        # plt.imshow(c41, cmap='viridis', interpolation='bilinear')
        # # 添加颜色条
        # plt.colorbar()
        # # 显示图像
        # plt.show()

        return c1, c2, c3, c4


# class CNN(nn.Module):
#     def __init__(self, inplanes=64, embed_dim=384):
#         super().__init__()
#
#         self.stem = nn.Sequential(*[
#             nn.Conv2d(3, inplanes, kernel_size=3, stride=2, padding=1, bias=False),
#             nn.BatchNorm2d(inplanes),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.BatchNorm2d(inplanes),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.BatchNorm2d(inplanes),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#         ])
#         self.conv2 = nn.Sequential(*[
#             nn.Conv2d(inplanes, 2 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
#             nn.BatchNorm2d(2 * inplanes),
#             nn.ReLU(inplace=True)
#         ])
#         self.conv3 = nn.Sequential(*[
#             nn.Conv2d(2 * inplanes, 4 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
#             nn.BatchNorm2d(4 * inplanes),
#             nn.ReLU(inplace=True)
#         ])
#         self.conv4 = nn.Sequential(*[
#             nn.Conv2d(4 * inplanes, 4 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
#             nn.BatchNorm2d(4 * inplanes),
#             nn.ReLU(inplace=True)
#         ])
#         self.fc1 = nn.Conv2d(inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
#         self.fc2 = nn.Conv2d(2 * inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
#         self.fc3 = nn.Conv2d(4 * inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
#         self.fc4 = nn.Conv2d(4 * inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
#         # self.MRM2 = ChunkConv(2 * inplanes)
#         # self.MRM3 = ChunkConv(4 * inplanes)
#         # self.MRM4 = ChunkConv(4 * inplanes)
#
#     def forward(self, x):
#         c1 = self.stem(x)
#         c2 = self.conv2(c1)
#         c3 = self.conv3(c2)
#         c4 = self.conv4(c3)
#         # c2 = self.MRM2(c2)
#         # c3 = self.MRM3(c3)
#         # c4 = self.MRM4(c4)
#         c1 = self.fc1(c1)
#         c2 = self.fc2(c2)
#         c3 = self.fc3(c3)
#         c4 = self.fc4(c4)
#         bs, dim, _, _ = c1.shape
#         # c1 = c1.view(bs, dim, -1).transpose(1, 2)  # 4s
#         c2 = c2.view(bs, dim, -1).transpose(1, 2)  # 8s
#         c3 = c3.view(bs, dim, -1).transpose(1, 2)  # 16s
#         c4 = c4.view(bs, dim, -1).transpose(1, 2)  # 32s
#
#         return c1, c2, c3, c4

# 2024/6/28
class RepDWConvS(nn.Module):
    def __init__(self, in_channels, stride=1, bias=True):
        super().__init__()
        self.stride = stride
        kwargs = {"in_channels": in_channels, "out_channels": in_channels, "groups": in_channels}
        self.conv_3_3 = nn.Conv2d(bias=bias, kernel_size=3, stride=stride, dilation=1, padding=1, **kwargs)
        self.conv_3_w = nn.Conv2d(bias=bias and stride == 1, kernel_size=(1, 3), stride=(1, stride), padding=(0, 1),
                                  **kwargs)
        self.conv_3_h = nn.Conv2d(bias=bias and stride == 1, kernel_size=(3, 1), stride=(stride, 1), padding=(1, 0),
                                  **kwargs)
        self.conv_2_2 = nn.Conv2d(bias=bias, kernel_size=2, stride=stride, dilation=2, padding=1, **kwargs)

    def forward(self, x):
        if self.stride == 1:
            return self.conv_3_3(x) + self.conv_3_h(x) + self.conv_3_w(x) + self.conv_2_2(x)
        return self.conv_3_3(x) + self.conv_3_h(self.conv_3_w(x)) + self.conv_2_2(x)


class RepDWConvM(nn.Module):
    def __init__(self, in_channels, stride=1, bias=True):
        super().__init__()
        kwargs = {"in_channels": in_channels, "out_channels": in_channels, "groups": in_channels}
        self.conv_5_5 = nn.Conv2d(bias=bias, kernel_size=(5, 5), stride=stride, padding=2, **kwargs)
        self.conv_5_3 = nn.Conv2d(bias=bias, kernel_size=(5, 3), stride=stride, padding=(2, 1), **kwargs)
        self.conv_3_5 = nn.Conv2d(bias=bias, kernel_size=(3, 5), stride=stride, padding=(1, 2), **kwargs)
        self.conv_5_w = nn.Conv2d(bias=False, kernel_size=(1, 5), stride=(1, stride), padding=(0, 2), **kwargs)
        self.conv_5_h = nn.Conv2d(bias=False, kernel_size=(5, 1), stride=(stride, 1), padding=(2, 0), **kwargs)

    def forward(self, x):
        return self.conv_5_5(x) + self.conv_5_3(x) + self.conv_3_5(x) + self.conv_5_h(self.conv_5_w(x))


class RepDWConvL(nn.Module):
    def __init__(self, in_channels, stride=1, bias=True):
        super().__init__()
        kwargs = {"in_channels": in_channels, "out_channels": in_channels, "groups": in_channels}
        self.conv_7_7 = nn.Conv2d(bias=bias, kernel_size=(7, 7), stride=stride, padding=3, **kwargs)
        self.conv_5_3 = nn.Conv2d(bias=bias, kernel_size=(5, 3), stride=stride, padding=(2, 1), **kwargs)
        self.conv_3_5 = nn.Conv2d(bias=bias, kernel_size=(3, 5), stride=stride, padding=(1, 2), **kwargs)
        self.conv_7_w = nn.Conv2d(bias=False, kernel_size=(1, 7), stride=(1, stride), padding=(0, 3), **kwargs)
        self.conv_7_h = nn.Conv2d(bias=False, kernel_size=(7, 1), stride=(stride, 1), padding=(3, 0), **kwargs)
        self.conv_5_w = nn.Conv2d(bias=False, kernel_size=(1, 5), stride=(1, stride), padding=(0, 2), **kwargs)
        self.conv_5_h = nn.Conv2d(bias=False, kernel_size=(5, 1), stride=(stride, 1), padding=(2, 0), **kwargs)

    def forward(self, x):
        return self.conv_7_7(x) + self.conv_5_3(x) + self.conv_3_5(x) + self.conv_7_h(self.conv_7_w(x)) + self.conv_5_h(
            self.conv_5_w(x))


class ChunkConv(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        assert in_channels % 4 == 0
        hidden_channels = in_channels // 4
        self.conv_s = RepDWConvS(hidden_channels)
        self.conv_m = RepDWConvM(hidden_channels)
        self.conv_l = RepDWConvL(hidden_channels)

    def forward(self, x):
        i, s, m, l = torch.chunk(x, 4, dim=1)
        return torch.cat((i, self.conv_s(s), self.conv_m(m), self.conv_l(l)), dim=1)
