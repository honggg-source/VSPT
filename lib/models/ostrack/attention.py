import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class FocusedLinearCross_Attention_toV(nn.Module):
    def __init__(self, dim,  num_heads=8, qkv_bias=False, attn_drop=0., n=1680, proj_drop=0.,focusing_factor=3, kernel_size=5):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        # 2024/6/29
        self.n = n
        self.focusing_factor = focusing_factor
        self.scale = nn.Parameter(torch.zeros(size=(1, 1, dim)))
        self.positional_encoding = nn.Parameter(torch.zeros(size=(1, self.n, dim)))
        self.dwc1 = nn.Conv2d(in_channels=head_dim, out_channels=head_dim, kernel_size=kernel_size,
                             groups=head_dim, padding=kernel_size // 2)

        self.dwc2 = nn.Conv2d(in_channels=head_dim, out_channels=head_dim, kernel_size=kernel_size,
                              groups=head_dim, padding=kernel_size // 2)

        self.dwc3 = nn.Conv2d(in_channels=head_dim, out_channels=head_dim, kernel_size=kernel_size,
                              groups=head_dim, padding=kernel_size // 2)

    def forward(self, query, feat, mask=None, return_attention=False):
        # x: B, N, C
        # mask: [B, N, ] torch.bool
        q = query
        k = feat
        v = feat
        B_q, N_q, C_q = q.shape
        B_k, N_k, C_k = k.shape

        k = k + self.positional_encoding
        focusing_factor = self.focusing_factor
        kernel_function = nn.ReLU()
        q = kernel_function(q) + 1e-6
        k = kernel_function(k) + 1e-6
        scale = nn.Softplus()(self.scale)
        q = q / scale
        k = k / scale
        q_norm = q.norm(dim=-1, keepdim=True)
        k_norm = k.norm(dim=-1, keepdim=True)
        q = q ** focusing_factor
        k = k ** focusing_factor
        q = (q / q.norm(dim=-1, keepdim=True)) * q_norm
        k = (k / k.norm(dim=-1, keepdim=True)) * k_norm

        q = q.reshape(B_q, N_q, self.num_heads, -1).permute(0, 2, 1, 3)
        k = k.reshape(B_k, N_k, self.num_heads, -1).permute(0, 2, 1, 3)
        v = v.reshape(B_k, N_k, self.num_heads, -1).permute(0, 2, 1, 3)

        z = 1 / (q @ k.mean(dim=-2, keepdim=True).transpose(-2, -1) + 1e-6)
        kv = (k.transpose(-2, -1) * (N_k ** -0.5)) @ (v * (N_k ** -0.5))
        x = q @ kv * z

        x = x.transpose(1, 2).reshape(B_q, N_q, C_q)
        v_z1 = v[:, : , :256 , :].contiguous()
        v_z2 = v[:, :, 256:256+64, :].contiguous()
        v_z3 = v[:, :, 256+64:256 + 64 + 16, :].contiguous()
        v_z1 = v_z1.reshape(B_k * self.num_heads,16,16,-1).permute(0, 3, 1, 2)
        v_z2 = v_z2.reshape(B_k * self.num_heads, 8, 8, -1).permute(0, 3, 1, 2)
        v_z3 = v_z3.reshape(B_k * self.num_heads, 4, 4, -1).permute(0, 3, 1, 2)
        v_z1 = self.dwc1(v_z1)
        v_z2 = self.dwc2(v_z2)
        v_z3 = self.dwc3(v_z3)
        v_z1 = F.interpolate(v_z1, scale_factor=0.5, mode='bilinear', align_corners=False)
        v_z3 = F.interpolate(v_z3, scale_factor=2, mode='bilinear', align_corners=False)
        v_z1 = v_z1.reshape(B_q, C_q, 64).permute(0, 2, 1)
        v_z2 = v_z2.reshape(B_q, C_q, 64).permute(0, 2, 1)
        v_z3 = v_z3.reshape(B_q, C_q, 64).permute(0, 2, 1)
        v_z = v_z1 + v_z2 + v_z3
        v_x1 = v[:, :, 336:336+1024, :].contiguous()
        v_x2 = v[:, :, 336+1024:336+1024+256, :].contiguous()
        v_x3 = v[:, :, 336+1024+256:, :].contiguous()
        v_x1 = v_x1.reshape(B_k * self.num_heads, 32, 32, -1).permute(0, 3, 1, 2)
        v_x2 = v_x2.reshape(B_k * self.num_heads, 16, 16, -1).permute(0, 3, 1, 2)
        v_x3 = v_x3.reshape(B_k * self.num_heads, 8, 8, -1).permute(0, 3, 1, 2)
        v_x1 = self.dwc1(v_x1)
        v_x2 = self.dwc2(v_x2)
        v_x3 = self.dwc3(v_x3)
        v_x1 = F.interpolate(v_x1, scale_factor=0.5, mode='bilinear', align_corners=False)
        v_x3 = F.interpolate(v_x3, scale_factor=2, mode='bilinear', align_corners=False)
        v_x1 = v_x1.reshape(B_q, C_q, 256).permute(0, 2, 1)
        v_x2 = v_x2.reshape(B_q, C_q, 256).permute(0, 2, 1)
        v_x3 = v_x3.reshape(B_q, C_q, 256).permute(0, 2, 1)
        v_x = v_x1 + v_x2 + v_x3
        v = torch.cat([v_z, v_x], dim=1)
        x = x + v

        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class FocusedLinearCross_Attention_toC(nn.Module):
    def __init__(self, dim,  num_heads=8, qkv_bias=False, attn_drop=0., n=320, proj_drop=0.,focusing_factor=3, kernel_size=5):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        # 2024/6/29
        self.n = n
        self.focusing_factor = focusing_factor
        self.scale = nn.Parameter(torch.zeros(size=(1, 1, dim)))
        self.positional_encoding = nn.Parameter(torch.zeros(size=(1, self.n, dim)))
        self.dwc = nn.Conv2d(in_channels=head_dim, out_channels=head_dim, kernel_size=kernel_size,
                             groups=head_dim, padding=kernel_size // 2)

    def forward(self, query, feat, mask=None, return_attention=False):
        # x: B, N, C
        # mask: [B, N, ] torch.bool
        q = query
        k = feat
        v = feat
        B_q, N_q, C_q = q.shape
        B_k, N_k, C_k = k.shape

        k = k + self.positional_encoding
        focusing_factor = self.focusing_factor
        kernel_function = nn.ReLU()
        q = kernel_function(q) + 1e-6
        k = kernel_function(k) + 1e-6
        scale = nn.Softplus()(self.scale)
        q = q / scale
        k = k / scale
        q_norm = q.norm(dim=-1, keepdim=True)
        k_norm = k.norm(dim=-1, keepdim=True)
        q = q ** focusing_factor
        k = k ** focusing_factor
        q = (q / q.norm(dim=-1, keepdim=True)) * q_norm
        k = (k / k.norm(dim=-1, keepdim=True)) * k_norm

        q = q.reshape(B_q, N_q, self.num_heads, -1).permute(0, 2, 1, 3)
        k = k.reshape(B_k, N_k, self.num_heads, -1).permute(0, 2, 1, 3)
        v = v.reshape(B_k, N_k, self.num_heads, -1).permute(0, 2, 1, 3)

        z = 1 / (q @ k.mean(dim=-2, keepdim=True).transpose(-2, -1) + 1e-6)
        kv = (k.transpose(-2, -1) * (N_k ** -0.5)) @ (v * (N_k ** -0.5))
        x = q @ kv * z

        x = x.transpose(1, 2).reshape(B_q, N_q, C_q)
        v_z = v[:, : , :64 , :].contiguous()
        v_z = v_z.reshape(B_k * self.num_heads,8,8,-1).permute(0, 3, 1, 2)
        v_z2 = self.dwc(v_z)
        v_z1 = F.interpolate(v_z2, scale_factor=2, mode='bilinear', align_corners=False)
        v_z3 = F.interpolate(v_z2, scale_factor=0.5, mode='bilinear', align_corners=False)
        v_z1 = v_z1.reshape(B_q, C_q, 256).permute(0, 2, 1)
        v_z2 = v_z2.reshape(B_q, C_q, 64).permute(0, 2, 1)
        v_z3 = v_z3.reshape(B_q, C_q, 16).permute(0, 2, 1)
        v_z = torch.cat([v_z1, v_z2, v_z3], dim=1)
        v_x = v[:, :, 64:, :].contiguous()
        v_x = v_x.reshape(B_k * self.num_heads, 16, 16, -1).permute(0, 3, 1, 2)
        v_x2 = self.dwc(v_x)
        v_x1 = F.interpolate(v_x2, scale_factor=2, mode='bilinear', align_corners=False)
        v_x3 = F.interpolate(v_x2, scale_factor=0.5, mode='bilinear', align_corners=False)
        v_x1 = v_x1.reshape(B_q, C_q, 1024).permute(0, 2, 1)
        v_x2 = v_x2.reshape(B_q, C_q, 256).permute(0, 2, 1)
        v_x3 = v_x3.reshape(B_q, C_q, 64).permute(0, 2, 1)
        v_x = torch.cat([v_x1, v_x2, v_x3], dim=1)
        v = torch.cat([v_z, v_x], dim=1)
        x = x + v

        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Cross_Attention(nn.Module):
    def __init__(self, dim,  num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.scale = dim ** -0.5
        # self.scale = head_dim ** -0.5  # NOTE: Small scale for attention map normalization

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)


    def forward(self, query, feat, mask=None, return_attention=False):
        # x: B, N, C
        # mask: [B, N, ] torch.bool
        B, N, C = query.shape
        q = query
        k = feat
        v = feat

        attn = (q @ k.transpose(-2, -1)) * self.scale  # B, lens_z, lens_x; B, lens_x, lens_z

        if mask is not None:
            attn = attn.masked_fill(mask.unsqueeze(1).unsqueeze(2), float('-inf'), )

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = attn @ v  # B, lens_z/x, C
        x = x.transpose(1, 2)  # B, C, lens_z/x
        x = x.reshape(B, -1, C)  # B, lens_z/x, C; NOTE: Rearrange channels, marginal improvement
        x = self.proj(x)
        x = self.proj_drop(x)

        if return_attention:
            return x, attn
        else:
            return x
