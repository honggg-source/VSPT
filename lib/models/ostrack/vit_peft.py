from functools import partial

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import Mlp, DropPath, trunc_normal_, lecun_normal_
from .utils import combine_tokens, token2feature, feature2token
from lib.models.layers.patch_embed import PatchEmbed
from .vit import VisionTransformer
from .adapter_modules import CNN, Injector,Extractor
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
import pdb

from .vision_transformer import PatchEmbed1, Block, CBlock, PatchEmbed_F

# from .pos_embed import get_2d_sincos_pos_embed
# import torchvision.transforms.functional as F1
import matplotlib.pyplot as plt




# set recommended archs
# infmae_vit_base_patch16 = infmae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, return_attention=False):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        # x_1 = x[:, 64:, :].cpu()
        # x_1 = x_1.view(1, 256, 768).mean(dim=2).squeeze().view(16, 16)
        # # 绘制图像并设置值范围和插值方法
        # plt.imshow(x_1, cmap='viridis', interpolation='bilinear')
        # # 添加颜色条
        # plt.colorbar()
        # # 显示图像
        # plt.show()

        if return_attention:
            return x, attn
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, return_attention=False):
        if return_attention:
            feat, attn = self.attn(self.norm1(x), True)
            x = x + self.drop_path(feat)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x, attn
        else:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x


class VisionTransformerP(VisionTransformer):
    """ Vision Transformer
    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929
    Includes distillation token & head support for `DeiT: Data-efficient Image Transformers`
        - https://arxiv.org/abs/2012.12877
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True,  distilled=False,injector_indexes=None,extractor_indexes=None,maeinjector_indexes=None,maeextractor_indexes=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None, with_cffn=True, cffn_ratio=0.25,
                 act_layer=None, weight_init='',search_size=None, template_size=None, prompt_type=None,init_values=0.,
                 new_patch_size=None,  conv_inplane=64,attn_drop=0.,with_cp=False,drop=0., drop_path=0.):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            distilled (bool): model includes a distillation token and head as in DeiT models
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
            weight_init: (str): weight init scheme
        """
        super().__init__()

        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        # encoder
        # self.patch_embed1 = PatchEmbed_F(
        #             img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)

        self.patch_embed1x = PatchEmbed1(
            img_size=256, patch_size=4, in_chans=in_chans, embed_dim=256)
        self.patch_embed2x = PatchEmbed1(
            img_size=64, patch_size=2, in_chans=256, embed_dim=384)
        self.patch_embed3x = PatchEmbed1(
            img_size=32, patch_size=2, in_chans=384, embed_dim=768)
        # self.avg_pool = torch.nn.functional.avg_pool2d(encoder_features, kernel_size=2, stride=2)

        self.patch_embed1z = PatchEmbed1(
            img_size=128, patch_size=4, in_chans=in_chans, embed_dim=256)
        self.patch_embed2z = PatchEmbed1(
            img_size=32, patch_size=2, in_chans=256, embed_dim=384)
        self.patch_embed3z = PatchEmbed1(
            img_size=16, patch_size=2, in_chans=384, embed_dim=768)

        self.patch_embed4 = nn.Linear(768, 768)
        self.stage1_output_decode = nn.Conv2d(256, 768, 2, stride=2)
        # self.stage1_output_decode = nn.Conv2d(256, 768, 4, stride=4)
        self.stage2_output_decode = nn.Conv2d(384, 768, 2, stride=2)
        self.stage3x_output_decode = nn.Linear(256,64)
        self.stage3z_output_decode = nn.Linear(64,16)

        # num_patches = self.patch_embed3.num_patches
        # self.pos_embed = nn.Parameter(torch.zeros(1, num_patches+1, 768), requires_grad=False)
        self.blocks1 = nn.ModuleList([
            CBlock(
                dim=256, num_heads=num_heads, mlp_ratio=4, qkv_bias=True, qk_scale=None,
                norm_layer=norm_layer)
            for i in range(2)])
        self.blocks2 = nn.ModuleList([
            CBlock(
                dim=384, num_heads=num_heads, mlp_ratio=4, qkv_bias=True, qk_scale=None,
                norm_layer=norm_layer)
            for i in range(2)])

        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        # num_patches = self.patch_embed.num_patches 224,16,3,768

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        # self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        H, W = search_size
        new_P_H, new_P_W = H // new_patch_size, W // new_patch_size
        self.num_patches_search=new_P_H * new_P_W
        H, W = template_size
        new_P_H, new_P_W = H // new_patch_size, W // new_patch_size
        self.num_patches_template=new_P_H * new_P_W
        """add here, no need use backbone.finetune_track """
        self.pos_embed_z = nn.Parameter(torch.zeros(1, self.num_patches_template, embed_dim))
        self.pos_embed_x = nn.Parameter(torch.zeros(1, self.num_patches_search, embed_dim))

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)])

        self.blocks3 = nn.Sequential(*[
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
            for i in range(11)])
        self.norm = norm_layer(embed_dim)

        self.init_weights(weight_init)
        """Adapter"""
        self.Adapter_level_embed = nn.Parameter(torch.zeros(3, embed_dim))#embed_dim=768
        self.Adapter_spm = CNN(inplanes=conv_inplane,#conv_inplace=64
                       embed_dim=embed_dim)
        self.Adapter_injector_indexes = injector_indexes
        self.Adapter_extractor_indexes = extractor_indexes
        # self.Adapter_maeinjector_indexes = maeinjector_indexes
        # self.Adapter_maeextractor_indexes = maeextractor_indexes
        # self.mae_outputs = []
        self.Adapter_injector_dict = nn.ModuleDict({
            index: Injector(dim=embed_dim,  num_heads=num_heads, init_values=init_values, attn_drop=attn_drop,norm_layer=norm_layer, with_cp=with_cp
                             ,cffn_ratio = cffn_ratio, drop = drop, drop_path = drop_path)
            for index in self.Adapter_injector_indexes
                         # + self.Adapter_maeinjector_indexes
        })
        self.Adapter_injector_dict1 = nn.ModuleDict({
            index: Injector(dim=embed_dim, num_heads=num_heads, init_values=init_values, attn_drop=attn_drop,
                            norm_layer=norm_layer, with_cp=with_cp
                            , cffn_ratio=cffn_ratio, drop=drop, drop_path=drop_path)
            for index in self.Adapter_injector_indexes
            # self.Adapter_maeinjector_indexes
        })
        self.Adapter_extractor_dict = nn.ModuleDict({
            index: Extractor(dim=embed_dim,  num_heads=num_heads, norm_layer=norm_layer,  with_cffn=with_cffn,
                                cffn_ratio=cffn_ratio, drop=drop, drop_path=drop_path, with_cp=with_cp)
            for index in self.Adapter_extractor_indexes
        })
        self.Adapter_extractor_dict1 = nn.ModuleDict({
            index: Extractor(dim=embed_dim, num_heads=num_heads, norm_layer=norm_layer, with_cffn=with_cffn,
                             cffn_ratio=cffn_ratio, drop=drop, drop_path=drop_path, with_cp=with_cp)
            for index in self.Adapter_extractor_indexes
        })

        self.Adapter_spm.apply(self._init_weights)
        self.Adapter_extractor_dict.apply(self._init_weights)
        self.Adapter_injector_dict.apply(self._init_weights)


    def _add_level_embed(self, c2, c3, c4):
        c2 = c2 + self.Adapter_level_embed[0]
        c3 = c3 + self.Adapter_level_embed[1]
        c4 = c4 + self.Adapter_level_embed[2]
        return c2, c3, c4



    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N = x.shape[0]
        if (x.shape[1] == 256):
            L = self.patch_embed3x.num_patches
        else:
            L = self.patch_embed3z.num_patches

        # patches = x.unfold(2, 16, 16).unfold(3, 16, 16)
        #
        # # 计算每个块的方差
        # variances = torch.var(patches, dim=(4, 5), unbiased=False)
        #
        # # 将方差展平为一维数组
        # variances = variances[:, 0, :, :].view(variances.size(0), -1)
        #
        # ids_shuffle = torch.argsort(torch.tensor(variances))
        #
        # # sort noise for each sample
        # ids_restore = torch.argsort(ids_shuffle, dim=1)

        x_ = torch.mean(x, dim=-1)
        # ori_img = x_.reshape(-1, 14, 14).cpu().squeeze().detach().numpy()
        # plt.imshow(ori_img, cmap='gray')
        # plt.show()

        ids_shuffle = torch.argsort(x_, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, ::4]  # in every 7 tokens take one token
        # x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, ::4] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return ids_keep, mask, ids_restore

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm) or isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward_features(self, z, x, mask_z=None, mask_x=None,
                         return_last_attn=False
                         ):
        B, H, W = x.shape[0], x.shape[2], x.shape[3]

        # SPM forward
        z_c1, z_c2, z_c3, z_c4 = self.Adapter_spm(z)#[768,32,32] [256,768] [64,768] [16,768]
        z_c2, z_c3, z_c4 = self._add_level_embed(z_c2, z_c3, z_c4)#[256,768]  [64,768] [16,768]
        z_c = torch.cat([z_c2, z_c3, z_c4], dim=1)#16,336,768
        x_c1, x_c2, x_c3, x_c4 = self.Adapter_spm(x)#[768,64,64] [1024,768] [256,768] [64,768]
        x_c2, x_c3, x_c4 = self._add_level_embed(x_c2, x_c3, x_c4)#[1024,768] [256,768] [64,768]
        x_c = torch.cat([x_c2, x_c3, x_c4], dim=1)#16,1344,768
        c = combine_tokens(z_c, x_c, mode=self.cat_mode)#16,1680,768

        x_o = x
        z_o = z
        z , H_z, W_z= self.patch_embed(z)#16,64,768
        x , H_x, W_x= self.patch_embed(x)#16,256,768
        bs_z, n_z, dim_z = z.shape
        bs_x, n_x, dim_x = x.shape

        # attention mask handling
        # B, H, W
        if mask_z is not None and mask_x is not None:
            mask_z = F.interpolate(mask_z[None].float(), scale_factor=1. / self.patch_size).to(torch.bool)[0]
            mask_z = mask_z.flatten(1).unsqueeze(-1)

            mask_x = F.interpolate(mask_x[None].float(), scale_factor=1. / self.patch_size).to(torch.bool)[0]
            mask_x = mask_x.flatten(1).unsqueeze(-1)

            mask_x = combine_tokens(mask_z, mask_x, mode=self.cat_mode)
            mask_x = mask_x.squeeze(-1)


# ####################################
#         # embed patches
        ids_keep_x, mask_x1, ids_restore_x = self.random_masking(x, 0.75)
        ids_keep_z, mask_z1, ids_restore_z = self.random_masking(z, 0.75)

        mask_for_patch1_x = mask_x1.reshape(-1, 16, 16).unsqueeze(-1).repeat(1, 1, 1, 16).reshape(-1, 16, 16, 4, 4).permute(
            0, 1, 3, 2, 4).reshape(x.shape[0], 64, 64).unsqueeze(1)#1,64,64
        mask_for_patch2_x = mask_x1.reshape(-1, 16, 16).unsqueeze(-1).repeat(1, 1, 1, 4).reshape(-1, 16, 16, 2, 2).permute(
            0, 1, 3, 2, 4).reshape(x.shape[0], 32, 32).unsqueeze(1)#1,32,32

        mask_for_patch1_z = mask_z1.reshape(-1, 8, 8).unsqueeze(-1).repeat(1, 1, 1, 16).reshape(-1, 8, 8, 4, 4).permute(
            0, 1, 3, 2, 4).reshape(z.shape[0], 32, 32).unsqueeze(1)#1,32,32
        mask_for_patch2_z = mask_z1.reshape(-1, 8, 8).unsqueeze(-1).repeat(1, 1, 1, 4).reshape(-1, 8, 8, 2, 2).permute(
            0, 1, 3, 2, 4).reshape(z.shape[0], 16, 16).unsqueeze(1)#1,16,16

        x1 = self.patch_embed1x(x_o)#256,64,64
        z1 = self.patch_embed1z(z_o)#256,32,32
        for blk in self.blocks1:
            x1 = blk(x1, 1 - mask_for_patch1_x)#F1_x:8,256,64,64
        stage1_embed_x = self.stage1_output_decode(x1).flatten(2).permute(0, 2, 1)#8,1024,768

        for blk in self.blocks1:
            z1 = blk(z1, 1 - mask_for_patch1_z)#F1_z:8,256,32,32
        # stage1_embed_z = self.stage1_output_decode(z1).flatten(2).permute(0, 2, 1)#8,256,768
        stage1_embed_z = self.stage1_output_decode(z1).flatten(2).permute(0, 2, 1)

        x1 = self.patch_embed2x(x1)#8,384,32,32
        z1 = self.patch_embed2z(z1)#8,384,16,16
        for blk in self.blocks2:
            x1 = blk(x1, 1 - mask_for_patch2_x)#F2_x:8,384,32,32
        stage2_embed_x = self.stage2_output_decode(x1).flatten(2).permute(0, 2, 1)#8,256,768
        for blk in self.blocks2:
            z1 = blk(z1, 1 - mask_for_patch2_z)#F2_z:8,384,16,16
        stage2_embed_z = self.stage2_output_decode(z1).flatten(2).permute(0, 2, 1)#8,64,768

        x1 = self.patch_embed3x(x1)#8,768,16,16
        x1 = x1.flatten(2).permute(0, 2, 1)  # 8,256,768
        x1 = self.patch_embed4(x1)  # 8,256,768
        # stage3_embed_x = self.stage3x_output_decode(x1.transpose(1,2)).transpose(1,2)

        # x_mae = torch.cat([stage1_embed_x, stage2_embed_x, stage3_embed_x], dim=1)#8,768,768

        z1 = self.patch_embed3z(z1)#8,768,8,8
        z1 = z1.flatten(2).permute(0, 2, 1)  # 8,64,768
        z1 = self.patch_embed4(z1)  # 8,64,768
        # stage3_embed_z = self.stage3z_output_decode(z1.transpose(1,2)).transpose(1,2)

        # z_mae = torch.cat([stage1_embed_z, stage2_embed_z,stage3_embed_z], dim=1)#8,192,768
        # c_mae = combine_tokens(z_mae, x_mae, mode=self.cat_mode)#8,960,768

        if self.add_cls_token:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            cls_tokens = cls_tokens + self.cls_pos_embed

        z1 += self.pos_embed_z
        x1 += self.pos_embed_x

        if self.add_sep_seg:
            x1 += self.search_segment_pos_embed
            z1 += self.template_segment_pos_embed

            # add pos embed w/o cls token
            # x = x + self.pos_embed
            # x = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, x.shape[-1]))  # (2,49,768)
            # stage1_embed = torch.gather(stage1_embed, dim=1,
            #                             index=ids_keep.unsqueeze(-1).repeat(1, 1, stage1_embed.shape[-1]))
            # stage2_embed = torch.gather(stage2_embed, dim=1,
            #                             index=ids_keep.unsqueeze(-1).repeat(1, 1, stage2_embed.shape[-1]))
            # stage1_embed stage2_embed:(2,49,768)

        # x1 = combine_tokens(z1, x1, mode=self.cat_mode)
        if self.add_cls_token:
            x1 = torch.cat([cls_tokens, x1], dim=1)

        x1 = self.pos_drop(x1)

        if self.add_cls_token:
            z1 = torch.cat([cls_tokens, z1], dim=1)

        z1 = self.pos_drop(z1)

        # apply Transformer blocks
        for bid, blk in enumerate(self.blocks3):
            x1 = blk(x1)
            # if bid == 1:
            #     mae_1 = x1
            # elif bid == 4:
            #     mae_4 = x1
            # elif bid == 7:
            #     mae_7 = x1
            # elif bid == 10:
            #     mae_10 = x1
        # stage3_embed_x = self.stage3x_output_decode(x1.transpose(1,2)).transpose(1,2)
        stage3_embed_x = x1.view(x1.size(0),64, 4, x1.size(2)).mean(dim=2)#8,64,768

        x_mae = torch.cat([stage1_embed_x, stage2_embed_x, stage3_embed_x], dim=1)#8,1344,768

        for bid, blk in enumerate(self.blocks3):
            z1 = blk(z1)

        # stage3_embed_z = self.stage3z_output_decode(z1.transpose(1,2)).transpose(1,2)
        stage3_embed_z = z1.view(z1.size(0), 16, 4, z1.size(2)).mean(dim=2)  # 8,16,768
        z_mae = torch.cat([stage1_embed_z, stage2_embed_z,stage3_embed_z], dim=1)#8,336,768
        c_mae = combine_tokens(z_mae, x_mae, mode=self.cat_mode)#8,1680,768

            # self.mae_outputs.append(x1)
            # x = x + stage1_embed + stage2_embed  # (2,49,768)
            # x = self.norm(x)
# ##########################################

        if self.add_cls_token:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            cls_tokens = cls_tokens + self.cls_pos_embed

        z += self.pos_embed_z
        x += self.pos_embed_x

        if self.add_sep_seg:
            x += self.search_segment_pos_embed
            z += self.template_segment_pos_embed

        x = combine_tokens(z, x, mode=self.cat_mode)
        if self.add_cls_token:
            x = torch.cat([cls_tokens, x], dim=1)

        x = self.pos_drop(x)

        for bid, blk in enumerate(self.blocks):
            #
            if str(bid) in self.Adapter_injector_indexes:
                x = self.Adapter_injector_dict[str(bid)](x,c,H_z,W_z,H_x,W_x)
                x = self.Adapter_injector_dict1[str(bid)](x,c_mae,H_z,W_z,H_x,W_x)

            # if str(bid) in self.Adapter_maeinjector_indexes:
            #     if bid == 1:
            #         x = self.Adapter_injector_dict1[str(bid)](x,mae_1,H_z,W_z,H_x,W_x)
            #     elif bid == 4:
            #         x = self.Adapter_injector_dict1[str(bid)](x,mae_4,H_z,W_z,H_x,W_x)
            #     elif bid == 7:
            #         x = self.Adapter_injector_dict1[str(bid)](x,mae_7,H_z,W_z,H_x,W_x)
            #     elif bid ==10:
            #         x = self.Adapter_injector_dict1[str(bid)](x, mae_10, H_z, W_z, H_x, W_x)

            x = blk(x)

            if str(bid) in self.Adapter_extractor_indexes:
                c = self.Adapter_extractor_dict[str(bid)](c,x,H_z,W_z,H_x,W_x)
                c_mae = self.Adapter_extractor_dict1[str(bid)](c_mae, x, H_z, W_z, H_x, W_x)

            # if str(bid) in self.Adapter_maeextractor_indexes:
            #     c = self.Adapter_extractor_dict[str(bid)](self.mae_outputs[bid],x,H_z,W_z,H_x,W_x)

            if bid == 11:
                c_z = c[:, :336, :]
                c_x = c[:, 336:, :]

                c_z2 = c_z[:, :256, :]
                c_z3 = c_z[:, 256:320, :]
                c_z4 = c_z[:, 320:, :]
                c_z2 = c_z2.transpose(1, 2).view(bs_z, dim_z, H_z * 2, W_z * 2)
                c_z3 = c_z3.transpose(1, 2).view(bs_z, dim_z, H_z, W_z)
                c_z4 = c_z4.transpose(1, 2).view(bs_z, dim_z, H_z // 2, W_z // 2)

                c_x2 = c_x[:, :1024, :]
                c_x3 = c_x[:, 1024:1280, :]
                c_x4 = c_x[:, 1280:, :]
                c_x2 = c_x2.transpose(1, 2).view(bs_x, dim_x, H_x * 2, W_x * 2)
                c_x3 = c_x3.transpose(1, 2).view(bs_x, dim_x, H_x, W_x)
                c_x4 = c_x4.transpose(1, 2).view(bs_x, dim_x, H_x // 2, W_x // 2)

                c_z2 = F.interpolate(c_z2, scale_factor=0.5, mode='bilinear', align_corners=False)
                c_z4 = F.interpolate(c_z4, scale_factor=2, mode='bilinear', align_corners=False)
                c_z = (c_z2 + c_z3 + c_z4).permute(0, 2, 3, 1).reshape(bs_z, 64, 768)

                c_x2 = F.interpolate(c_x2, scale_factor=0.5, mode='bilinear', align_corners=False)
                c_x4 = F.interpolate(c_x4, scale_factor=2, mode='bilinear', align_corners=False)
                c_x = (c_x2 + c_x3 + c_x4).permute(0, 2, 3, 1).reshape(bs_x, 256, 768)

                c = torch.cat((c_z, c_x), dim=1)
                x = x + c

        x = self.norm(x)
        aux_dict = {"attn": None}
        return x, aux_dict

    def forward(self, z, x, ce_template_mask=None, ce_keep_rate=None,
                tnc_keep_rate=None,
                return_last_attn=False):

        x, aux_dict = self.forward_features(z, x)

        return x, aux_dict


def _create_vision_transformer(pretrained=False, **kwargs):
    model = VisionTransformerP(**kwargs)

    if pretrained:
        if 'npz' in pretrained:
            model.load_pretrained(pretrained, prefix='')
        else:
            checkpoint = torch.load(pretrained, map_location="cpu")
            missing_keys, unexpected_keys = model.load_state_dict(checkpoint["net"], strict=False)
            print('Load pretrained OSTrack without CE from: ' + pretrained)
            print(f"missing_keys: {missing_keys}")
            print(f"unexpected_keys: {unexpected_keys}")

    return model


def vit_base_patch16_224_prompt(pretrained=False, **kwargs):
    """
    ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    """
    # model_kwargs = dict(
    #     patch_size=16, embed_dim=768, depth=12, num_heads=12, r=2,
    #     lora_before_blocks=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11'],
    #     add_after_blocks=['6', '7', '8', '9', '10', '11'], **kwargs)

    # injector_indexes=['0', '3', '6', '9']
    # extractor_indexes = ['2', '5', '8', '11']

    model_kwargs = dict(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        img_size=[224, 56, 28],
        injector_indexes=['0', '3', '6', '9'],
        extractor_indexes=['2', '5', '8', '11'],
        # maeinjector_indexes=['1', '4', '7','10'],
        # maeextractor_indexes=['3', '6', '9'],
        **kwargs)
    model = _create_vision_transformer(pretrained=pretrained, **model_kwargs)
    return model
