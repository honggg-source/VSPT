"""
Basic OSTrack model.
"""
import math
import os
from collections import OrderedDict
from typing import List

import torch
from torch import nn
from torch.nn.modules.transformer import _get_clones
from timm.models.layers import to_2tuple

from lib.models.layers.head import build_box_head
from lib.models.ostrack.vit import vit_base_patch16_224
from lib.models.ostrack.vit_ce import vit_large_patch16_224_ce, vit_base_patch16_224_ce
from lib.models.ostrack.vit_peft import vit_base_patch16_224_prompt
from lib.utils.box_ops import box_xyxy_to_cxcywh


class OSTrack(nn.Module):
    """ This is the base class for OSTrack """

    def __init__(self, transformer, box_head, aux_loss=False, head_type="CORNER"):
        """ Initializes the model.
        Parameters:
            transformer: torch module of the transformer architecture.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.backbone = transformer
        self.box_head = box_head

        self.aux_loss = aux_loss
        self.head_type = head_type
        if head_type == "CORNER" or head_type == "CENTER":
            self.feat_sz_s = int(box_head.feat_sz)#16
            self.feat_len_s = int(box_head.feat_sz ** 2)#256

        if self.aux_loss:
            self.box_head = _get_clones(self.box_head, 6)

    def forward(self, template: torch.Tensor,
                search: torch.Tensor,
                ce_template_mask=None,
                ce_keep_rate=None,
                return_last_attn=False,
                ):
        x, aux_dict = self.backbone(z=template, x=search,
                                    ce_template_mask=ce_template_mask,
                                    ce_keep_rate=ce_keep_rate,
                                    return_last_attn=return_last_attn, )

        # Forward head
        feat_last = x
        if isinstance(x, list):
            feat_last = x[-1]
        out = self.forward_head(feat_last, None)

        out.update(aux_dict)
        out['backbone_feat'] = x
        return out

    def forward_head(self, cat_feature, gt_score_map=None):
        """
        cat_feature: output embeddings of the backbone, it can be (HW1+HW2, B, C) or (HW2, B, C)
        """
        enc_opt = cat_feature[:, -self.feat_len_s:]  # encoder output for the search region (B, HW, C)
        opt = (enc_opt.unsqueeze(-1)).permute((0, 3, 2, 1)).contiguous()
        bs, Nq, C, HW = opt.size()
        opt_feat = opt.view(-1, C, self.feat_sz_s, self.feat_sz_s)

        if self.head_type == "CORNER":
            # run the corner head
            pred_box, score_map = self.box_head(opt_feat, True)
            outputs_coord = box_xyxy_to_cxcywh(pred_box)
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map,
                   }
            return out

        elif self.head_type == "CENTER":
            # run the center head
            score_map_ctr, bbox, size_map, offset_map = self.box_head(opt_feat, gt_score_map)
            # outputs_coord = box_xyxy_to_cxcywh(bbox)
            outputs_coord = bbox
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map_ctr,
                   'size_map': size_map,
                   'offset_map': offset_map}
            return out
        else:
            raise NotImplementedError


def build_ostrack(cfg, training=True):
    current_dir = os.path.dirname(os.path.abspath(__file__))  # This is your Project Root
    pretrained_path = os.path.join(current_dir, 'pretrained_models')
    if cfg.MODEL.PRETRAIN_FILE and ('OSTrack' not in cfg.MODEL.PRETRAIN_FILE) and training:
        pretrained = os.path.join(pretrained_path, cfg.MODEL.PRETRAIN_FILE)
    else:
        pretrained = ''

    if cfg.MODEL.BACKBONE.TYPE == 'vit_base_patch16_224_peft':
        backbone = vit_base_patch16_224_prompt(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                                               search_size=to_2tuple(cfg.DATA.SEARCH.SIZE),
                                               template_size=to_2tuple(cfg.DATA.TEMPLATE.SIZE),
                                               new_patch_size=cfg.MODEL.BACKBONE.STRIDE,
                                               prompt_type=cfg.TRAIN.PROMPT.TYPE)
        hidden_dim = backbone.embed_dim
        patch_start_index = 1

    elif cfg.MODEL.BACKBONE.TYPE == 'vit_base_patch16_224':
        backbone = vit_base_patch16_224(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE)
        hidden_dim = backbone.embed_dim
        patch_start_index = 1

    elif cfg.MODEL.BACKBONE.TYPE == 'vit_base_patch16_224_ce':
        backbone = vit_base_patch16_224_ce(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                                           ce_loc=cfg.MODEL.BACKBONE.CE_LOC,
                                           ce_keep_ratio=cfg.MODEL.BACKBONE.CE_KEEP_RATIO,
                                           )
        hidden_dim = backbone.embed_dim
        patch_start_index = 1

    elif cfg.MODEL.BACKBONE.TYPE == 'vit_large_patch16_224_ce':
        backbone = vit_large_patch16_224_ce(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                                            ce_loc=cfg.MODEL.BACKBONE.CE_LOC,
                                            ce_keep_ratio=cfg.MODEL.BACKBONE.CE_KEEP_RATIO,
                                            )

        hidden_dim = backbone.embed_dim#768
        patch_start_index = 1
    else:
        raise NotImplementedError

    # backbone.finetune_track(cfg=cfg, patch_start_index=patch_start_index)

    box_head = build_box_head(cfg, hidden_dim)

    model = OSTrack(
        backbone,
        box_head,
        aux_loss=False,
        head_type=cfg.MODEL.HEAD.TYPE,#center
    )

    if 'OSTrack' in cfg.MODEL.PRETRAIN_FILE and training:

        checkpoint1 = torch.load(cfg.MODEL.PRETRAIN_FILE_MAE, map_location="cpu")
        pretrained_dict = checkpoint1.get("model", checkpoint1)

        new_pretrained_dict = OrderedDict()
        for k, v in pretrained_dict.items():
            # 只选择键名以 'blocks3.' 开头的权重
            if k.startswith(('blocks1.', 'blocks2.', 'blocks3.', 'patch_embed4', 'stage2_output_decode')):
                new_pretrained_dict["backbone." + k] = v
            if 'patch_embed1' in k:
                # .replace() 只会替换第一个匹配项，所以是安全的
                key_x = k.replace('patch_embed1', 'patch_embed1x')
                key_z = k.replace('patch_embed1', 'patch_embed1z')
                new_pretrained_dict['backbone.' + key_x] = v
                new_pretrained_dict['backbone.' + key_z] = v
                print(f"复制 {k} -> {key_x} & {key_z}")

            elif 'patch_embed2' in k:
                key_x = k.replace('patch_embed2', 'patch_embed2x')
                key_z = k.replace('patch_embed2', 'patch_embed2z')
                new_pretrained_dict['backbone.' + key_x] = v
                new_pretrained_dict['backbone.' + key_z] = v
                print(f"复制 {k} -> {key_x} & {key_z}")

            elif 'patch_embed3' in k:
                key_x = k.replace('patch_embed3', 'patch_embed3x')
                key_z = k.replace('patch_embed3', 'patch_embed3z')
                new_pretrained_dict['backbone.' + key_x] = v
                new_pretrained_dict['backbone.' + key_z] = v
                print(f"复制 {k} -> {key_x} & {key_z}")
        # for k, v in pretrained_dict.items():
        #     if k == "pos_embed":
        #         pretrained_pos_embed = v
        #         embed_dim = pretrained_pos_embed.shape[-1]
        #         cls_pos_embed = torch.randn(1, 1, embed_dim)
        #         new_pos_embed = torch.cat((pretrained_pos_embed, cls_pos_embed), dim=1)
        #         new_pretrained_dict["backbone.pos_embed"] = new_pos_embed
        #     else:
        #         new_pretrained_dict["backbone." + k] = v
        missing_keys1, unexpected_keys1 = model.load_state_dict(new_pretrained_dict, strict=False)
        # 输出MAE预训练权重加载信息
        print("\n=== MAE预训练权重加载信息 ===")
        print(f"MAE预训练文件路径: {cfg.MODEL.PRETRAIN_FILE_MAE}")
        print(f"MAE权重字典键数量: {len(pretrained_dict)}")
        print(f"实际加载的MAE权重键数量: {len(new_pretrained_dict)}")

        # 输出实际加载的MAE权重键名
        print("\n实际加载的MAE权重键名:")
        for key in new_pretrained_dict.keys():
            print(f"  {key}")

        # 统计MAE权重参数数量
        mae_total_params = sum(p.numel() for p in new_pretrained_dict.values())
        print(f"\nMAE总权重参数数量: {mae_total_params:,}")

        print("=== MAE权重加载完成 ===\n")



        checkpoint = torch.load(cfg.MODEL.PRETRAIN_FILE, map_location="cpu")
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint["net"], strict=False)

        print('Load pretrained model from: ' + cfg.MODEL.PRETRAIN_FILE)
        print(f"missing_keys: {missing_keys}")
        print(f"unexpected_keys: {unexpected_keys}")
        print('Load pretrained_mae model from: ' + cfg.MODEL.PRETRAIN_FILE_MAE)
        print(f"missing_keys: {missing_keys1}")
        print(f"unexpected_keys: {unexpected_keys1}")

    return model
