# introduce attention or DGCNN in grouping
from typing import Union, Optional
import math
import random
from functools import partial
import pdb
import numpy as np
import torch
import torch.nn as nn

from utils import misc
from utils.checkpoint import get_missing_parameters_message, get_unexpected_parameters_message
from utils.logger import *
from utils.order import get_z_order
from utils.order import get_hilbert_order
from timm.models.layers import trunc_normal_
from timm.models.layers import DropPath

from extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2
from mamba_ssm.modules.mamba_simple import Mamba

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

from knn_cuda import KNN
from .block import Block
from .build import MODELS


class Encoder(nn.Module):  ## Embedding module
    def __init__(self, encoder_channel, in_channel=3):
        super().__init__()
        self.encoder_channel = encoder_channel
        self.first_conv = nn.Sequential(
            nn.Conv1d(in_channel, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1)
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, self.encoder_channel, 1)
        )

    def forward(self, point_groups):
        '''
            point_groups : B G N 3
            -----------------
            feature_global : B G C
        '''
        bs, g, n, num_D = point_groups.shape
        point_groups = point_groups.reshape(bs * g, n, num_D)
        # encoder
        feature = self.first_conv(point_groups.transpose(2, 1))  # BG 256 n
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]  # BG 256 1
        feature = torch.cat([feature_global.expand(-1, -1, n), feature], dim=1)  # BG 512 n
        feature = self.second_conv(feature)  # BG 1024 n
        feature_global = torch.max(feature, dim=2, keepdim=False)[0]  # BG 1024
        return feature_global.reshape(bs, g, self.encoder_channel)


class Group(nn.Module):  # FPS + KNN
    def __init__(self, num_group, group_size):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size
        self.knn = KNN(k=self.group_size, transpose_mode=True)

    def forward(self, xyz, points=None):
        '''
            input: B N 3
            points: B, N, D
            ---------------------------
            output: B G M 3
            center : B G 3
        '''
        batch_size, num_points, _ = xyz.shape
        # fps the centers out
        center = misc.fps(xyz, self.num_group)  # B G 3
        # knn to get the neighborhood
        # import ipdb; ipdb.set_trace()
        # idx = knn_query(xyz, center, self.group_size)  # B G M
        _, idx = self.knn(xyz, center)  # B G M
        assert idx.size(1) == self.num_group
        assert idx.size(2) == self.group_size
        idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)
        neighborhood = xyz.view(batch_size * num_points, -1)[idx, :]
        neighborhood = neighborhood.view(batch_size, self.num_group, self.group_size, 3).contiguous()
        # normalize
        neighborhood = neighborhood - center.unsqueeze(2)

        if points is not None:
            grouped_points = points.view(batch_size * num_points, -1)[idx, :]
            grouped_points = grouped_points.view(batch_size, self.num_group, self.group_size, -1).contiguous()
            new_points = torch.cat([neighborhood, grouped_points], dim=-1)

            return neighborhood, new_points, center
        else:
            return neighborhood, center


# https://github.com/huggingface/transformers/blob/c28d04e9e252a1a099944e325685f14d242ecdcd/src/transformers/models/gpt2/modeling_gpt2.py#L454
def _init_weights(
        module,
        n_layer,
        initializer_range=0.02,  # Now only used for embedding layer.
        rescale_prenorm_residual=True,
        n_residuals_per_layer=1,  # Change to 2 if we have MLP
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


def create_block(
        d_model,
        ssm_cfg=None,
        norm_epsilon=1e-5,
        rms_norm=False,
        residual_in_fp32=False,
        fused_add_norm=False,
        layer_idx=None,
        drop_path=0.,
        device=None,
        dtype=None,
):
    if ssm_cfg is None:
        ssm_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}

    mixer_cls = partial(Mamba, layer_idx=layer_idx, **ssm_cfg, **factory_kwargs)
    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    )
    block = Block(
        d_model,
        mixer_cls,
        norm_cls=norm_cls,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
        drop_path=drop_path,
    )
    block.layer_idx = layer_idx
    return block


class MixerModel(nn.Module):
    def __init__(
            self,
            d_model: int,
            n_layer: int,
            ssm_cfg=None,
            norm_epsilon: float = 1e-5,
            rms_norm: bool = False,
            initializer_cfg=None,
            fused_add_norm=False,
            residual_in_fp32=False,
            drop_out_in_block: int = 0.,
            drop_path: int = 0.1,
            device=None,
            dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32

        # self.embedding = nn.Embedding(vocab_size, d_model, **factory_kwargs)

        # We change the order of residual and layer norm:
        # Instead of LN -> Attn / MLP -> Add, we do:
        # Add -> LN -> Attn / MLP / Mixer, returning both the residual branch (output of Add) and
        # the main branch (output of MLP / Mixer). The model definition is unchanged.
        # This is for performance reason: we can fuse add + layer_norm.
        self.fused_add_norm = fused_add_norm
        if self.fused_add_norm:
            if layer_norm_fn is None or rms_norm_fn is None:
                raise ImportError("Failed to import Triton LayerNorm / RMSNorm kernels")

        self.layers = nn.ModuleList(
            [
                create_block(
                    d_model,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    drop_path=drop_path,
                    **factory_kwargs,
                )
                for i in range(n_layer)
            ]
        )

        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            d_model, eps=norm_epsilon, **factory_kwargs
        )

        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.drop_out_in_block = nn.Dropout(drop_out_in_block) if drop_out_in_block > 0. else nn.Identity()

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.layers)
        }

    def forward(self, input_ids, pos, inference_params=None):
        hidden_states = input_ids  # + pos
        residual = None
        hidden_states = hidden_states + pos
        for layer in self.layers:                   # hidden_states: all 32, 192, 384
            hidden_states, residual = layer(
                hidden_states, residual, inference_params=inference_params
            )
            hidden_states = self.drop_out_in_block(hidden_states)
        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
            hidden_states = fused_add_norm_fn(
                hidden_states,
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )

        return hidden_states


class Normalize(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        norm = torch.norm(x, p=2, dim=self.dim, keepdim=True)
        return x / norm
    

class CalImportance(nn.Module):  ## Embedding module
    def __init__(self, encoder_channel, detach_score_prediction):
        super().__init__()

        self.detach_score_prediction = detach_score_prediction

        self.feature_extractor1 = nn.Sequential(
            nn.Conv1d(encoder_channel, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1)
        )
        self.feature_extractor2 = nn.Sequential(
            nn.Conv1d(256, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, encoder_channel, 1)
        )

        self.projection = nn.Sequential(
            nn.Conv1d(encoder_channel, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1),
            Normalize(dim=1)
        )

        self.cal_imp_score = nn.Sequential(
            nn.Conv1d(encoder_channel, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 1, 1)
        )

    def forward(self, patch):
        '''
            patch : B G D
        '''
        patch = patch.transpose(2, 1)       # 32, 384, 64

        patch_feature = self.feature_extractor2(self.feature_extractor1(patch))       # 32, 384, 64

        map_patch_feature = self.projection(patch_feature)

        if self.detach_score_prediction:
            pred_importance = self.cal_imp_score(patch_feature.detach())
        else:
            pred_importance = self.cal_imp_score(patch_feature)
        
        return map_patch_feature.transpose(2, 1), pred_importance.transpose(2, 1)     # 32, 64, 256 & 32, 64, 1


class CalImportanceSimple(nn.Module):  ## Embedding module
    def __init__(self, encoder_channel, detach_score_prediction):
        super().__init__()

        self.detach_score_prediction = detach_score_prediction

        self.projection = nn.Sequential(
            nn.Conv1d(encoder_channel, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1),
            Normalize(dim=1)
        )

        self.cal_imp_score = nn.Sequential(
            nn.Conv1d(encoder_channel, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 1, 1)
        )

    def forward(self, patch):
        '''
            patch : B G D
        '''
        patch = patch.transpose(2, 1)       # 32, 384, 64

        map_patch_feature = self.projection(patch)

        if self.detach_score_prediction:
            pred_importance = self.cal_imp_score(patch.detach())
        else:
            pred_importance = self.cal_imp_score(patch)
        
        return map_patch_feature.transpose(2, 1), pred_importance.transpose(2, 1)     # 32, 64, 256 & 32, 64, 1



class GLR_loss(nn.Module):
    def __init__(self):
        super(GLR_loss, self).__init__()
        self.cossim = nn.CosineSimilarity(dim=-1)

    def forward(self, global_feature, local_feature, mask=None):
        # global feature: [B, 256]
        # local feature: [B, 64, 256]
        local_feature = local_feature.transpose(2, 1)
        B, D, N = local_feature.shape
        device = global_feature.device
        local_feature = local_feature.transpose(1, 0).reshape(D, -1)
        score = torch.matmul(global_feature, local_feature) * 64.     # B, B*N
        score = score.view(B, -1).transpose(1, 0)       # (B*N), B
        label = torch.arange(B).unsqueeze(1).expand(B, N).reshape(-1).to(device)   # B*N

        if mask is None:
            CELoss = nn.CrossEntropyLoss()
            loss = CELoss(score, label)
        else:
            CELoss = nn.CrossEntropyLoss(reduction='none')
            loss = (CELoss(score, label) * mask.reshape(-1)).sum() / mask.sum()
        return loss
    

class SA(nn.Module):  # FPS + KNN
    def __init__(self, num_group, group_size, encoder_channel, in_channel=3):
        super().__init__()

        self.group_divider = Group(num_group, group_size)
        self.encoder = Encoder(encoder_channel, in_channel)

    def forward(self, xyz, points=None):
        '''
            input: B N 3
            ---------------------------
            group_input_tokens: B G D
            center : B G 3
        '''
        if points is not None:
            neighborhood_xyz, neighborhood_feature, center = self.group_divider(xyz, points)
            group_input_tokens = self.encoder(neighborhood_feature)     # B G D: 32, 64, 384
        else:
            neighborhood_xyz, center = self.group_divider(xyz)      # pts: 32, 1024, 3, neighborhood: 32, 64, 32, 3, center: 32, 64, 3
            group_input_tokens = self.encoder(neighborhood_xyz)     # B G D: 32, 64, 384

        return center, group_input_tokens
    

def get_graph_feature(x, k=20):
    # x: B, N, D
    batch_size, num_points, num_dim = x.shape

    _, idx = KNN(k=k, transpose_mode=True)(x, x)       # B G (equals to N here) n_neighbor (k)

    idx_base = torch.arange(0, batch_size, device=x.device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dim).contiguous()       # B, N, n_neighbor, D
    x = x.view(batch_size, num_points, 1, num_dim).repeat(1, 1, k, 1)
    feature = torch.cat((feature-x, x), dim=3)

    return feature             # B, N, n_neighbor, D


class EdgeConv(nn.Module):  # FPS + KNN
    def __init__(self, k, hidden_dimension):
        super().__init__()

        self.k = k
        self.hidden_dimension = hidden_dimension
        self.conv1 = nn.Sequential(
            nn.Conv1d(6, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, hidden_dimension, 1)
        )
        
    def forward(self, xyz):
        '''
            input: B N 3
            ---------------------------
            feature: B N D
            center : B N 3
        '''
        center = xyz
        bs, n_points, _ = xyz.shape
        x = get_graph_feature(xyz, k=self.k)                                                 # B, N, n_neighbor, D

        x = x.reshape(bs * n_points, self.k, -1)
        
        x = self.conv1(x.transpose(2, 1))                                                    # B*N, D1, n_neighbor
        feature_global = x.max(dim=2, keepdim=True)[0]                                       # B*N, D1, 1
        feature = torch.cat([feature_global.repeat(1, 1, self.k), x], dim=1)                 # B*N, D2, n_neighbor
        feature = self.conv2(feature)                                                        # B*N, D3, n_neighbor
        feature = feature.max(dim=2, keepdim=False)[0]                                       # B*N, D3
        feature = feature.reshape(bs, n_points, self.hidden_dimension)                       # B, N, D3

        return center, feature
    

@MODELS.register_module()
class PointMamba(nn.Module):
    def __init__(self, config, **kwargs):
        super(PointMamba, self).__init__()
        self.config = config

        self.trans_dim = config.trans_dim
        self.depth = config.depth
        self.cls_dim = config.cls_dim

        self.group_size = config.group_size
        self.num_group = config.num_group
        self.encoder_dims = config.encoder_dims

        self.use_simple_score_predictor = config.use_simple_score_predictor

        self.type_pooling = config.type_pooling
        self.type_weighting = config.type_weighting
        self.detach_mapping = config.detach_mapping
        self.detach_score_prediction = config.detach_score_prediction
        self.mode_sort = config.mode_sort
        self.mode_group = config.mode_group
        
        if self.mode_group == 'SA':
            self.group_divider = SA(num_group=self.num_group, group_size=self.group_size, encoder_channel=self.encoder_dims)
        elif self.mode_group == 'EdgeConv':
            self.group_divider = EdgeConv(k=self.group_size, hidden_dimension=self.encoder_dims)

        if self.use_simple_score_predictor:
            self.importance_cal_block = CalImportanceSimple(self.encoder_dims, self.detach_score_prediction)
        else:
            self.importance_cal_block = CalImportance(self.encoder_dims, self.detach_score_prediction)

        self.use_cls_token = False if not hasattr(self.config, "use_cls_token") else self.config.use_cls_token
        self.drop_path = 0. if not hasattr(self.config, "drop_path") else self.config.drop_path
        self.rms_norm = False if not hasattr(self.config, "rms_norm") else self.config.rms_norm
        self.drop_out_in_block = 0. if not hasattr(self.config, "drop_out_in_block") else self.config.drop_out_in_block

        if self.use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
            self.cls_pos = nn.Parameter(torch.randn(1, 1, self.trans_dim))
            trunc_normal_(self.cls_token, std=.02)
            trunc_normal_(self.cls_pos, std=.02)

        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )

        self.blocks = MixerModel(d_model=self.trans_dim,
                                 n_layer=self.depth,
                                 rms_norm=self.rms_norm,
                                 drop_out_in_block=self.drop_out_in_block,
                                 drop_path=self.drop_path)

        self.norm = nn.LayerNorm(self.trans_dim)

        self.HEAD_CHANEL = 1
        if self.use_cls_token:
            self.HEAD_CHANEL += 1

        self.cls_head_finetune = nn.Sequential(
            nn.Linear(self.trans_dim * self.HEAD_CHANEL, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, self.cls_dim)
        )

        self.global_projection = nn.Sequential(
            nn.Conv1d(self.trans_dim * self.HEAD_CHANEL, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1),
            Normalize(dim=1)
        )

        self.build_loss_func()

        self.drop_out = nn.Dropout(config.drop_out) if "drop_out" in config else nn.Dropout(0)

    def build_loss_func(self):
        self.loss_ce = nn.CrossEntropyLoss()
        self.loss_glr = GLR_loss()
        self.loss_importance = nn.SmoothL1Loss(reduction='none')

    def get_loss_acc(self, ret, gt, patch_f, global_f, pred_score, cos_sim):
        # CE loss
        loss_CE = self.loss_ce(ret, gt.long())

        # global-to-local align loss
        loss_GLR = self.loss_glr(global_f, patch_f)

        # importance regression loss
        loss_SCORE = self.loss_importance(pred_score.squeeze(-1), cos_sim).sum(dim=-1).mean()

        # pred
        pred = ret.argmax(-1)
        acc = (pred == gt).sum() / float(gt.size(0))
        return loss_CE, loss_GLR, loss_SCORE, acc * 100

    def load_model_from_ckpt(self, bert_ckpt_path):
        if bert_ckpt_path is not None:
            ckpt = torch.load(bert_ckpt_path)
            base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['base_model'].items()}

            for k in list(base_ckpt.keys()):
                # if k.startswith('MAE_encoder'):
                #     base_ckpt[k[len('MAE_encoder.'):]] = base_ckpt[k]
                #     del base_ckpt[k]
                if k.startswith('base_model'):
                    base_ckpt[k[len('base_model.'):]] = base_ckpt[k]
                    del base_ckpt[k]

            incompatible = self.load_state_dict(base_ckpt, strict=False)

            if incompatible.missing_keys:
                print_log('missing_keys', logger='Mamba')
                print_log(
                    get_missing_parameters_message(incompatible.missing_keys),
                    logger='Mamba'
                )
            if incompatible.unexpected_keys:
                print_log('unexpected_keys', logger='Mamba')
                print_log(
                    get_unexpected_parameters_message(incompatible.unexpected_keys),
                    logger='Mamba'
                )

            print_log(f'[Mamba] Successful Loading the ckpt from {bert_ckpt_path}', logger='Mamba')
        else:
            print_log('Training from scratch!!!', logger='Mamba')
            self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, pts):
        device = pts.device                                           # pts: 32, 1024, 3
        center, group_input_tokens = self.group_divider(pts)          # center: B, G, 3, group_input_tokens: B, G, D
        pos = self.pos_embed(center)                                  # B, G, D

        # introduce a local-to-global mapping layer here
        # option: cls on patch, local-to-global alignment on patch, cosine similarity regression on patch
        if self.detach_mapping:
            map_patch_feature, cal_importance = self.importance_cal_block(group_input_tokens.detach())      # map_patch_feature: B, G, D1; cal_importance: B, G, 1
        else:
            map_patch_feature, cal_importance = self.importance_cal_block(group_input_tokens)               # map_patch_feature: B, G, D1; cal_importance: B, G, 1

        # reordering strategy
        if self.mode_sort == 'max':
            # importance: max to min
            importance_order = cal_importance.argsort(dim=1, descending=True)
            sort_cal_importance = cal_importance.gather(dim=1, index=torch.tile(importance_order, (1, 1, cal_importance.shape[-1])))

            group_input_tokens = group_input_tokens.gather(dim=1, index=torch.tile(importance_order, (1, 1, group_input_tokens.shape[-1])))
            pos = pos.gather(dim=1, index=torch.tile(importance_order, (1, 1, pos.shape[-1]))) 

        elif self.mode_sort == 'min':
            # importance: min to max
            importance_order = cal_importance.argsort(dim=1, descending=False)
            sort_cal_importance = cal_importance.gather(dim=1, index=torch.tile(importance_order, (1, 1, cal_importance.shape[-1])))

            group_input_tokens = group_input_tokens.gather(dim=1, index=torch.tile(importance_order, (1, 1, group_input_tokens.shape[-1])))
            pos = pos.gather(dim=1, index=torch.tile(importance_order, (1, 1, pos.shape[-1]))) 
        
        elif self.mode_sort == 'both':
            # importance: max to min and min to max
            importance_order1 = cal_importance.argsort(dim=1, descending=True)
            sort_cal_importance1 = cal_importance.gather(dim=1, index=torch.tile(importance_order1, (1, 1, cal_importance.shape[-1])))
            group_input_tokens1 = group_input_tokens.gather(dim=1, index=torch.tile(importance_order1, (1, 1, group_input_tokens.shape[-1])))
            pos1 = pos.gather(dim=1, index=torch.tile(importance_order1, (1, 1, pos.shape[-1]))) 

            importance_order2 = cal_importance.argsort(dim=1, descending=False)
            sort_cal_importance2 = cal_importance.gather(dim=1, index=torch.tile(importance_order2, (1, 1, cal_importance.shape[-1])))
            group_input_tokens2 = group_input_tokens.gather(dim=1, index=torch.tile(importance_order2, (1, 1, group_input_tokens.shape[-1])))
            pos2 = pos.gather(dim=1, index=torch.tile(importance_order2, (1, 1, pos.shape[-1]))) 

            sort_cal_importance = torch.cat([sort_cal_importance1, sort_cal_importance2], dim=1)
            group_input_tokens = torch.cat([group_input_tokens1, group_input_tokens2], dim=1)
            pos = torch.cat([pos1, pos2], dim=1) 

        else:
            assert self.mode_sort == 'max', 'Unknown sort mode'
 
        x = group_input_tokens
        x = self.drop_out(x)
        x = self.blocks(x, pos)
        x = self.norm(x)                                                                            # B, G, D

        if self.type_pooling == 'ave':
            concat_f = x[:, :].mean(dim=1)                                                          # B, D
        
        elif self.type_pooling == 'max':
            concat_f = x[:, :].max(dim=1)[0]
        
        elif self.type_pooling == 'important':
            if self.type_weighting == 'direct':
                weight = sort_cal_importance.repeat(1, 1, x.shape[-1])
                concat_f = (x * weight).sum(dim=1)
            
            elif self.type_weighting == 'drop_neg':
                weight = sort_cal_importance.clamp(0, 1).repeat(1, 1, x.shape[-1])
                concat_f = (x * weight).sum(dim=1)
            
            elif self.type_weighting == 'sfm':
                weight = nn.Softmax(dim=1)(sort_cal_importance).repeat(1, 1, x.shape[-1])
                concat_f = (x * weight).sum(dim=1)
            
            else:
                # TODO find some new solutions
                concat_f = x[:, :].mean(dim=1)   
        
        else:
            # TODO find some new solutions
            concat_f = x[:, :].mean(dim=1)   

        # global mapping --> 256 dimension
        map_global_feature = self.global_projection(concat_f.unsqueeze(-1)).squeeze(-1)            # B, D1

        # cls
        ret = self.cls_head_finetune(concat_f)                                                     # B, num_class

        return ret, map_patch_feature, map_global_feature, cal_importance


@MODELS.register_module()
class BasePointMamba(nn.Module):
    def __init__(self, config, **kwargs):
        super(BasePointMamba, self).__init__()
        self.config = config

        self.trans_dim = config.trans_dim
        self.depth = config.depth
        self.cls_dim = config.cls_dim
        self.mode_group = config.mode_group
        self.group_size = config.group_size
        self.num_group = config.num_group
        self.encoder_dims = config.encoder_dims

        self.mode_sort = config.mode_sort
        self.mode_encoder = config.mode_encoder
        self.Transformer_encoder_num_heads = config.Transformer_encoder_num_heads

        # self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)
        if self.mode_group == 'SA':
            self.group_divider = SA(num_group=self.num_group, group_size=self.group_size, encoder_channel=self.encoder_dims)
        elif self.mode_group == 'EdgeConv':
            self.group_divider = EdgeConv(k=self.group_size, hidden_dimension=self.encoder_dims)
        elif self.mode_group == 'Attention':
            self.group_divider = Attention(num_group=self.num_group, group_size=self.group_size, hidden_dimension=self.encoder_dims, use_cls_token=self.config.attention_use_cls_token, depth=self.config.attention_depth, drop_path_rate=self.config.attention_drop_path_rate, num_heads=self.config.attention_num_heads)

        self.encoder = Encoder(encoder_channel=self.encoder_dims)

        self.use_cls_token = False if not hasattr(self.config, "use_cls_token") else self.config.use_cls_token
        self.drop_path = 0. if not hasattr(self.config, "drop_path") else self.config.drop_path
        self.rms_norm = False if not hasattr(self.config, "rms_norm") else self.config.rms_norm
        self.drop_out_in_block = 0. if not hasattr(self.config, "drop_out_in_block") else self.config.drop_out_in_block

        if self.use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
            self.cls_pos = nn.Parameter(torch.randn(1, 1, self.trans_dim))
            trunc_normal_(self.cls_token, std=.02)
            trunc_normal_(self.cls_pos, std=.02)

        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )

        if self.mode_encoder == 'Mamba':
            self.blocks = MixerModel(d_model=self.trans_dim,
                                    n_layer=self.depth,
                                    rms_norm=self.rms_norm,
                                    drop_out_in_block=self.drop_out_in_block,
                                    drop_path=self.drop_path)
            
        elif self.mode_encoder == 'Transformer':
            dpr = [x.item() for x in torch.linspace(0, self.config.attention_drop_path_rate, self.depth)]
            self.blocks = TransformerEncoder(
                embed_dim=self.trans_dim,
                depth=self.depth,
                drop_path_rate=dpr,
                num_heads=self.Transformer_encoder_num_heads,
            )

        self.norm = nn.LayerNorm(self.trans_dim)

        self.HEAD_CHANEL = 1
        if self.use_cls_token:
            self.HEAD_CHANEL += 1

        self.cls_head_finetune = nn.Sequential(
            nn.Linear(self.trans_dim * self.HEAD_CHANEL, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, self.cls_dim)
        )

        self.build_loss_func()

        self.drop_out = nn.Dropout(config.drop_out) if "drop_out" in config else nn.Dropout(0)

    def build_loss_func(self):
        self.loss_ce = nn.CrossEntropyLoss()

    def get_loss_acc(self, ret, gt):
        # CE loss
        loss_CE = self.loss_ce(ret, gt.long())

        # pred
        pred = ret.argmax(-1)
        acc = (pred == gt).sum() / float(gt.size(0))
        return loss_CE, acc * 100

    def load_model_from_ckpt(self, bert_ckpt_path):
        if bert_ckpt_path is not None:
            ckpt = torch.load(bert_ckpt_path)
            base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['base_model'].items()}

            for k in list(base_ckpt.keys()):
                # if k.startswith('MAE_encoder'):
                #     base_ckpt[k[len('MAE_encoder.'):]] = base_ckpt[k]
                #     del base_ckpt[k]
                if k.startswith('base_model'):
                    base_ckpt[k[len('base_model.'):]] = base_ckpt[k]
                    del base_ckpt[k]

            incompatible = self.load_state_dict(base_ckpt, strict=False)

            if incompatible.missing_keys:
                print_log('missing_keys', logger='Mamba')
                print_log(
                    get_missing_parameters_message(incompatible.missing_keys),
                    logger='Mamba'
                )
            if incompatible.unexpected_keys:
                print_log('unexpected_keys', logger='Mamba')
                print_log(
                    get_unexpected_parameters_message(incompatible.unexpected_keys),
                    logger='Mamba'
                )

            print_log(f'[Mamba] Successful Loading the ckpt from {bert_ckpt_path}', logger='Mamba')
        else:
            print_log('Training from scratch!!!', logger='Mamba')
            self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, pts):

        device = pts.device                                           # pts: 32, 1024, 3
        center, group_input_tokens = self.group_divider(pts)          # center: B, G, 3, group_input_tokens: B, G, D
        pos = self.pos_embed(center)                                  # B, G, D

        # ordering strategy
        if self.mode_sort == 'random':
            # importance: random
            rows = []
            for _ in range(group_input_tokens.shape[0]):
                row = torch.randperm(group_input_tokens.shape[1])
                rows.append(row)
            random_order = torch.stack(rows).to(device).unsqueeze(-1)

            group_input_tokens = group_input_tokens.gather(dim=1, index=torch.tile(random_order, (1, 1, group_input_tokens.shape[-1])))
            pos = pos.gather(dim=1, index=torch.tile(random_order, (1, 1, pos.shape[-1]))) 

        elif self.mode_sort == 'no':
            # importance: initial order
            group_input_tokens = group_input_tokens
            pos = pos

        elif self.mode_sort == 'triple_xyz':
            center_x = center[:, :, 0].argsort(dim=-1)[:, :, None]      # 32, 64, 1
            center_y = center[:, :, 1].argsort(dim=-1)[:, :, None]
            center_z = center[:, :, 2].argsort(dim=-1)[:, :, None]
            group_input_tokens_x = group_input_tokens.gather(dim=1, index=torch.tile(center_x, (1, 1, group_input_tokens.shape[-1])))
            group_input_tokens_y = group_input_tokens.gather(dim=1, index=torch.tile(center_y, (1, 1, group_input_tokens.shape[-1])))
            group_input_tokens_z = group_input_tokens.gather(dim=1, index=torch.tile(center_z, (1, 1, group_input_tokens.shape[-1])))
            pos_x = pos.gather(dim=1, index=torch.tile(center_x, (1, 1, pos.shape[-1])))        # 32, 64, 384
            pos_y = pos.gather(dim=1, index=torch.tile(center_y, (1, 1, pos.shape[-1])))
            pos_z = pos.gather(dim=1, index=torch.tile(center_z, (1, 1, pos.shape[-1])))

            group_input_tokens = torch.cat([group_input_tokens_x, group_input_tokens_y, group_input_tokens_z], dim=1)                                               # 32, 192, 384
            pos = torch.cat([pos_x, pos_y, pos_z], dim=1)   

        elif self.mode_sort == 'z_order':
            z_order = get_z_order(center)

            group_input_tokens = group_input_tokens.gather(dim=1, index=torch.tile(z_order, (1, 1, group_input_tokens.shape[-1])))
            pos = pos.gather(dim=1, index=torch.tile(z_order, (1, 1, pos.shape[-1])))
        
        elif self.mode_sort == 'hilbert_order':
            Hilbert_order = get_hilbert_order(center)

            group_input_tokens = group_input_tokens.gather(dim=1, index=torch.tile(Hilbert_order, (1, 1, group_input_tokens.shape[-1])))
            pos = pos.gather(dim=1, index=torch.tile(Hilbert_order, (1, 1, pos.shape[-1])))

        else:
            assert self.mode_sort == 'max', 'Unknown sort mode'
 
        x = group_input_tokens
        x = self.drop_out(x)
        x = self.blocks(x, pos)
        x = self.norm(x)                                                                        # 32, 64, 384

        concat_f = x[:, :].mean(dim=1)               # average pooling, 32, 384

        # cls
        ret = self.cls_head_finetune(concat_f)

        return ret
    

def get_neighbor_feature(x, k=20):
    # x: B, N, D
    batch_size, num_points, num_dim = x.shape

    _, idx = KNN(k=k, transpose_mode=True)(x, x)       # B G (equals to N here) n_neighbor (k)

    idx_base = torch.arange(0, batch_size, device=x.device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dim).contiguous()       # B, N, n_neighbor, D

    return feature             # B, N, n_neighbor, D


class MlpLayer(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    

class AttentionLayer(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    

class AttentionBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MlpLayer(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.attn = AttentionLayer(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        
    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class TransformerEncoder(nn.Module):
    """ Transformer Encoder without hierarchical structure
    """
    def __init__(self, embed_dim=384, depth=4, num_heads=6, mlp_ratio=4., qkv_bias=False, qk_scale=None,
        drop_rate=0., attn_drop_rate=0., drop_path_rate=0.):
        super().__init__()
        
        self.blocks = nn.ModuleList([
            AttentionBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, 
                drop_path = drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate
                )
            for i in range(depth)])

    def forward(self, x, pos):
        for _, block in enumerate(self.blocks):
            x = block(x + pos)
        return x
    

class Attention(nn.Module):  # FPS + KNN
    def __init__(self, num_group, group_size, hidden_dimension, use_cls_token, depth, drop_path_rate, num_heads):
        super().__init__()

        self.num_group = num_group
        self.group_size = group_size
        self.hidden_dimension = hidden_dimension
        self.use_cls_token = use_cls_token
        self.depth = depth
        self.drop_path_rate = drop_path_rate
        self.num_heads = num_heads

        self.set_group = Group(num_group=self.num_group, group_size=self.group_size)

        self.embedding = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, hidden_dimension)
        )  

        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dimension))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, hidden_dimension))

        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, hidden_dimension)
        )  

        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        self.blocks = TransformerEncoder(
            embed_dim = hidden_dimension,
            depth = self.depth,
            drop_path_rate = dpr,
            num_heads = self.num_heads
        )

        self.norm = nn.LayerNorm(hidden_dimension)

        self.reduce_dim = nn.Linear(self.hidden_dimension * 2,  self.hidden_dimension)

    def forward(self, xyz):
        '''
            input: B N 3
            ---------------------------
            feature: B N D
            center : B N 3
        '''
        batch_size, num_points, num_dimension = xyz.shape
        neighborhood, center = self.set_group(xyz)                                                                      # xyz: B, N, 3, neighborhood: B, num_group, group_size, 3, center: B, num_group, 3

        neighborhood = neighborhood.reshape(-1, self.group_size, num_dimension)                                         # B*num_group, group_size, 3
        neighor_embedding = self.embedding(neighborhood)                                                                # map to embedding, B*num_group, group_size, D
        pos_embedding = self.pos_embed(neighborhood)                                                                    # map to embedding, B*num_group, group_size, D

        if self.use_cls_token:
            cls_tokens = self.cls_token.expand(neighor_embedding.size(0), -1, -1)                                       # B*num_group, 1, D
            cls_pos = self.cls_pos.expand(neighor_embedding.size(0), -1, -1)                                            # B*num_group, 1, D

            neighor_embedding = torch.cat([neighor_embedding, cls_tokens], dim=1)
            pos_embedding = torch.cat([pos_embedding, cls_pos], dim=1)

        # transformer
        neighor_embedding = self.blocks(neighor_embedding, pos_embedding)
        neighor_embedding = self.norm(neighor_embedding)

        # get global feature and cat with neighbor feature
        if self.use_cls_token:
            global_feature = neighor_embedding[:, 0]
            group_feature = torch.cat([neighor_embedding.max(dim=1)[0], global_feature], dim=-1)                        # B*num_group, D
            group_feature = self.reduce_dim(group_feature)

        else:
            global_feature = neighor_embedding.max(dim=1)[0]
            group_feature = global_feature
        
        group_feature = group_feature.reshape(batch_size, self.num_group, -1)                                           # B, num_group, D

        return center, group_feature
    

class ComplexAttention(nn.Module):  # FPS + KNN
    def __init__(self, num_group, group_size, hidden_dimension, use_cls_token, depth, drop_path_rate, num_heads):
        super().__init__()

        self.num_group = num_group
        self.group_size = group_size
        self.hidden_dimension = hidden_dimension
        self.depth = depth
        self.drop_path_rate = drop_path_rate
        self.num_heads = num_heads

        self.set_group = Group(num_group=self.num_group, group_size=self.group_size)

        self.embedding1 = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, hidden_dimension)
        )  

        self.pos_embed1 = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, hidden_dimension)
        )  

        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        self.blocks1 = TransformerEncoder(
            embed_dim = hidden_dimension,
            depth = self.depth,
            drop_path_rate = dpr,
            num_heads = self.num_heads
        )

        self.norm1 = nn.LayerNorm(hidden_dimension)

        self.reduce_dim = nn.Linear(self.hidden_dimension * 2,  self.hidden_dimension)

        self.embedding2 = nn.Sequential(
            nn.Linear(hidden_dimension, 128),
            nn.GELU(),
            nn.Linear(128, hidden_dimension)
        )  

        self.pos_embed2 = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, hidden_dimension)
        )  

        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        self.blocks2 = TransformerEncoder(
            embed_dim = hidden_dimension,
            depth = self.depth,
            drop_path_rate = dpr,
            num_heads = self.num_heads
        )

        self.norm2 = nn.LayerNorm(hidden_dimension)

    def forward(self, xyz):
        '''
            input: B N 3
            ---------------------------
            feature: B N D
            center : B N 3
        '''
        batch_size, num_points, num_dimension = xyz.shape
        neighborhood, center = self.set_group(xyz)                                                                       # xyz: B, N, 3, neighborhood: B, num_group, group_size, 3, center: B, num_group, 3

        neighborhood = neighborhood.reshape(-1, self.group_size, num_dimension)                                          # B*num_group, group_size, 3
        neighor_embedding = self.embedding1(neighborhood)                                                                # map to embedding, B*num_group, group_size, D
        pos_embedding = self.pos_embed1(neighborhood)                                                                    # map to embedding, B*num_group, group_size, D

        # transformer
        neighor_embedding = self.blocks1(neighor_embedding, pos_embedding)
        neighor_embedding = self.norm1(neighor_embedding)                                                                # B*num_group, group_size, D

        # get global feature and cat with neighbor feature
        global_feature = neighor_embedding.max(dim=1)[0]
        group_feature = global_feature
        
        neighor_embedding = torch.cat([neighor_embedding, group_feature.unsqueeze(1).repeat(1, neighor_embedding.shape[1], 1)], dim=-1)      # B*num_group, group_size, 2D
        
        # second Transformer layer
        neighor_embedding = self.reduce_dim(neighor_embedding)
        neighor_embedding = self.embedding2(neighor_embedding)                                                                # map to embedding, B*num_group, group_size, D
        pos_embedding = self.pos_embed2(neighborhood)                                                                         # map to embedding, B*num_group, group_size, D
    
        # transformer
        neighor_embedding = self.blocks2(neighor_embedding, pos_embedding)
        neighor_embedding = self.norm2(neighor_embedding)                                                                # B*num_group, group_size, D

        # get global feature
        global_feature = neighor_embedding.max(dim=1)[0]
        group_feature = global_feature

        group_feature = group_feature.reshape(batch_size, self.num_group, -1)                                            # B, num_group, D

        return center, group_feature


## Transformers
class TransformerMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = TransformerMlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.attn = TransformerAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        
    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
    

@MODELS.register_module()
class PointMambaFormer(nn.Module):
    def __init__(self, config, **kwargs):
        super(PointMambaFormer, self).__init__()
        self.config = config

        self.trans_dim = config.trans_dim
        self.depth = config.depth
        self.cls_dim = config.cls_dim

        self.group_size = config.group_size
        self.num_group = config.num_group
        self.encoder_dims = config.encoder_dims

        self.type_pooling = config.type_pooling
        self.type_weighting = config.type_weighting
        self.detach_mapping = config.detach_mapping
        self.detach_score_prediction = config.detach_score_prediction
        self.mode_sort = config.mode_sort
        self.mode_group = config.mode_group
        self.mode_encoder = config.mode_encoder
        self.Transformer_encoder_num_heads = config.Transformer_encoder_num_heads

        self.use_simple_score_predictor = config.use_simple_score_predictor
        
        if self.mode_group == 'SA':
            self.group_divider = SA(num_group=self.num_group, group_size=self.group_size, encoder_channel=self.encoder_dims)
        elif self.mode_group == 'EdgeConv':
            self.group_divider = EdgeConv(k=self.group_size, hidden_dimension=self.encoder_dims)
        elif self.mode_group == 'Attention':
            self.group_divider = Attention(num_group=self.num_group, group_size=self.group_size, hidden_dimension=self.encoder_dims, use_cls_token=self.config.attention_use_cls_token, depth=self.config.attention_depth, drop_path_rate=self.config.attention_drop_path_rate, num_heads=self.config.attention_num_heads)
        elif self.mode_group == 'ComplexAttention':
            self.group_divider = ComplexAttention(num_group=self.num_group, group_size=self.group_size, hidden_dimension=self.encoder_dims, use_cls_token=self.config.attention_use_cls_token, depth=self.config.attention_depth, drop_path_rate=self.config.attention_drop_path_rate, num_heads=self.config.attention_num_heads)

        if self.use_simple_score_predictor:
            self.importance_cal_block = CalImportanceSimple(self.encoder_dims, self.detach_score_prediction)
        else:
            self.importance_cal_block = CalImportance(self.encoder_dims, self.detach_score_prediction)

        self.use_cls_token = False if not hasattr(self.config, "use_cls_token") else self.config.use_cls_token
        self.drop_path = 0. if not hasattr(self.config, "drop_path") else self.config.drop_path
        self.rms_norm = False if not hasattr(self.config, "rms_norm") else self.config.rms_norm
        self.drop_out_in_block = 0. if not hasattr(self.config, "drop_out_in_block") else self.config.drop_out_in_block

        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )

        if self.mode_encoder == 'Mamba':
            self.blocks = MixerModel(d_model=self.trans_dim,
                                    n_layer=self.depth,
                                    rms_norm=self.rms_norm,
                                    drop_out_in_block=self.drop_out_in_block,
                                    drop_path=self.drop_path)
            
        elif self.mode_encoder == 'Transformer':
            dpr = [x.item() for x in torch.linspace(0, self.config.attention_drop_path_rate, self.depth)]
            self.blocks = TransformerEncoder(
                embed_dim=self.trans_dim,
                depth=self.depth,
                drop_path_rate=dpr,
                num_heads=self.Transformer_encoder_num_heads,
            )

        self.norm = nn.LayerNorm(self.trans_dim)

        self.HEAD_CHANEL = 1
        if self.use_cls_token:
            self.HEAD_CHANEL += 1

        g_input_dim = self.trans_dim * self.HEAD_CHANEL

        self.cls_head_finetune = nn.Sequential(
            nn.Linear(g_input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, self.cls_dim)
        )

        self.global_projection = nn.Sequential(
            nn.Conv1d(g_input_dim, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1),
            Normalize(dim=1)
        )

        self.build_loss_func()

        self.drop_out = nn.Dropout(config.drop_out) if "drop_out" in config else nn.Dropout(0)

    def build_loss_func(self):
        self.loss_ce = nn.CrossEntropyLoss()
        self.loss_glr = GLR_loss()
        self.loss_importance = nn.SmoothL1Loss(reduction='none')

    def get_loss_acc(self, ret, gt, patch_f, global_f, pred_score, cos_sim):
        # CE loss
        loss_CE = self.loss_ce(ret, gt.long())

        # global-to-local align loss
        loss_GLR = self.loss_glr(global_f, patch_f)

        # importance regression loss
        loss_SCORE = self.loss_importance(pred_score.squeeze(-1), cos_sim).sum(dim=-1).mean()

        # pred
        pred = ret.argmax(-1)
        acc = (pred == gt).sum() / float(gt.size(0))
        return loss_CE, loss_GLR, loss_SCORE, acc * 100

    def load_model_from_ckpt(self, bert_ckpt_path):
        if bert_ckpt_path is not None:
            ckpt = torch.load(bert_ckpt_path)
            base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['base_model'].items()}

            for k in list(base_ckpt.keys()):
                # if k.startswith('MAE_encoder'):
                #     base_ckpt[k[len('MAE_encoder.'):]] = base_ckpt[k]
                #     del base_ckpt[k]
                if k.startswith('base_model'):
                    base_ckpt[k[len('base_model.'):]] = base_ckpt[k]
                    del base_ckpt[k]

            incompatible = self.load_state_dict(base_ckpt, strict=False)

            if incompatible.missing_keys:
                print_log('missing_keys', logger='Mamba')
                print_log(
                    get_missing_parameters_message(incompatible.missing_keys),
                    logger='Mamba'
                )
            if incompatible.unexpected_keys:
                print_log('unexpected_keys', logger='Mamba')
                print_log(
                    get_unexpected_parameters_message(incompatible.unexpected_keys),
                    logger='Mamba'
                )

            print_log(f'[Mamba] Successful Loading the ckpt from {bert_ckpt_path}', logger='Mamba')
        else:
            print_log('Training from scratch!!!', logger='Mamba')
            self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, pts):
        device = pts.device                                           # pts: 32, 1024, 3
        center, group_input_tokens = self.group_divider(pts)          # center: B, G, 3, group_input_tokens: B, G, D
        pos = self.pos_embed(center)                                  # B, G, D

        # introduce a local-to-global mapping layer here
        # option: cls on patch, local-to-global alignment on patch, cosine similarity regression on patch
        if self.detach_mapping:
            map_patch_feature, cal_importance = self.importance_cal_block(group_input_tokens.detach())      # map_patch_feature: B, G, D1; cal_importance: B, G, 1
        else:
            map_patch_feature, cal_importance = self.importance_cal_block(group_input_tokens)               # map_patch_feature: B, G, D1; cal_importance: B, G, 1

        # importance_order
        if self.mode_sort == 'max':
            # importance: max to min
            importance_order = cal_importance.argsort(dim=1, descending=True)
            sort_cal_importance = cal_importance.gather(dim=1, index=torch.tile(importance_order, (1, 1, cal_importance.shape[-1])))

            group_input_tokens = group_input_tokens.gather(dim=1, index=torch.tile(importance_order, (1, 1, group_input_tokens.shape[-1])))
            pos = pos.gather(dim=1, index=torch.tile(importance_order, (1, 1, pos.shape[-1]))) 

            # to recover the feature of each token
            inv_importance_order = torch.argsort(importance_order, dim=1)
            inv_importance_order_expanded = inv_importance_order.expand(-1, -1, group_input_tokens.size(2))

        elif self.mode_sort == 'min':
            # importance: min to max
            importance_order = cal_importance.argsort(dim=1, descending=False)
            sort_cal_importance = cal_importance.gather(dim=1, index=torch.tile(importance_order, (1, 1, cal_importance.shape[-1])))

            group_input_tokens = group_input_tokens.gather(dim=1, index=torch.tile(importance_order, (1, 1, group_input_tokens.shape[-1])))
            pos = pos.gather(dim=1, index=torch.tile(importance_order, (1, 1, pos.shape[-1]))) 

            # to recover the feature of each token
            inv_importance_order = torch.argsort(importance_order, dim=1)
            inv_importance_order_expanded = inv_importance_order.expand(-1, -1, group_input_tokens.size(2))

        elif self.mode_sort == 'both':
            # importance: max to min and min to max
            importance_order1 = cal_importance.argsort(dim=1, descending=True)
            sort_cal_importance1 = cal_importance.gather(dim=1, index=torch.tile(importance_order1, (1, 1, cal_importance.shape[-1])))
            group_input_tokens1 = group_input_tokens.gather(dim=1, index=torch.tile(importance_order1, (1, 1, group_input_tokens.shape[-1])))
            pos1 = pos.gather(dim=1, index=torch.tile(importance_order1, (1, 1, pos.shape[-1]))) 

            importance_order2 = cal_importance.argsort(dim=1, descending=False)
            sort_cal_importance2 = cal_importance.gather(dim=1, index=torch.tile(importance_order2, (1, 1, cal_importance.shape[-1])))
            group_input_tokens2 = group_input_tokens.gather(dim=1, index=torch.tile(importance_order2, (1, 1, group_input_tokens.shape[-1])))
            pos2 = pos.gather(dim=1, index=torch.tile(importance_order2, (1, 1, pos.shape[-1]))) 

            sort_cal_importance = torch.cat([sort_cal_importance1, sort_cal_importance2], dim=1)
            group_input_tokens = torch.cat([group_input_tokens1, group_input_tokens2], dim=1)
            pos = torch.cat([pos1, pos2], dim=1) 

            # to recover the feature of each token
            inv_importance_order1 = torch.argsort(importance_order1, dim=1)
            inv_importance_order2 = torch.argsort(importance_order2, dim=1)
            inv_importance_order_expanded1 = inv_importance_order1.expand(-1, -1, group_input_tokens.size(2))
            inv_importance_order_expanded2 = inv_importance_order2.expand(-1, -1, group_input_tokens.size(2))

        elif self.mode_sort == 'no':
            group_input_tokens = group_input_tokens
            pos = pos

        else:
            assert self.mode_sort == 'max', 'Unknown sort mode'

        x = group_input_tokens
        x = self.drop_out(x)
        x = self.blocks(x, pos)
        x = self.norm(x)            

        if self.type_pooling == 'ave':
            global_feature = x[:, :].mean(dim=1)                                                          # B, D
        
        elif self.type_pooling == 'max':
            global_feature = x[:, :].max(dim=1)[0]
        
        elif self.type_pooling == 'important':
            if self.type_weighting == 'direct':
                weight = sort_cal_importance.repeat(1, 1, x.shape[-1])
                global_feature = (x * weight).sum(dim=1)
            
            elif self.type_weighting == 'drop_neg':
                weight = sort_cal_importance.clamp(0, 1).repeat(1, 1, x.shape[-1])
                global_feature = (x * weight).sum(dim=1)
            
            elif self.type_weighting == 'sfm':
                weight = nn.Softmax(dim=1)(sort_cal_importance).repeat(1, 1, x.shape[-1])
                global_feature = (x * weight).sum(dim=1)
            
            else:
                # TODO find some new solutions
                global_feature = x[:, :].mean(dim=1)   
        
        else:
            # TODO find some new solutions
            global_feature = x[:, :].mean(dim=1)   

        # global mapping --> 256 dimension
        map_global_feature = self.global_projection(global_feature.unsqueeze(-1)).squeeze(-1)            # B, D1

        # cls
        ret = self.cls_head_finetune(global_feature)                                                     # B, num_class
        
        return ret, map_patch_feature, map_global_feature, cal_importance
    
