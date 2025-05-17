from typing import Optional, Tuple
import math
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from timm.models.layers import DropPath, trunc_normal_
from logger import get_missing_parameters_message, get_unexpected_parameters_message

import pdb

from pointnet2_ops import pointnet2_utils
from knn_cuda import KNN
from pointnet2_utils import PointNetFeaturePropagation

from mamba_ssm.modules.mamba_simple import Mamba

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None


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
                # Special Scaled Initialization --> There are 2 Layer Norms per Mamba Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


class Block(nn.Module):
    def __init__(
            self, dim, mixer_cls, norm_cls=nn.LayerNorm, fused_add_norm=False, residual_in_fp32=False, drop_path=0.
    ):
        """
        Simple block wrapping a mixer class with LayerNorm/RMSNorm and residual connection"

        This Block has a slightly different structure compared to a regular
        prenorm Mamba block.
        The standard block is: LN -> MHA/MLP -> Add.
        [Ref: https://arxiv.org/abs/2002.04745]
        Here we have: Add -> LN -> Mixer, returning both
        the hidden_states (output of the mixer) and the residual.
        This is purely for performance reasons, as we can fuse add and LayerNorm.
        The residual needs to be provided (except for the very first block).
        """
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.mixer = mixer_cls(dim)
        self.norm = norm_cls(dim)

        # drop path 
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"

    def forward(self, hidden_states: Tensor, residual: Optional[Tensor] = None, inference_params=None):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Mixer(LN(residual))
        """
        if not self.fused_add_norm:
            residual = (self.drop_path(hidden_states) + residual) if residual is not None else hidden_states
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            hidden_states, residual = fused_add_norm_fn(
                self.drop_path(hidden_states),
                self.norm.weight,
                self.norm.bias,
                residual=residual,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
                eps=self.norm.eps,
            )
        hidden_states = self.mixer(hidden_states, inference_params=inference_params)
        return hidden_states, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)


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
        dtype=None, ):
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


def fps(data, number):
    '''
        data B N 3
        number int
    '''
    fps_idx = pointnet2_utils.furthest_point_sample(data, number)
    fps_data = pointnet2_utils.gather_operation(data.transpose(1, 2).contiguous(), fps_idx).transpose(1, 2).contiguous()
    return fps_data


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
        center = fps(xyz, self.num_group)  # B G 3
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
        for layer in self.layers:
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


class MixerModelForSegmentation(MixerModel):
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
            drop_path: int = 0.1,
            fetch_idx: Tuple[int] = [3, 7, 11],
            device=None,
            dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super(MixerModel, self).__init__()
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

        self.fetch_idx = fetch_idx

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

    def forward(self, input_ids, pos, inference_params=None):
        hidden_states = input_ids  # + pos
        residual = None
        hidden_states = hidden_states + pos
        feature_list = []
        for idx, layer in enumerate(self.layers):
            hidden_states, residual = layer(
                hidden_states, residual, inference_params=inference_params
            )
            if idx in self.fetch_idx:
                if not self.fused_add_norm:
                    residual_output = (hidden_states + residual) if residual is not None else hidden_states
                    hidden_states_output = self.norm_f(residual_output.to(dtype=self.norm_f.weight.dtype))
                else:
                    # Set prenorm=False here since we don't need the residual
                    fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
                    hidden_states_output = fused_add_norm_fn(
                        hidden_states,
                        self.norm_f.weight,
                        self.norm_f.bias,
                        eps=self.norm_f.eps,
                        residual=residual,
                        prenorm=False,
                        residual_in_fp32=self.residual_in_fp32,
                    )
                feature_list.append(hidden_states_output)
        return feature_list


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
    

class get_model(nn.Module):
    def __init__(self, cls_dim, config=None):
        super().__init__()
        self.config = config

        self.trans_dim = config.trans_dim
        self.depth = config.depth
        self.cls_dim = cls_dim

        self.group_size = config.group_size
        self.num_group = config.num_group
        self.encoder_dims = 384

        self.detach_mapping = config.detach_mapping
        self.detach_score_prediction = config.detach_score_prediction
        self.mode_sort = config.mode_sort
        self.mode_group = config.mode_group
        self.use_simple_score_predictor = config.use_simple_score_predictor
        self.mode_map_feature = config.mode_map_feature
        self.cat_group_token = config.cat_group_token

        if self.mode_group == 'SA':
            self.group_divider = SA(num_group=self.num_group, group_size=self.group_size, encoder_channel=self.encoder_dims)
        elif self.mode_group == 'Attention':
            self.group_divider = Attention(num_group=self.num_group, group_size=self.group_size, hidden_dimension=self.encoder_dims, use_cls_token=self.config.attention_use_cls_token, depth=self.config.attention_depth, drop_path_rate=self.config.attention_drop_path_rate, num_heads=self.config.attention_num_heads)

        if self.use_simple_score_predictor:
            self.importance_cal_block = CalImportanceSimple(self.encoder_dims, self.detach_score_prediction)
        else:
            self.importance_cal_block = CalImportance(self.encoder_dims, self.detach_score_prediction)

        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )
        self.blocks = MixerModelForSegmentation(d_model=self.trans_dim,
                                                n_layer=self.depth,
                                                rms_norm=config.rms_norm,
                                                drop_path=config.drop_path,
                                                fetch_idx=config.fetch_idx)

        self.norm = nn.LayerNorm(self.trans_dim)

        self.label_conv = nn.Sequential(nn.Conv1d(16, 64, kernel_size=1, bias=False),
                                        nn.BatchNorm1d(64),
                                        nn.LeakyReLU(0.2))

        if self.cat_group_token:
            self.propagation_0 = PointNetFeaturePropagation(in_channel=1152 + 3 + 384, mlp=[self.trans_dim * 4, 1024])
        else:
            self.propagation_0 = PointNetFeaturePropagation(in_channel=1152 + 3, mlp=[self.trans_dim * 4, 1024])

        if self.mode_map_feature == 'max':
            self.global_projection = nn.Sequential(
                nn.Conv1d(1152, 128, 1),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),
                nn.Conv1d(128, 256, 1),
                Normalize(dim=1)
            )
        elif self.mode_map_feature == 'ave':
            self.global_projection = nn.Sequential(
                nn.Conv1d(1152, 128, 1),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),
                nn.Conv1d(128, 256, 1),
                Normalize(dim=1)
            )
        else:
            self.global_projection = nn.Sequential(
                nn.Conv1d(2368, 128, 1),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),
                nn.Conv1d(128, 256, 1),
                Normalize(dim=1)
            )

        self.convs1 = nn.Conv1d(3392, 512, 1)
        self.dp1 = nn.Dropout(0.5)
        self.convs2 = nn.Conv1d(512, 256, 1)
        self.convs3 = nn.Conv1d(256, self.cls_dim, 1)
        self.bns1 = nn.BatchNorm1d(512)
        self.bns2 = nn.BatchNorm1d(256)

        self.relu = nn.ReLU()

    def get_loss_acc(self, ret, gt):
        loss = self.loss_ce(ret, gt.long())
        pred = ret.argmax(-1)
        acc = (pred == gt).sum() / float(gt.size(0))
        return loss, acc * 100

    def load_model_from_ckpt(self, bert_ckpt_path):
        if bert_ckpt_path is not None:
            ckpt = torch.load(bert_ckpt_path)
            base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['base_model'].items()}
            for k in list(base_ckpt.keys()):
                if k.startswith('MAE_encoder'):
                    base_ckpt[k[len('MAE_encoder.'):]] = base_ckpt[k]
                    del base_ckpt[k]
                elif k.startswith('base_model'):
                    base_ckpt[k[len('base_model.'):]] = base_ckpt[k]
                    del base_ckpt[k]
            incompatible = self.load_state_dict(base_ckpt, strict=False)
            if incompatible.missing_keys:
                print('missing_keys')
                print(get_missing_parameters_message(incompatible.missing_keys))
            if incompatible.unexpected_keys:
                print('unexpected_keys')
                print(get_unexpected_parameters_message(incompatible.unexpected_keys))
            print(f'[Mamba] Successful Loading the ckpt from {bert_ckpt_path}')
        else:
            print(f'[Mamba] No ckpt is loaded, training from scratch!')

    def forward(self, pts, cls_label):
        B, C, N = pts.shape
        pts = pts.transpose(-1, -2)  # B N 3
        # divide the point cloud in the same form. This is important
        center, group_input_tokens = self.group_divider(pts)          # center: B, G, 3, group_input_tokens: B, G, D

        pos = self.pos_embed(center)

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
            center = center.gather(dim=1, index=torch.tile(importance_order, (1, 1, center.shape[-1]))) 

        elif self.mode_sort == 'min':
            # importance: min to max
            importance_order = cal_importance.argsort(dim=1, descending=False)
            sort_cal_importance = cal_importance.gather(dim=1, index=torch.tile(importance_order, (1, 1, cal_importance.shape[-1])))

            group_input_tokens = group_input_tokens.gather(dim=1, index=torch.tile(importance_order, (1, 1, group_input_tokens.shape[-1])))
            pos = pos.gather(dim=1, index=torch.tile(importance_order, (1, 1, pos.shape[-1]))) 
            center = center.gather(dim=1, index=torch.tile(importance_order, (1, 1, center.shape[-1]))) 

        elif self.mode_sort == 'both':
            # importance: max to min and min to max
            importance_order1 = cal_importance.argsort(dim=1, descending=True)
            sort_cal_importance1 = cal_importance.gather(dim=1, index=torch.tile(importance_order1, (1, 1, cal_importance.shape[-1])))
            group_input_tokens1 = group_input_tokens.gather(dim=1, index=torch.tile(importance_order1, (1, 1, group_input_tokens.shape[-1])))
            pos1 = pos.gather(dim=1, index=torch.tile(importance_order1, (1, 1, pos.shape[-1]))) 
            center1 = center.gather(dim=1, index=torch.tile(importance_order1, (1, 1, center.shape[-1]))) 

            importance_order2 = cal_importance.argsort(dim=1, descending=False)
            sort_cal_importance2 = cal_importance.gather(dim=1, index=torch.tile(importance_order2, (1, 1, cal_importance.shape[-1])))
            group_input_tokens2 = group_input_tokens.gather(dim=1, index=torch.tile(importance_order2, (1, 1, group_input_tokens.shape[-1])))
            pos2 = pos.gather(dim=1, index=torch.tile(importance_order2, (1, 1, pos.shape[-1]))) 
            center2 = center.gather(dim=1, index=torch.tile(importance_order2, (1, 1, center.shape[-1]))) 

            sort_cal_importance = torch.cat([sort_cal_importance1, sort_cal_importance2], dim=1)
            group_input_tokens = torch.cat([group_input_tokens1, group_input_tokens2], dim=1)
            pos = torch.cat([pos1, pos2], dim=1) 
            center = torch.cat([center1, center2], dim=1) 

        else:
            assert self.mode_sort == 'max', 'Unknown sort mode'

        x = group_input_tokens
        feature_list = self.blocks(x, pos)

        feature_list = [self.norm(x).transpose(-1, -2).contiguous() for x in feature_list]      # transpose!!! B, D2, num_group*2
        x = torch.cat((feature_list), dim=1)                                                    # multi-layer hidden state: B, D2*3, num_group*2
        # TODO cat with group feature or not
        x_max = torch.max(x, 2)[0]                                                              # max pooling: B, D2*2
        x_avg = torch.mean(x, 2)                                                                # average pooling: B, D2*2
        x_max_feature = x_max.view(B, -1).unsqueeze(-1).repeat(1, 1, N)                         # B, D2*2, num_points
        x_avg_feature = x_avg.view(B, -1).unsqueeze(-1).repeat(1, 1, N)
        cls_label_one_hot = cls_label.view(B, 16, 1)
        cls_label_feature = self.label_conv(cls_label_one_hot).repeat(1, 1, N)                  # B, D3, num_points
        x_global_feature = torch.cat((x_max_feature, x_avg_feature, cls_label_feature), 1)      # B, (D2*2 + D2*2 + D3), num_points

        if self.mode_map_feature == 'max':
            map_global_feature = self.global_projection(x_max.unsqueeze(-1)).squeeze(-1)
        elif self.mode_map_feature == 'ave':
            map_global_feature = self.global_projection(x_avg.unsqueeze(-1)).squeeze(-1)
        else:
            map_global_feature = self.global_projection(x_global_feature[:, :, 0].unsqueeze(-1)).squeeze(-1)

        if self.cat_group_token:
            group_feature = torch.cat([x, group_input_tokens.transpose(-1, -2)], dim=1)
        else:
            group_feature = x

        f_level_0 = self.propagation_0(pts.transpose(-1, -2), center.transpose(-1, -2), pts.transpose(-1, -2), group_feature)       # B, D4, num_points

        x = torch.cat((f_level_0, x_global_feature), 1)
        x = self.relu(self.bns1(self.convs1(x)))
        x = self.dp1(x)
        x = self.relu(self.bns2(self.convs2(x)))
        x = self.convs3(x)
        x = F.log_softmax(x, dim=1)
        x = x.permute(0, 2, 1)
        
        return x, map_patch_feature, map_global_feature, cal_importance


class get_model2(nn.Module):
    def __init__(self, cls_dim, config=None):
        super().__init__()
        self.config = config

        self.trans_dim = config.trans_dim
        self.depth = config.depth
        self.cls_dim = cls_dim

        self.group_size = config.group_size
        self.num_group = config.num_group
        self.encoder_dims = 384

        self.detach_mapping = config.detach_mapping
        self.detach_score_prediction = config.detach_score_prediction
        self.mode_sort = config.mode_sort
        self.mode_group = config.mode_group
        self.use_simple_score_predictor = config.use_simple_score_predictor
        self.mode_map_feature = config.mode_map_feature
        self.cat_group_token = config.cat_group_token
        self.use_cls_feature = config.use_cls_feature

        if self.mode_group == 'SA':
            self.group_divider = SA(num_group=self.num_group, group_size=self.group_size, encoder_channel=self.encoder_dims)
        elif self.mode_group == 'Attention':
            self.group_divider = Attention(num_group=self.num_group, group_size=self.group_size, hidden_dimension=self.encoder_dims, use_cls_token=self.config.attention_use_cls_token, depth=self.config.attention_depth, drop_path_rate=self.config.attention_drop_path_rate, num_heads=self.config.attention_num_heads)

        if self.use_simple_score_predictor:
            self.importance_cal_block = CalImportanceSimple(self.encoder_dims, self.detach_score_prediction)
        else:
            self.importance_cal_block = CalImportance(self.encoder_dims, self.detach_score_prediction)

        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )
        self.blocks = MixerModelForSegmentation(d_model=self.trans_dim,
                                                n_layer=self.depth,
                                                rms_norm=config.rms_norm,
                                                drop_path=config.drop_path,
                                                fetch_idx=config.fetch_idx)

        self.norm = nn.LayerNorm(self.trans_dim)

        self.label_conv = nn.Sequential(nn.Conv1d(16, 64, kernel_size=1, bias=False),
                                        nn.BatchNorm1d(64),
                                        nn.LeakyReLU(0.2))

        if self.cat_group_token:
            input_dim = 2304 + 384
        else:
            input_dim = 2304
            
        self.reduce_dim = nn.Linear(input_dim, 1024)

        self.propagation_0 = PointNetFeaturePropagation(in_channel=1024 + 3, mlp=[self.trans_dim * 4, 1024])

        self.global_projection = nn.Sequential(
            nn.Conv1d(1024, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1),
            Normalize(dim=1)
        )

        if self.use_cls_feature:
            self.convs1 = nn.Conv1d(2048 + 64, 512, 1)
        else:
            self.convs1 = nn.Conv1d(2048, 512, 1)
        self.dp1 = nn.Dropout(0.5)
        self.convs2 = nn.Conv1d(512, 256, 1)
        self.convs3 = nn.Conv1d(256, self.cls_dim, 1)
        self.bns1 = nn.BatchNorm1d(512)
        self.bns2 = nn.BatchNorm1d(256)

        self.relu = nn.ReLU()

    def get_loss_acc(self, ret, gt):
        loss = self.loss_ce(ret, gt.long())
        pred = ret.argmax(-1)
        acc = (pred == gt).sum() / float(gt.size(0))
        return loss, acc * 100

    def load_model_from_ckpt(self, bert_ckpt_path):
        if bert_ckpt_path is not None:
            ckpt = torch.load(bert_ckpt_path)
            base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['base_model'].items()}
            for k in list(base_ckpt.keys()):
                if k.startswith('MAE_encoder'):
                    base_ckpt[k[len('MAE_encoder.'):]] = base_ckpt[k]
                    del base_ckpt[k]
                elif k.startswith('base_model'):
                    base_ckpt[k[len('base_model.'):]] = base_ckpt[k]
                    del base_ckpt[k]
            incompatible = self.load_state_dict(base_ckpt, strict=False)
            if incompatible.missing_keys:
                print('missing_keys')
                print(get_missing_parameters_message(incompatible.missing_keys))
            if incompatible.unexpected_keys:
                print('unexpected_keys')
                print(get_unexpected_parameters_message(incompatible.unexpected_keys))
            print(f'[Mamba] Successful Loading the ckpt from {bert_ckpt_path}')
        else:
            print(f'[Mamba] No ckpt is loaded, training from scratch!')

    def forward(self, pts, cls_label):
        B, C, N = pts.shape
        pts = pts.transpose(-1, -2)  # B N 3
        # divide the point cloud in the same form. This is important
        center, group_input_tokens = self.group_divider(pts)          # center: B, G, 3, group_input_tokens: B, G, D
        center_initial = center

        pos = self.pos_embed(center)

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
            center = center.gather(dim=1, index=torch.tile(importance_order, (1, 1, center.shape[-1]))) 

            # to recover the feature of each token
            inv_importance_order = torch.argsort(importance_order, dim=1)

        elif self.mode_sort == 'min':
            # importance: min to max
            importance_order = cal_importance.argsort(dim=1, descending=False)
            sort_cal_importance = cal_importance.gather(dim=1, index=torch.tile(importance_order, (1, 1, cal_importance.shape[-1])))

            group_input_tokens = group_input_tokens.gather(dim=1, index=torch.tile(importance_order, (1, 1, group_input_tokens.shape[-1])))
            pos = pos.gather(dim=1, index=torch.tile(importance_order, (1, 1, pos.shape[-1]))) 
            center = center.gather(dim=1, index=torch.tile(importance_order, (1, 1, center.shape[-1]))) 

            # to recover the feature of each token
            inv_importance_order = torch.argsort(importance_order, dim=1)

        elif self.mode_sort == 'both':
            # importance: max to min and min to max
            importance_order1 = cal_importance.argsort(dim=1, descending=True)
            sort_cal_importance1 = cal_importance.gather(dim=1, index=torch.tile(importance_order1, (1, 1, cal_importance.shape[-1])))
            group_input_tokens1 = group_input_tokens.gather(dim=1, index=torch.tile(importance_order1, (1, 1, group_input_tokens.shape[-1])))
            pos1 = pos.gather(dim=1, index=torch.tile(importance_order1, (1, 1, pos.shape[-1]))) 
            center1 = center.gather(dim=1, index=torch.tile(importance_order1, (1, 1, center.shape[-1]))) 

            importance_order2 = cal_importance.argsort(dim=1, descending=False)
            sort_cal_importance2 = cal_importance.gather(dim=1, index=torch.tile(importance_order2, (1, 1, cal_importance.shape[-1])))
            group_input_tokens2 = group_input_tokens.gather(dim=1, index=torch.tile(importance_order2, (1, 1, group_input_tokens.shape[-1])))
            pos2 = pos.gather(dim=1, index=torch.tile(importance_order2, (1, 1, pos.shape[-1]))) 
            center2 = center.gather(dim=1, index=torch.tile(importance_order2, (1, 1, center.shape[-1]))) 

            sort_cal_importance = torch.cat([sort_cal_importance1, sort_cal_importance2], dim=1)
            group_input_tokens = torch.cat([group_input_tokens1, group_input_tokens2], dim=1)
            pos = torch.cat([pos1, pos2], dim=1) 
            center = torch.cat([center1, center2], dim=1) 

            # to recover the feature of each token
            inv_importance_order1 = torch.argsort(importance_order1, dim=1)
            inv_importance_order2 = torch.argsort(importance_order2, dim=1)

        else:
            assert self.mode_sort == 'max', 'Unknown sort mode'

        x = group_input_tokens
        feature_list = self.blocks(x, pos)                                      

        feature_list = [self.norm(x).transpose(-1, -2).contiguous() for x in feature_list]      # transpose!!! B, D2, num_group*2
        x = torch.cat((feature_list), dim=1)                                                    # multi-layer hidden state: B, D2*3, num_group*2

        if self.mode_sort == 'both':
            x1 = x[:, :, :self.num_group]
            x2 = x[:, :, self.num_group:]

            inv_importance_order1_expand = inv_importance_order1.expand(-1, -1, x1.size(1)).transpose(-1, -2)       # 1152
            inv_importance_order2_expand = inv_importance_order2.expand(-1, -1, x2.size(1)).transpose(-1, -2)
            patch_feature1 = torch.gather(x1, 2, inv_importance_order1_expand) 
            patch_feature2 = torch.gather(x2, 2, inv_importance_order2_expand) 
            patch_feature = torch.cat([patch_feature1, patch_feature2], dim=1)                                      # 2304
        else:
            inv_importance_order_expand = inv_importance_order.expand(-1, -1, x.size(1)).transpose(-1, -2)          # 1152
            patch_feature = torch.gather(x, 2, inv_importance_order_expand) 
        
        if self.cat_group_token:
            patch_feature = torch.cat([patch_feature, group_input_tokens.transpose(-1, -2)], dim=1)
        
        patch_feature = self.reduce_dim(patch_feature.transpose(-1, -2)).transpose(-1, -2)
        f_level_0 = self.propagation_0(pts.transpose(-1, -2), center_initial.transpose(-1, -2), pts.transpose(-1, -2), patch_feature)       # B, D4, num_points

        if self.mode_map_feature == 'max':
            feature = torch.max(patch_feature, 2)[0]
        elif self.mode_map_feature == 'ave':
            feature = torch.mean(patch_feature, 2)   
        else:
            assert self.mode_map_feature == 'max', 'Unknown pooling'

        map_global_feature = self.global_projection(feature.unsqueeze(-1)).squeeze(-1)

        cls_label_one_hot = cls_label.view(B, 16, 1)
        cls_label_feature = self.label_conv(cls_label_one_hot).repeat(1, 1, N)                  # B, D3, num_points

        feature = feature.unsqueeze(-1).repeat(1, 1, f_level_0.shape[-1])

        if self.use_cls_feature:
            feature = torch.cat([feature, cls_label_feature], dim=1)

        x = torch.cat((f_level_0, feature), 1)
        x = self.relu(self.bns1(self.convs1(x)))
        x = self.dp1(x)
        x = self.relu(self.bns2(self.convs2(x)))
        x = self.convs3(x)
        x = F.log_softmax(x, dim=1)
        x = x.permute(0, 2, 1)
        
        return x, map_patch_feature, map_global_feature, cal_importance
    

class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target):
        total_loss = F.nll_loss(pred, target)
        return total_loss


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
    