# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from vggt.layers import Mlp
from vggt.layers.block import Block
from vggt.heads.head_act import activate_pose
from vggt.utils.pose_enc import extri_intri_to_pose_encoding


class WristHead(nn.Module):
    """
    WristHead predicts wrist camera parameters from token representations using iterative refinement.

    Modified to concatenate tokens from two views (ext1 and ext2) and output a single wrist pose.
    """

    def __init__(
        self,
        dim_in: int = 2048,
        trunk_depth: int = 4,
        pose_encoding_type: str = "absT_quaR_FoV",
        num_heads: int = 16,
        mlp_ratio: int = 4,
        init_values: float = 0.01,
        trans_act: str = "linear",
        quat_act: str = "linear",
        fl_act: str = "relu",  # Field of view activations: ensures FOV values are positive.
        token_aggregation: str = "attention",  # "attention", "mean", "weighted_mean"
        # 多层Transformer聚合参数
        aggregation_num_layers: int = 3,    # Transformer层数
        aggregation_num_heads: int = 8,     # 注意力头数
        aggregation_dropout: float = 0.1,   # Dropout比例
    ):
        super().__init__()

        if pose_encoding_type == "absT_quaR_FoV":
            self.target_dim = 9
        else:
            raise ValueError(f"Unsupported wrist encoding type: {pose_encoding_type}")

        self.trans_act = trans_act
        self.quat_act = quat_act
        self.fl_act = fl_act
        self.trunk_depth = trunk_depth
        self.token_aggregation = token_aggregation

        # Build the trunk using a sequence of transformer blocks.
        self.trunk = nn.Sequential(
            *[
                Block(dim=dim_in, num_heads=num_heads, mlp_ratio=mlp_ratio, init_values=init_values)
                for _ in range(trunk_depth)
            ]
        )

        # Normalizations for wrist token and trunk output.
        self.token_norm = nn.LayerNorm(dim_in)
        self.trunk_norm = nn.LayerNorm(dim_in)

        # Token aggregation modules
        if self.token_aggregation == "attention":
            # Multi-layer transformer for token aggregation
            self.token_query = nn.Parameter(torch.randn(1, 1, dim_in))
            self.token_aggregation_layers = self._build_token_transformer_layers(
                dim_in, 
                num_heads=aggregation_num_heads, 
                num_layers=aggregation_num_layers, 
                dropout=aggregation_dropout
            )
            nn.init.normal_(self.token_query, std=1e-6)
        elif self.token_aggregation == "weighted_mean":
            # Learnable weights for weighted mean aggregation
            self.token_weights = nn.Parameter(torch.ones(1, 1, 1))  # Will be expanded based on actual token count

        # Learnable empty wrist pose token.
        self.empty_pose_tokens = nn.Parameter(torch.zeros(1, 1, self.target_dim))
        self.embed_pose = nn.Linear(self.target_dim, dim_in)

        # Module for producing modulation parameters: shift, scale, and a gate.
        self.poseLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim_in, 3 * dim_in, bias=True))

        # Adaptive layer normalization without affine parameters.
        self.adaln_norm = nn.LayerNorm(dim_in, elementwise_affine=False, eps=1e-6)
        self.pose_branch = Mlp(in_features=dim_in, hidden_features=dim_in // 2, out_features=self.target_dim, drop=0)

    def _build_token_transformer_layers(self, dim_in: int, num_heads: int = 8, 
                                       num_layers: int = 3, dropout: float = 0.1) -> nn.ModuleList:
        """
        构建多层Transformer用于token聚合
        每层包含：Attention + MLP + LayerNorm + 残差连接
        
        Args:
            dim_in: 输入维度
            num_heads: 注意力头数
            num_layers: Transformer层数
            dropout: dropout比例
            
        Returns:
            nn.ModuleList: Transformer层列表
        """
        layers = nn.ModuleList()
        
        for i in range(num_layers):
            layer = nn.ModuleDict({
                # 多头注意力
                'attention': nn.MultiheadAttention(
                    dim_in, num_heads=num_heads, dropout=dropout, batch_first=True
                ),
                'norm1': nn.LayerNorm(dim_in),
                'dropout1': nn.Dropout(dropout),
                
                # MLP
                'mlp': Mlp(
                    in_features=dim_in, 
                    hidden_features=dim_in * 4,  # 4倍扩展，标准Transformer设置
                    out_features=dim_in,
                    drop=dropout
                ),
                'norm2': nn.LayerNorm(dim_in),
                'dropout2': nn.Dropout(dropout),
            })
            layers.append(layer)
        
        return layers

    def aggregate_tokens(self, all_tokens: torch.Tensor) -> torch.Tensor:
        """
        Aggregate all tokens into a single representation for wrist pose prediction.
        
        Args:
            all_tokens: [B, S, num_tokens, C] - all tokens from aggregator
            
        Returns:
            aggregated_tokens: [B, S, C] - single aggregated token per sequence
        """
        B, S, num_tokens, C = all_tokens.shape
        
        if self.token_aggregation == "mean":
            # Simple mean pooling
            return all_tokens.mean(dim=2)  # [B, S, C]
            
        elif self.token_aggregation == "weighted_mean":
            # Learnable weighted mean
            if self.token_weights.size(-1) != num_tokens:
                # Expand weights to match token count
                self.token_weights.data = self.token_weights.data.expand(-1, -1, num_tokens)
            
            weights = F.softmax(self.token_weights, dim=-1)  # [1, 1, num_tokens]
            return torch.sum(all_tokens * weights.unsqueeze(-1), dim=2)  # [B, S, C]
            
        elif self.token_aggregation == "attention":
            # Multi-layer Transformer-based aggregation
            # Reshape for attention: [B*S, num_tokens, C]
            tokens_flat = all_tokens.view(B * S, num_tokens, C)
            
            # Expand query: [B*S, 1, C]
            query = self.token_query.expand(B * S, -1, -1)
            
            # 通过多层Transformer处理
            # 将query和tokens拼接，query在最前面
            combined_tokens = torch.cat([query, tokens_flat], dim=1)  # [B*S, 1+num_tokens, C]
            
            # 依次通过每个Transformer层
            for layer in self.token_aggregation_layers:
                # Self-attention with residual connection
                attn_output, _ = layer['attention'](
                    combined_tokens, combined_tokens, combined_tokens
                )
                combined_tokens = layer['norm1'](combined_tokens + layer['dropout1'](attn_output))
                
                # MLP with residual connection  
                mlp_output = layer['mlp'](combined_tokens)
                combined_tokens = layer['norm2'](combined_tokens + layer['dropout2'](mlp_output))
            
            # 提取query token作为最终聚合结果: [B*S, 1, C]
            aggregated_query = combined_tokens[:, 0:1, :]
            
            # Reshape back: [B, S, C]
            return aggregated_query.view(B, S, C)
        else:
            raise ValueError(f"Unknown token aggregation method: {self.token_aggregation}")

    def forward(self, aggregated_tokens_list: list, num_iterations: int = 4) -> list:
        """
        Forward pass to predict wrist camera parameters using concatenated tokens from two views.

        Args:
            aggregated_tokens_list (list): List of token tensors from the network;
                the last tensor is used for prediction. Expected shape: [B, S, num_tokens, C]
                where S=2 represents two views (ext1 and ext2).
            num_iterations (int, optional): Number of iterative refinement steps. Defaults to 4.

        Returns:
            list: A list of predicted wrist encodings (post-activation) from each iteration.
                Each element has shape [B, 1, target_dim] - single wrist pose per batch.
        """
        # Use tokens from the last block for wrist prediction.
        tokens = aggregated_tokens_list[-1]  # [B, S, num_tokens, C]
        
        # 🔥 NEW: Concatenate tokens from two views along the sequence dimension
        # tokens shape: [B, S, num_tokens, C] where S=2 (ext1, ext2)
        B, S, num_tokens, C = tokens.shape
        # assert S == 2, f"Expected 2 views (S=2), got S={S}"
        
        # Reshape to concatenate tokens from both views
        # [B, S, num_tokens, C] -> [B, S*num_tokens, C]
        tokens_concatenated = tokens.view(B, S * num_tokens, C)
        
        # Aggregate concatenated tokens into a single representation
        wrist_tokens = self.aggregate_tokens(tokens_concatenated.unsqueeze(1))  # [B, 1, C]
        wrist_tokens = self.token_norm(wrist_tokens)

        pred_wrist_enc_list = self.trunk_fn(wrist_tokens, num_iterations)
        return pred_wrist_enc_list

    def trunk_fn(self, wrist_tokens: torch.Tensor, num_iterations: int) -> list:
        """
        Iteratively refine wrist pose predictions.

        Args:
            wrist_tokens (torch.Tensor): Aggregated wrist tokens with shape [B, 1, C].
            num_iterations (int): Number of refinement iterations.

        Returns:
            list: List of activated wrist encodings from each iteration.
        """
        B, S, C = wrist_tokens.shape
        assert S == 1, f"Expected single wrist pose (S=1), got S={S}"
        
        pred_wrist_enc = None
        pred_wrist_enc_list = []

        for _ in range(num_iterations):
            # Use a learned empty pose for the first iteration.
            if pred_wrist_enc is None:
                module_input = self.embed_pose(self.empty_pose_tokens.expand(B, S, -1))
            else:
                # Detach the previous prediction to avoid backprop through time.
                pred_wrist_enc = pred_wrist_enc.detach()
                module_input = self.embed_pose(pred_wrist_enc)

            # Generate modulation parameters and split them into shift, scale, and gate components.
            shift_msa, scale_msa, gate_msa = self.poseLN_modulation(module_input).chunk(3, dim=-1)

            # Adaptive layer normalization and modulation.
            wrist_tokens_modulated = gate_msa * modulate(self.adaln_norm(wrist_tokens), shift_msa, scale_msa)
            wrist_tokens_modulated = wrist_tokens_modulated + wrist_tokens

            wrist_tokens_modulated = self.trunk(wrist_tokens_modulated)
            # Compute the delta update for the wrist encoding.
            pred_wrist_enc_delta = self.pose_branch(self.trunk_norm(wrist_tokens_modulated))

            if pred_wrist_enc is None:
                pred_wrist_enc = pred_wrist_enc_delta
            else:
                pred_wrist_enc = pred_wrist_enc + pred_wrist_enc_delta

            # Apply final activation functions for translation, quaternion, and field-of-view.
            activated_wrist = activate_pose(
                pred_wrist_enc, trans_act=self.trans_act, quat_act=self.quat_act, fl_act=self.fl_act
            )
            pred_wrist_enc_list.append(activated_wrist)

        return pred_wrist_enc_list


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """
    Modulate the input tensor using scaling and shifting parameters.
    """
    # modified from https://github.com/facebookresearch/DiT/blob/796c29e532f47bba17c5b9c5eb39b9354b8b7c64/models.py#L19
    return x * (1 + scale) + shift

    def encode_gt_wrist_pose(self, wrist_extrinsics, wrist_intrinsics, image_hw):
        """
        Encode ground truth wrist pose using the same encoding as predictions.
        
        Args:
            wrist_extrinsics (torch.Tensor): Ground truth wrist extrinsics [B, 4, 4]
            wrist_intrinsics (torch.Tensor): Ground truth wrist intrinsics [B, 3, 3]
            image_hw (tuple): Image height and width
            
        Returns:
            torch.Tensor: Encoded wrist pose [B, pose_dim]
        """
        return extri_intri_to_pose_encoding(
            wrist_extrinsics, 
            wrist_intrinsics, 
            image_hw, 
            pose_encoding_type=self.pose_encoding_type
        ) 