# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin  # used for model hub

from vggt.models.aggregator import Aggregator
from vggt.heads.camera_head import CameraHead
from vggt.heads.dpt_head import DPTHead
from vggt.heads.track_head import TrackHead
from vggt.heads.wrist_head import WristHead


class VGGT(nn.Module, PyTorchModelHubMixin):
    def __init__(self, img_size=518, patch_size=14, embed_dim=1024,
                 enable_camera=True, enable_point=True, enable_depth=True, enable_track=True, enable_wrist=False,
                 wrist_head_config=None, pretrained=None, use_lora=False, lora_rank=16, lora_alpha=32):
        super().__init__()

        self.aggregator = Aggregator(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim)

        self.camera_head = CameraHead(dim_in=2 * embed_dim) if enable_camera else None
        self.point_head = DPTHead(dim_in=2 * embed_dim, output_dim=4, activation="inv_log", conf_activation="expp1") if enable_point else None
        self.depth_head = DPTHead(dim_in=2 * embed_dim, output_dim=2, activation="exp", conf_activation="expp1") if enable_depth else None
        self.track_head = TrackHead(dim_in=2 * embed_dim, patch_size=patch_size) if enable_track else None
        
        # 🔥 NEW: 支持自定义WristHead配置
        if enable_wrist:
            wrist_kwargs = {"dim_in": 2 * embed_dim}
            if wrist_head_config:
                wrist_kwargs.update(wrist_head_config)
            self.wrist_head = WristHead(**wrist_kwargs)
        else:
            self.wrist_head = None
        
        # Load pretrained model if specified
        if pretrained is not None:
            self._load_pretrained(pretrained)
            
        # Apply LoRA if enabled
        if use_lora:
            self._setup_lora(lora_rank, lora_alpha)
    
    def _load_pretrained(self, pretrained_name):
        """Load pretrained model weights"""
        try:
            print(f"🔄 Loading pretrained model: {pretrained_name}")
            pretrained_model = VGGT.from_pretrained(pretrained_name)
            
            # Load compatible weights
            pretrained_dict = pretrained_model.state_dict()
            model_dict = self.state_dict()
            
            # Filter compatible keys
            compatible_dict = {}
            skipped_keys = []
            for k, v in pretrained_dict.items():
                if k in model_dict and model_dict[k].shape == v.shape:
                    compatible_dict[k] = v
                else:
                    skipped_keys.append(k)
                    # 注释掉每个跳过key的详细输出
                    # print(f"Skipping {k}: shape mismatch or missing")
                    
            model_dict.update(compatible_dict)
            self.load_state_dict(model_dict, strict=False)
            
            # 简化输出，只显示汇总信息
            print(f"✅ Loaded {len(compatible_dict)}/{len(pretrained_dict)} pretrained weights")
            if skipped_keys:
                print(f"⚠️  Skipped {len(skipped_keys)} incompatible keys")
            
        except Exception as e:
            print(f"⚠️  Warning: Failed to load pretrained model {pretrained_name}: {e}")
            print("🔄 Continuing with random initialization...")
    
    def _setup_lora(self, lora_rank, lora_alpha):
        """Setup LoRA for efficient fine-tuning"""
        try:
            from peft import LoraConfig, get_peft_model
            
            # Configure LoRA for attention layers
            lora_config = LoraConfig(
                r=lora_rank,
                lora_alpha=lora_alpha,
                target_modules=["to_q", "to_k", "to_v", "to_out"],  # Common attention module names
                lora_dropout=0.1,
                bias="none",
                task_type="FEATURE_EXTRACTION"
            )
            
            # Apply LoRA to the aggregator (main feature extractor)
            self.aggregator = get_peft_model(self.aggregator, lora_config)
            
            print(f"LoRA applied with rank={lora_rank}, alpha={lora_alpha}")
            
        except ImportError:
            print("Warning: peft library not found. Install with: pip install peft")
            print("Continuing without LoRA...")
        except Exception as e:
            print(f"Warning: Failed to setup LoRA: {e}")
            print("Continuing without LoRA...")

    def forward(self, images: torch.Tensor, query_points: torch.Tensor = None, only_wrist: bool = False):
        """
        Forward pass of the VGGT model.

        Args:
            images (torch.Tensor): Input images with shape [S, 3, H, W] or [B, S, 3, H, W], in range [0, 1].
                B: batch size, S: sequence length, 3: RGB channels, H: height, W: width
            query_points (torch.Tensor, optional): Query points for tracking, in pixel coordinates.
                Shape: [N, 2] or [B, N, 2], where N is the number of query points.
                Default: None

        Returns:
            dict: A dictionary containing the following predictions:
                - pose_enc (torch.Tensor): Camera pose encoding with shape [B, S, 9] (from the last iteration)
                - depth (torch.Tensor): Predicted depth maps with shape [B, S, H, W, 1]
                - depth_conf (torch.Tensor): Confidence scores for depth predictions with shape [B, S, H, W]
                - world_points (torch.Tensor): 3D world coordinates for each pixel with shape [B, S, H, W, 3]
                - world_points_conf (torch.Tensor): Confidence scores for world points with shape [B, S, H, W]
                - images (torch.Tensor): Original input images, preserved for visualization

                If query_points is provided, also includes:
                - track (torch.Tensor): Point tracks with shape [B, S, N, 2] (from the last iteration), in pixel coordinates
                - vis (torch.Tensor): Visibility scores for tracked points with shape [B, S, N]
                - conf (torch.Tensor): Confidence scores for tracked points with shape [B, S, N]
        """        
        # If without batch dimension, add it
        if len(images.shape) == 4:
            images = images.unsqueeze(0)
            
        if query_points is not None and len(query_points.shape) == 2:
            query_points = query_points.unsqueeze(0)
        aggregated_tokens_list, patch_start_idx = self.aggregator(images)

        predictions = {}

        with torch.cuda.amp.autocast(enabled=False):
            if self.camera_head is not None:
                pose_enc_list = self.camera_head(aggregated_tokens_list)
                predictions["pose_enc"] = pose_enc_list[-1]  # pose encoding of the last iteration
                predictions["pose_enc_list"] = pose_enc_list
                
            if self.depth_head is not None:
                depth, depth_conf = self.depth_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )
                predictions["depth"] = depth
                predictions["depth_conf"] = depth_conf

            if self.point_head is not None:
                pts3d, pts3d_conf = self.point_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )
                predictions["world_points"] = pts3d
                predictions["world_points_conf"] = pts3d_conf
                
            if self.wrist_head is not None:
                wrist_pose_enc_list = self.wrist_head(aggregated_tokens_list)
                # 🔥 NEW: wrist_head now outputs single wrist pose [B, 1, target_dim] instead of [B, S, target_dim]
                predictions["wrist_pose_enc"] = wrist_pose_enc_list[-1]  # wrist pose encoding of the last iteration
                predictions["wrist_pose_enc_list"] = wrist_pose_enc_list

        if self.track_head is not None and query_points is not None:
            track_list, vis, conf = self.track_head(
                aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx, query_points=query_points
            )
            predictions["track"] = track_list[-1]  # track of the last iteration
            predictions["vis"] = vis
            predictions["conf"] = conf

        if not self.training:
            predictions["images"] = images  # store the images for visualization during inference

        return predictions

