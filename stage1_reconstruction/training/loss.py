# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F

from dataclasses import dataclass
from vggt.utils.pose_enc import extri_intri_to_pose_encoding
from training.train_utils.general import check_and_fix_inf_nan
from math import ceil, floor
import logging
from typing import Dict, Optional, Tuple


@dataclass(eq=False)
class MultitaskLoss(torch.nn.Module):
    """
    Multi-task loss module that combines different loss types for VGGT.
    
    Supports:
    - Camera loss
    - Depth loss 
    - Point loss
    - Tracking loss (not cleaned yet, dirty code is at the bottom of this file)
    """
    def __init__(self, camera=None, depth=None, point=None, track=None, wrist=None, projection=None, **kwargs):
        super().__init__()
        # Loss configuration dictionaries for each task
        self.camera = camera
        self.depth = depth
        self.point = point
        self.track = track
        self.wrist = wrist
        self.projection = projection

    def forward(self, predictions, batch) -> torch.Tensor:
        """
        Compute the total multi-task loss.
        
        Args:
            predictions: Dict containing model predictions for different tasks
            batch: Dict containing ground truth data and masks
            
        Returns:
            Dict containing individual losses and total objective
        """
        total_loss = 0
        loss_dict = {}
        
        # Camera pose loss - if pose encodings are predicted
        if "pose_enc_list" in predictions:
            camera_loss_dict = compute_camera_loss(predictions, batch, **self.camera)   
            camera_loss = camera_loss_dict["loss_camera"] * self.camera["weight"]   
            total_loss = total_loss + camera_loss
            loss_dict.update(camera_loss_dict)
        
        # Depth estimation loss - compute only if GT depth is available and configured
        if "depth" in predictions and 'depths' in batch.keys():
            depth_loss_dict = compute_depth_loss(predictions, batch, **self.depth)
            depth_loss = depth_loss_dict["loss_conf_depth"] + depth_loss_dict["loss_reg_depth"] + depth_loss_dict["loss_grad_depth"]
            depth_loss = depth_loss * self.depth["weight"]
            total_loss = total_loss + depth_loss
            loss_dict.update(depth_loss_dict)

        # 3D point reconstruction loss - compute only if GT point cloud is available and configured
        if "world_points" in predictions and "world_points" in batch.keys():
            point_loss_dict = compute_point_loss(predictions, batch, **self.point)
            point_loss = point_loss_dict["loss_conf_point"] + point_loss_dict["loss_reg_point"] + point_loss_dict["loss_grad_point"]
            point_loss = point_loss * self.point["weight"]
            total_loss = total_loss + point_loss
            loss_dict.update(point_loss_dict)

        # Tracking loss - not cleaned yet, dirty code is at the bottom of this file
        if "track" in predictions:
            raise NotImplementedError("Track loss is not cleaned up yet")
            
        # Projection loss - for wrist track implementation
        if "track_pairs" in batch and self.projection is not None:
            projection_loss_dict = compute_projection_loss(predictions, batch, **self.projection)
            projection_loss = projection_loss_dict["loss_projection"] * self.projection["weight"]
            total_loss = total_loss + projection_loss
            loss_dict.update(projection_loss_dict)
            
        # Wrist pose loss - if wrist pose encodings are predicted  
        if "wrist_pose_enc_list" in predictions and self.wrist is not None:
            wrist_loss_dict = compute_wrist_loss(predictions, batch, **self.wrist)   
            wrist_loss = wrist_loss_dict["loss_wrist"] * self.wrist["weight"]   
            total_loss = total_loss + wrist_loss
            loss_dict.update(wrist_loss_dict)
        
        loss_dict["objective"] = total_loss

        return loss_dict


def compute_camera_loss(
    pred_dict,              # predictions dict, contains pose encodings
    batch_data,             # ground truth and mask batch dict
    loss_type="l1",         # "l1" or "l2" loss
    gamma=0.6,              # temporal decay weight for multi-stage training
    pose_encoding_type="absT_quaR_FoV",
    weight_trans=1.0,       # weight for translation loss
    weight_rot=1.0,         # weight for rotation loss
    weight_focal=0.5,       # weight for focal length loss
    **kwargs
):
    # List of predicted pose encodings per stage
    pred_pose_encodings = pred_dict['pose_enc_list']
    # Binary mask for valid points per frame (B, N, H, W)
    if 'point_masks' in batch_data.keys():
        point_masks = batch_data['point_masks']
        valid_frame_mask = point_masks[:, 0].sum(dim=[-1, -2]) > 100
    else:
        point_masks = None
        valid_frame_mask = torch.ones_like(pred_pose_encodings[0][:, :]).to(torch.bool)
    # Only consider frames with enough valid points (>100)
    
    # Number of prediction stages
    n_stages = len(pred_pose_encodings)

    # Get ground truth camera extrinsics and intrinsics
    gt_extrinsics = batch_data['extrinsics']
    gt_intrinsics = batch_data['intrinsics']
    
    image_hw = batch_data['images'].shape[-2:]
    # Encode ground truth pose to match predicted encoding format
    gt_pose_encoding = extri_intri_to_pose_encoding(
        gt_extrinsics, gt_intrinsics, image_hw, pose_encoding_type=pose_encoding_type
    )

    # Initialize loss accumulators for translation, rotation, focal length
    total_loss_T = total_loss_R = total_loss_FL = 0

    # Compute loss for each prediction stage with temporal weighting
    for stage_idx in range(n_stages):
        # Later stages get higher weight (gamma^0 = 1.0 for final stage)
        stage_weight = gamma ** (n_stages - stage_idx - 1)
        pred_pose_stage = pred_pose_encodings[stage_idx]

        if valid_frame_mask.sum() == 0:
            # If no valid frames, set losses to zero to avoid gradient issues
            loss_T_stage = (pred_pose_stage * 0).mean()
            loss_R_stage = (pred_pose_stage * 0).mean()
            loss_FL_stage = (pred_pose_stage * 0).mean()
        else:
            # Only consider valid frames for loss computation
            loss_T_stage, loss_R_stage, loss_FL_stage = camera_loss_single(
                pred_pose_stage[valid_frame_mask].clone(),
                gt_pose_encoding[valid_frame_mask].clone(),
                loss_type=loss_type
            )
        # Accumulate weighted losses across stages
        total_loss_T += loss_T_stage * stage_weight
        total_loss_R += loss_R_stage * stage_weight
        total_loss_FL += loss_FL_stage * stage_weight

    # Average over all stages
    avg_loss_T = total_loss_T / n_stages
    avg_loss_R = total_loss_R / n_stages
    avg_loss_FL = total_loss_FL / n_stages

    # Compute total weighted camera loss
    total_camera_loss = (
        avg_loss_T * weight_trans +
        avg_loss_R * weight_rot +
        avg_loss_FL * weight_focal
    )

    # Return loss dictionary with individual components
    return {
        "loss_camera": total_camera_loss,
        "loss_T": avg_loss_T,
        "loss_R": avg_loss_R,
        "loss_FL": avg_loss_FL
    }

def camera_loss_single(pred_pose_enc, gt_pose_enc, loss_type="l1"):
    """
    Computes translation, rotation, and focal loss for a batch of pose encodings.
    
    Args:
        pred_pose_enc: (N, D) predicted pose encoding
        gt_pose_enc: (N, D) ground truth pose encoding
        loss_type: "l1" (abs error) or "l2" (euclidean error)
    Returns:
        loss_T: translation loss (mean)
        loss_R: rotation loss (mean)
        loss_FL: focal length/intrinsics loss (mean)
    
    NOTE: The paper uses smooth l1 loss, but we found l1 loss is more stable than smooth l1 and l2 loss.
        So here we use l1 loss.
    """
    if loss_type == "l1":
        # Translation: first 3 dims; Rotation: next 4 (quaternion); Focal/Intrinsics: last dims
        loss_T = (pred_pose_enc[..., :3] - gt_pose_enc[..., :3]).abs()
        loss_R = (pred_pose_enc[..., 3:7] - gt_pose_enc[..., 3:7]).abs()
        loss_FL = (pred_pose_enc[..., 7:] - gt_pose_enc[..., 7:]).abs()
    elif loss_type == "l2":
        # L2 norm for each component
        loss_T = (pred_pose_enc[..., :3] - gt_pose_enc[..., :3]).norm(dim=-1, keepdim=True)
        loss_R = (pred_pose_enc[..., 3:7] - gt_pose_enc[..., 3:7]).norm(dim=-1)
        loss_FL = (pred_pose_enc[..., 7:] - gt_pose_enc[..., 7:]).norm(dim=-1)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

    # Check/fix numerical issues (nan/inf) for each loss component
    loss_T = check_and_fix_inf_nan(loss_T, "loss_T")
    loss_R = check_and_fix_inf_nan(loss_R, "loss_R")
    loss_FL = check_and_fix_inf_nan(loss_FL, "loss_FL")

    # Clamp outlier translation loss to prevent instability, then average
    loss_T = loss_T.clamp(max=100).mean()
    loss_R = loss_R.mean()
    loss_FL = loss_FL.mean()

    return loss_T, loss_R, loss_FL


def compute_wrist_loss(
    pred_dict,              # predictions dict, contains wrist pose encodings
    batch_data,             # ground truth and mask batch dict
    loss_type="l1",         # "l1" or "l2" loss
    gamma=0.6,              # temporal decay weight for multi-stage training
    pose_encoding_type="absT_quaR_FoV",
    weight_trans=1.0,       # weight for translation loss - set to 0 to only supervise intrinsic
    weight_rot=1.0,         # weight for rotation loss - set to 0 to only supervise intrinsic
    weight_focal=1.0,       # weight for focal length loss - only this is used now
    **kwargs
):
    """
    Compute wrist camera pose loss.
    
    Args:
        pred_dict: Dictionary containing predicted wrist pose encodings
        batch_data: Dictionary containing ground truth wrist extrinsics and intrinsics
        loss_type: Type of loss ("l1" or "l2")
        gamma: Temporal decay weight for multi-stage training
        pose_encoding_type: Type of pose encoding
        weight_trans: Weight for translation loss (set to 0 to only supervise intrinsic)
        weight_rot: Weight for rotation loss (set to 0 to only supervise intrinsic)
        weight_focal: Weight for focal length loss (only this is used now)
        
    Returns:
        Dictionary containing wrist loss components
    """
    from vggt.utils.pose_enc import extri_intri_to_pose_encoding
    
    # List of predicted wrist pose encodings per stage
    pred_wrist_pose_encodings = pred_dict['wrist_pose_enc_list']
    n_stages = len(pred_wrist_pose_encodings)

    # Get ground truth wrist extrinsics and intrinsics
    gt_wrist_extrinsics = batch_data['wrist_extrinsics']  # [B, 4, 4]
    gt_wrist_intrinsics = batch_data['wrist_intrinsics']  # [B, 3, 3]
    image_hw = batch_data['images'].shape[-2:]

    # Encode ground truth wrist pose to match predicted encoding format
    gt_wrist_pose_encoding = extri_intri_to_pose_encoding(
        gt_wrist_extrinsics, gt_wrist_intrinsics, image_hw, pose_encoding_type=pose_encoding_type
    )
    # Initialize loss accumulators
    total_loss_T = total_loss_R = total_loss_FL = 0

    # Compute loss for each prediction stage with temporal weighting
    for stage_idx in range(n_stages):
        # Later stages get higher weight (gamma^0 = 1.0 for final stage)
        stage_weight = gamma ** (n_stages - stage_idx - 1)
        pred_wrist_pose_stage = pred_wrist_pose_encodings[stage_idx]
        
        # 🔥 NEW: Ensure both pred and GT have shape [B, 1, target_dim]
        assert pred_wrist_pose_stage.size(1) == 1, f"Expected pred wrist pose to have S=1, got {pred_wrist_pose_stage.size(1)}"
        assert gt_wrist_pose_encoding.size(1) == 1, f"Expected GT wrist pose to have S=1, got {gt_wrist_pose_encoding.size(1)}"
        
        # Compute loss for this stage

        loss_T_stage, loss_R_stage, loss_FL_stage = camera_loss_single(
            pred_wrist_pose_stage,
            gt_wrist_pose_encoding,
            loss_type=loss_type
        )
        
        # Accumulate weighted losses across stages
        total_loss_T += loss_T_stage * stage_weight
        total_loss_R += loss_R_stage * stage_weight
        total_loss_FL += loss_FL_stage * stage_weight

    # Average over all stages
    avg_loss_T = total_loss_T / n_stages
    avg_loss_R = total_loss_R / n_stages
    avg_loss_FL = total_loss_FL / n_stages

    # Compute total weighted wrist loss - only focal length loss is used
    total_wrist_loss = (
        avg_loss_T * weight_trans +
        avg_loss_R * weight_rot +
        avg_loss_FL * weight_focal
    )

    # Return loss dictionary with individual components
    return {
        "loss_wrist": total_wrist_loss,
        "loss_wrist_T": avg_loss_T,
        "loss_wrist_R": avg_loss_R,
        "loss_wrist_FL": avg_loss_FL
    }


def compute_point_loss(predictions, batch, gamma=1.0, alpha=0.2, gradient_loss_fn = None, valid_range=-1, **kwargs):
    """
    Compute point loss.
    
    Args:
        predictions: Dict containing 'world_points' and 'world_points_conf'
        batch: Dict containing ground truth 'world_points' and 'point_masks'
        gamma: Weight for confidence loss
        alpha: Weight for confidence regularization
        gradient_loss_fn: Type of gradient loss to apply
        valid_range: Quantile range for outlier filtering
    """
    pred_points = predictions['world_points']
    pred_points_conf = predictions['world_points_conf']
    gt_points = batch['world_points']
    gt_points_mask = batch['point_masks']
    
    gt_points = check_and_fix_inf_nan(gt_points, "gt_points")
    
    if gt_points_mask.sum() < 100:
        # If there are less than 100 valid points, skip this batch
        dummy_loss = (0.0 * pred_points).mean()
        loss_dict = {f"loss_conf_point": dummy_loss,
                    f"loss_reg_point": dummy_loss,
                    f"loss_grad_point": dummy_loss,}
        return loss_dict
    
    # Compute confidence-weighted regression loss with optional gradient loss
    loss_conf, loss_grad, loss_reg = regression_loss(pred_points, gt_points, gt_points_mask, conf=pred_points_conf,
                                             gradient_loss_fn=gradient_loss_fn, gamma=gamma, alpha=alpha, valid_range=valid_range)
    
    loss_dict = {
        f"loss_conf_point": loss_conf,
        f"loss_reg_point": loss_reg,
        f"loss_grad_point": loss_grad,
    }
    
    return loss_dict


def compute_depth_loss(predictions, batch, gamma=1.0, alpha=0.2, gradient_loss_fn = None, valid_range=-1, **kwargs):
    """
    Compute depth loss.
    
    Args:
        predictions: Dict containing 'depth' and 'depth_conf'
        batch: Dict containing ground truth 'depths', 'point_masks', and 'depth_nan_masks'
        gamma: Weight for confidence loss
        alpha: Weight for confidence regularization
        gradient_loss_fn: Type of gradient loss to apply
        valid_range: Quantile range for outlier filtering
    """
    pred_depth = predictions['depth']
    pred_depth_conf = predictions['depth_conf']

    gt_depth = batch['depths']
    gt_depth = check_and_fix_inf_nan(gt_depth, "gt_depth")
    gt_depth = gt_depth[..., None]              # (B, S, H, W, 1)
    
    # 原有的point_masks表示基于深度值的有效性
    point_masks = batch['point_masks'].clone()   # (B, S, H, W)
    
    # 新增：depth_nan_masks表示原始数据中的NaN/Inf位置，这些位置不应参与监督
    if 'depth_nan_masks' in batch:
        depth_nan_masks = batch['depth_nan_masks']  # (B, S, H, W) - True表示原始NaN位置
        # 创建最终的有效性mask：既要有有效深度值，又不能是原始NaN位置
        gt_depth_mask = point_masks & (~depth_nan_masks)  # 排除NaN位置
        
    else:
        # 如果没有depth_nan_masks，使用原来的逻辑
        gt_depth_mask = point_masks
        logging.warning("No depth_nan_masks found in batch, using original point_masks only")

    if gt_depth_mask.sum() < 100:
        # If there are less than 100 valid points, skip this batch
        dummy_loss = (0.0 * pred_depth).mean()
        loss_dict = {f"loss_conf_depth": dummy_loss,
                    f"loss_reg_depth": dummy_loss,
                    f"loss_grad_depth": dummy_loss,}
        return loss_dict

    # NOTE: we put conf inside regression_loss so that we can also apply conf loss to the gradient loss in a multi-scale manner
    # this is hacky, but very easier to implement
    loss_conf, loss_grad, loss_reg = regression_loss(pred_depth, gt_depth, gt_depth_mask, conf=pred_depth_conf,
                                             gradient_loss_fn=gradient_loss_fn, gamma=gamma, alpha=alpha, valid_range=valid_range)

    loss_dict = {
        f"loss_conf_depth": loss_conf,
        f"loss_reg_depth": loss_reg,    
        f"loss_grad_depth": loss_grad,
    }

    return loss_dict


def regression_loss(pred, gt, mask, conf=None, gradient_loss_fn=None, gamma=1.0, alpha=0.2, valid_range=-1):
    """
    Core regression loss function with confidence weighting and optional gradient loss.
    
    Computes:
    1. gamma * ||pred - gt||^2 * conf - alpha * log(conf)
    2. Optional gradient loss
    
    Args:
        pred: (B, S, H, W, C) predicted values
        gt: (B, S, H, W, C) ground truth values
        mask: (B, S, H, W) valid pixel mask
        conf: (B, S, H, W) confidence weights (optional)
        gradient_loss_fn: Type of gradient loss ("normal", "grad", etc.)
        gamma: Weight for confidence loss
        alpha: Weight for confidence regularization
        valid_range: Quantile range for outlier filtering
    
    Returns:
        loss_conf: Confidence-weighted loss
        loss_grad: Gradient loss (0 if not specified)
        loss_reg: Regular L2 loss
    """
    bb, ss, hh, ww, nc = pred.shape

    # Compute L2 distance between predicted and ground truth points
    loss_reg = torch.norm(gt[mask] - pred[mask], dim=-1)
    loss_reg = check_and_fix_inf_nan(loss_reg, "loss_reg")

    # Confidence-weighted loss: gamma * loss * conf - alpha * log(conf)
    # This encourages the model to be confident on easy examples and less confident on hard ones
    loss_conf = gamma * loss_reg * conf[mask] - alpha * torch.log(conf[mask])
    loss_conf = check_and_fix_inf_nan(loss_conf, "loss_conf")
        
    # Initialize gradient loss
    loss_grad = 0

    # Prepare confidence for gradient loss if needed
    if gradient_loss_fn is not None and "conf" in gradient_loss_fn:
        to_feed_conf = conf.reshape(bb*ss, hh, ww)
    else:
        to_feed_conf = None

    # Compute gradient loss if specified for spatial smoothness
    if gradient_loss_fn is not None and "normal" in gradient_loss_fn:
        # Surface normal-based gradient loss
        loss_grad = gradient_loss_multi_scale_wrapper(
            pred.reshape(bb*ss, hh, ww, nc),
            gt.reshape(bb*ss, hh, ww, nc),
            mask.reshape(bb*ss, hh, ww),
            gradient_loss_fn=normal_loss,
            scales=3,
            conf=to_feed_conf,
        )
    elif gradient_loss_fn is not None and "grad" in gradient_loss_fn:
        # Standard gradient-based loss
        loss_grad = gradient_loss_multi_scale_wrapper(
            pred.reshape(bb*ss, hh, ww, nc),
            gt.reshape(bb*ss, hh, ww, nc),
            mask.reshape(bb*ss, hh, ww),
            gradient_loss_fn=gradient_loss,
            conf=to_feed_conf,
        )

    # Process confidence-weighted loss
    if loss_conf.numel() > 0:
        # Filter out outliers using quantile-based thresholding
        if valid_range>0:
            loss_conf = filter_by_quantile(loss_conf, valid_range)

        loss_conf = check_and_fix_inf_nan(loss_conf, f"loss_conf_depth")
        loss_conf = loss_conf.mean()
    else:
        loss_conf = (0.0 * pred).mean()

    # Process regular regression loss
    if loss_reg.numel() > 0:
        # Filter out outliers using quantile-based thresholding
        if valid_range>0:
            loss_reg = filter_by_quantile(loss_reg, valid_range)

        loss_reg = check_and_fix_inf_nan(loss_reg, f"loss_reg_depth")
        loss_reg = loss_reg.mean()
    else:
        loss_reg = (0.0 * pred).mean()

    return loss_conf, loss_grad, loss_reg


def gradient_loss_multi_scale_wrapper(prediction, target, mask, scales=4, gradient_loss_fn = None, conf=None):
    """
    Multi-scale gradient loss wrapper. Applies gradient loss at multiple scales by subsampling the input.
    This helps capture both fine and coarse spatial structures.
    
    Args:
        prediction: (B, H, W, C) predicted values
        target: (B, H, W, C) ground truth values  
        mask: (B, H, W) valid pixel mask
        scales: Number of scales to use
        gradient_loss_fn: Gradient loss function to apply
        conf: (B, H, W) confidence weights (optional)
    """
    total = 0
    for scale in range(scales):
        step = pow(2, scale)  # Subsample by 2^scale

        total += gradient_loss_fn(
            prediction[:, ::step, ::step],
            target[:, ::step, ::step],
            mask[:, ::step, ::step],
            conf=conf[:, ::step, ::step] if conf is not None else None
        )

    total = total / scales
    return total


def normal_loss(prediction, target, mask, cos_eps=1e-8, conf=None, gamma=1.0, alpha=0.2):
    """
    Surface normal-based loss for geometric consistency.
    
    Computes surface normals from 3D point maps using cross products of neighboring points,
    then measures the angle between predicted and ground truth normals.
    
    Args:
        prediction: (B, H, W, 3) predicted 3D coordinates/points
        target: (B, H, W, 3) ground-truth 3D coordinates/points
        mask: (B, H, W) valid pixel mask
        cos_eps: Epsilon for numerical stability in cosine computation
        conf: (B, H, W) confidence weights (optional)
        gamma: Weight for confidence loss
        alpha: Weight for confidence regularization
    """
    # Convert point maps to surface normals using cross products
    pred_normals, pred_valids = point_map_to_normal(prediction, mask, eps=cos_eps)
    gt_normals,   gt_valids   = point_map_to_normal(target,     mask, eps=cos_eps)

    # Only consider regions where both predicted and GT normals are valid
    all_valid = pred_valids & gt_valids  # shape: (4, B, H, W)

    # Early return if not enough valid points
    divisor = torch.sum(all_valid)
    if divisor < 10:
        return 0

    # Extract valid normals
    pred_normals = pred_normals[all_valid].clone()
    gt_normals = gt_normals[all_valid].clone()

    # Compute cosine similarity between corresponding normals
    dot = torch.sum(pred_normals * gt_normals, dim=-1)

    # Clamp dot product to [-1, 1] for numerical stability
    dot = torch.clamp(dot, -1 + cos_eps, 1 - cos_eps)

    # Compute loss as 1 - cos(theta), instead of arccos(dot) for numerical stability
    loss = 1 - dot

    # Return mean loss if we have enough valid points
    if loss.numel() < 10:
        return 0
    else:
        loss = check_and_fix_inf_nan(loss, "normal_loss")

        if conf is not None:
            # Apply confidence weighting
            conf = conf[None, ...].expand(4, -1, -1, -1)
            conf = conf[all_valid].clone()

            loss = gamma * loss * conf - alpha * torch.log(conf)
            return loss.mean()
        else:
            return loss.mean()


def gradient_loss(prediction, target, mask, conf=None, gamma=1.0, alpha=0.2):
    """
    Gradient-based loss. Computes the L1 difference between adjacent pixels in x and y directions.
    
    Args:
        prediction: (B, H, W, C) predicted values
        target: (B, H, W, C) ground truth values
        mask: (B, H, W) valid pixel mask
        conf: (B, H, W) confidence weights (optional)
        gamma: Weight for confidence loss
        alpha: Weight for confidence regularization
    """
    # Expand mask to match prediction channels
    mask = mask[..., None].expand(-1, -1, -1, prediction.shape[-1])
    M = torch.sum(mask, (1, 2, 3))

    # Compute difference between prediction and target
    diff = prediction - target
    diff = torch.mul(mask, diff)

    # Compute gradients in x direction (horizontal)
    grad_x = torch.abs(diff[:, :, 1:] - diff[:, :, :-1])
    mask_x = torch.mul(mask[:, :, 1:], mask[:, :, :-1])
    grad_x = torch.mul(mask_x, grad_x)

    # Compute gradients in y direction (vertical)
    grad_y = torch.abs(diff[:, 1:, :] - diff[:, :-1, :])
    mask_y = torch.mul(mask[:, 1:, :], mask[:, :-1, :])
    grad_y = torch.mul(mask_y, grad_y)

    # Clamp gradients to prevent outliers
    grad_x = grad_x.clamp(max=100)
    grad_y = grad_y.clamp(max=100)

    # Apply confidence weighting if provided
    if conf is not None:
        conf = conf[..., None].expand(-1, -1, -1, prediction.shape[-1])
        conf_x = conf[:, :, 1:]
        conf_y = conf[:, 1:, :]

        grad_x = gamma * grad_x * conf_x - alpha * torch.log(conf_x)
        grad_y = gamma * grad_y * conf_y - alpha * torch.log(conf_y)

    # Sum gradients and normalize by number of valid pixels
    grad_loss = torch.sum(grad_x, (1, 2, 3)) + torch.sum(grad_y, (1, 2, 3))
    divisor = torch.sum(M)

    if divisor == 0:
        return 0
    else:
        grad_loss = torch.sum(grad_loss) / divisor

    return grad_loss


def point_map_to_normal(point_map, mask, eps=1e-6):
    """
    Convert 3D point map to surface normal vectors using cross products.
    
    Computes normals by taking cross products of neighboring point differences.
    Uses 4 different cross-product directions for robustness.
    
    Args:
        point_map: (B, H, W, 3) 3D points laid out in a 2D grid
        mask: (B, H, W) valid pixels (bool)
        eps: Epsilon for numerical stability in normalization
    
    Returns:
        normals: (4, B, H, W, 3) normal vectors for each of the 4 cross-product directions
        valids: (4, B, H, W) corresponding valid masks
    """
    with torch.cuda.amp.autocast(enabled=False):
        # Pad inputs to avoid boundary issues
        padded_mask = F.pad(mask, (1, 1, 1, 1), mode='constant', value=0)
        pts = F.pad(point_map.permute(0, 3, 1, 2), (1,1,1,1), mode='constant', value=0).permute(0, 2, 3, 1)

        # Get neighboring points for each pixel
        center = pts[:, 1:-1, 1:-1, :]   # B,H,W,3
        up     = pts[:, :-2,  1:-1, :]
        left   = pts[:, 1:-1, :-2 , :]
        down   = pts[:, 2:,   1:-1, :]
        right  = pts[:, 1:-1, 2:,   :]

        # Compute direction vectors from center to neighbors
        up_dir    = up    - center
        left_dir  = left  - center
        down_dir  = down  - center
        right_dir = right - center

        # Compute four cross products for different normal directions
        n1 = torch.cross(up_dir,   left_dir,  dim=-1)  # up x left
        n2 = torch.cross(left_dir, down_dir,  dim=-1)  # left x down
        n3 = torch.cross(down_dir, right_dir, dim=-1)  # down x right
        n4 = torch.cross(right_dir,up_dir,    dim=-1)  # right x up

        # Validity masks - require both direction pixels to be valid
        v1 = padded_mask[:, :-2,  1:-1] & padded_mask[:, 1:-1, 1:-1] & padded_mask[:, 1:-1, :-2]
        v2 = padded_mask[:, 1:-1, :-2 ] & padded_mask[:, 1:-1, 1:-1] & padded_mask[:, 2:,   1:-1]
        v3 = padded_mask[:, 2:,   1:-1] & padded_mask[:, 1:-1, 1:-1] & padded_mask[:, 1:-1, 2:]
        v4 = padded_mask[:, 1:-1, 2:  ] & padded_mask[:, 1:-1, 1:-1] & padded_mask[:, :-2,  1:-1]

        # Stack normals and validity masks
        normals = torch.stack([n1, n2, n3, n4], dim=0)  # shape [4, B, H, W, 3]
        valids  = torch.stack([v1, v2, v3, v4], dim=0)  # shape [4, B, H, W]

        # Normalize normal vectors
        normals = F.normalize(normals, p=2, dim=-1, eps=eps)

    return normals, valids


def filter_by_quantile(loss_tensor, valid_range, min_elements=1000, hard_max=100):
    """
    Filter loss tensor by keeping only values below a certain quantile threshold.
    
    This helps remove outliers that could destabilize training.
    
    Args:
        loss_tensor: Tensor containing loss values
        valid_range: Float between 0 and 1 indicating the quantile threshold
        min_elements: Minimum number of elements required to apply filtering
        hard_max: Maximum allowed value for any individual loss
    
    Returns:
        Filtered and clamped loss tensor
    """
    if loss_tensor.numel() <= min_elements:
        # Too few elements, just return as-is
        return loss_tensor

    # Randomly sample if tensor is too large to avoid memory issues
    if loss_tensor.numel() > 100000000:
        # Flatten and randomly select 1M elements
        indices = torch.randperm(loss_tensor.numel(), device=loss_tensor.device)[:1_000_000]
        loss_tensor = loss_tensor.view(-1)[indices]

    # First clamp individual values to prevent extreme outliers
    loss_tensor = loss_tensor.clamp(max=hard_max)

    # Compute quantile threshold
    quantile_thresh = torch_quantile(loss_tensor.detach(), valid_range)
    quantile_thresh = min(quantile_thresh, hard_max)

    # Apply quantile filtering if enough elements remain
    quantile_mask = loss_tensor < quantile_thresh
    if quantile_mask.sum() > min_elements:
        return loss_tensor[quantile_mask]
    return loss_tensor


def torch_quantile(
    input,
    q,
    dim = None,
    keepdim: bool = False,
    *,
    interpolation: str = "nearest",
    out: torch.Tensor = None,
) -> torch.Tensor:
    """Better torch.quantile for one SCALAR quantile.

    Using torch.kthvalue. Better than torch.quantile because:
        - No 2**24 input size limit (pytorch/issues/67592),
        - Much faster, at least on big input sizes.

    Arguments:
        input (torch.Tensor): See torch.quantile.
        q (float): See torch.quantile. Supports only scalar input
            currently.
        dim (int | None): See torch.quantile.
        keepdim (bool): See torch.quantile. Supports only False
            currently.
        interpolation: {"nearest", "lower", "higher"}
            See torch.quantile.
        out (torch.Tensor | None): See torch.quantile. Supports only
            None currently.
    """
    # https://github.com/pytorch/pytorch/issues/64947
    # Sanitization: q
    try:
        q = float(q)
        assert 0 <= q <= 1
    except Exception:
        raise ValueError(f"Only scalar input 0<=q<=1 is currently supported (got {q})!")

    # Handle dim=None case
    if dim_was_none := dim is None:
        dim = 0
        input = input.reshape((-1,) + (1,) * (input.ndim - 1))

    # Set interpolation method
    if interpolation == "nearest":
        inter = round
    elif interpolation == "lower":
        inter = floor
    elif interpolation == "higher":
        inter = ceil
    else:
        raise ValueError(
            "Supported interpolations currently are {'nearest', 'lower', 'higher'} "
            f"(got '{interpolation}')!"
        )

    # Validate out parameter
    if out is not None:
        raise ValueError(f"Only None value is currently supported for out (got {out})!")

    # Compute k-th value
    k = inter(q * (input.shape[dim] - 1)) + 1
    out = torch.kthvalue(input, k, dim, keepdim=True, out=out)[0]

    # Handle keepdim and dim=None cases
    if keepdim:
        return out
    if dim_was_none:
        return out.squeeze()
    else:
        return out.squeeze(dim)

    return out


########################################################################################
########################################################################################

# Dirty code for tracking loss:

########################################################################################
########################################################################################

'''
def _compute_losses(self, coord_preds, vis_scores, conf_scores, batch):
    """Compute tracking losses using sequence_loss"""
    gt_tracks = batch["tracks"]  # B, S, N, 2
    gt_track_vis_mask = batch["track_vis_mask"]  # B, S, N

    # if self.training and hasattr(self, "train_query_points"):
    train_query_points = coord_preds[-1].shape[2]
    gt_tracks = gt_tracks[:, :, :train_query_points]
    gt_tracks = check_and_fix_inf_nan(gt_tracks, "gt_tracks", hard_max=None)

    gt_track_vis_mask = gt_track_vis_mask[:, :, :train_query_points]

    # Create validity mask that filters out tracks not visible in first frame
    valids = torch.ones_like(gt_track_vis_mask)
    mask = gt_track_vis_mask[:, 0, :] == True
    valids = valids * mask.unsqueeze(1)



    if not valids.any():
        print("No valid tracks found in first frame")
        print("seq_name: ", batch["seq_name"])
        print("ids: ", batch["ids"])
        print("time: ", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

        dummy_coord = coord_preds[0].mean() * 0          # keeps graph & grads
        dummy_vis = vis_scores.mean() * 0
        if conf_scores is not None:
            dummy_conf = conf_scores.mean() * 0
        else:
            dummy_conf = 0
        return dummy_coord, dummy_vis, dummy_conf                # three scalar zeros


    # Compute tracking loss using sequence_loss
    track_loss = sequence_loss(
        flow_preds=coord_preds,
        flow_gt=gt_tracks,
        vis=gt_track_vis_mask,
        valids=valids,
        **self.loss_kwargs
    )

    vis_loss = F.binary_cross_entropy_with_logits(vis_scores[valids], gt_track_vis_mask[valids].float())

    vis_loss = check_and_fix_inf_nan(vis_loss, "vis_loss", hard_max=None)


    # within 3 pixels
    if conf_scores is not None:
        gt_conf_mask = (gt_tracks - coord_preds[-1]).norm(dim=-1) < 3
        conf_loss = F.binary_cross_entropy_with_logits(conf_scores[valids], gt_conf_mask[valids].float())
        conf_loss = check_and_fix_inf_nan(conf_loss, "conf_loss", hard_max=None)
    else:
        conf_loss = 0

    return track_loss, vis_loss, conf_loss



def reduce_masked_mean(x, mask, dim=None, keepdim=False):
    for a, b in zip(x.size(), mask.size()):
        assert a == b
    prod = x * mask

    if dim is None:
        numer = torch.sum(prod)
        denom = torch.sum(mask)
    else:
        numer = torch.sum(prod, dim=dim, keepdim=keepdim)
        denom = torch.sum(mask, dim=dim, keepdim=keepdim)

    mean = numer / denom.clamp(min=1)
    mean = torch.where(denom > 0,
                       mean,
                       torch.zeros_like(mean))
    return mean


def sequence_loss(flow_preds, flow_gt, vis, valids, gamma=0.8, vis_aware=False, huber=False, delta=10, vis_aware_w=0.1, **kwargs):
    """Loss function defined over sequence of flow predictions"""
    B, S, N, D = flow_gt.shape
    assert D == 2
    B, S1, N = vis.shape
    B, S2, N = valids.shape
    assert S == S1
    assert S == S2
    n_predictions = len(flow_preds)
    flow_loss = 0.0

    for i in range(n_predictions):
        i_weight = gamma ** (n_predictions - i - 1)
        flow_pred = flow_preds[i]

        i_loss = (flow_pred - flow_gt).abs()  # B, S, N, 2
        i_loss = check_and_fix_inf_nan(i_loss, f"i_loss_iter_{i}", hard_max=None)

        i_loss = torch.mean(i_loss, dim=3) # B, S, N

        # Combine valids and vis for per-frame valid masking.
        combined_mask = torch.logical_and(valids, vis)

        num_valid_points = combined_mask.sum()

        if vis_aware:
            combined_mask = combined_mask.float() * (1.0 + vis_aware_w)  # Add, don't add to the mask itself.
            flow_loss += i_weight * reduce_masked_mean(i_loss, combined_mask)
        else:
            if num_valid_points > 2:
                i_loss = i_loss[combined_mask]
                flow_loss += i_weight * i_loss.mean()
            else:
                i_loss = check_and_fix_inf_nan(i_loss, f"i_loss_iter_safe_check_{i}", hard_max=None)
                flow_loss += 0 * i_loss.mean()

    # Avoid division by zero if n_predictions is 0 (though it shouldn't be).
    if n_predictions > 0:
        flow_loss = flow_loss / n_predictions

    return flow_loss
'''


def compute_projection_loss(
    predictions: Dict,
    batch: Dict,
    weight: float = 1.0,
    depth_loss_weight: float = 0.1,
    track_confidence_threshold: float = 0.3,  # 降低置信度阈值
    max_track_points: int = 128,
    **kwargs
) -> Dict:
    """
    Compute projection loss for wrist track implementation.
    
    Args:
        predictions: Model predictions containing world points and wrist pose
        batch: Batch data containing track_pairs and other data
        weight: Weight for the projection loss
        depth_loss_weight: Weight for depth loss when Z is negative
        track_confidence_threshold: Confidence threshold for track points
        max_track_points: Maximum number of track points to use
        
    Returns:
        Dictionary containing projection loss components
    """
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    
    # Check if we have the required data
    if "wrist_pose_enc" not in predictions or "world_points" not in predictions:
        return {"loss_projection": torch.tensor(0.0, device=predictions["wrist_pose_enc"].device)}
    
    if "track_pairs" not in batch:
        return {"loss_projection": torch.tensor(0.0, device=predictions["wrist_pose_enc"].device)}
    
    track_pairs = batch["track_pairs"]
    if len(track_pairs.get("wrist_uv", [])) == 0:
        return {"loss_projection": torch.tensor(0.0, device=predictions["wrist_pose_enc"].device)}
    
    # Get wrist pose and convert to extrinsic/intrinsic
    wrist_pose_enc = predictions["wrist_pose_enc"]  # [B, S, 9] where S=2
    # Take the first sequence (or you can choose which one to use)
    wrist_ext, _ = pose_encoding_to_extri_intri(
        wrist_pose_enc,  # Already has sequence dimension
        image_size_hw=(batch["images"].shape[-2], batch["images"].shape[-1]),
        build_intrinsics=False  # Don't build intrinsics from predictions
    )
    
    # Use predicted wrist extrinsics (what the model should learn)
    # wrist_ext is already in [B, 1, 3, 4] format from pose_encoding_to_extri_intri
    
    # Use GT wrist intrinsics (camera parameters are fixed)
    wrist_intrinsics = batch["wrist_intrinsics"]  # 先获取原始数据
    # 确保维度正确：如果已经是[B, 1, 3, 3]就不需要unsqueeze，如果是[B, 3, 3]就unsqueeze
    if wrist_intrinsics.dim() == 3 and wrist_intrinsics.shape[-2:] == (3, 3):
        wrist_intrinsics = wrist_intrinsics.unsqueeze(1)  # [B, 1, 3, 3]
    elif wrist_intrinsics.dim() == 4 and wrist_intrinsics.shape[-2:] == (3, 3):
        # 已经是[B, 1, 3, 3]格式，不需要修改
        pass
    else:
        raise ValueError(f"Unexpected wrist_intrinsics shape: {wrist_intrinsics.shape}, expected [B, 3, 3] or [B, 1, 3, 3]")
    
    # Get world points from predictions
    world_points = predictions["world_points"]  # [B, S, H, W, 3]
    
    # Convert track pairs to tensors for batch processing
    wrist_uv_tensor = torch.tensor(track_pairs["wrist_uv"], device=wrist_pose_enc.device, dtype=torch.float32)
    
    # 🎯 新的数据结构：使用统一的ext_uv字段
    if "ext_uv" in track_pairs:
        # 单/多视角统一数据结构，必须提供pair_type表示对应的ext索引
        ext_uv_tensor = torch.tensor(track_pairs["ext_uv"], device=wrist_pose_enc.device, dtype=torch.float32)
        pair_type_tensor = torch.tensor(track_pairs.get("pair_type", []), device=wrist_pose_enc.device, dtype=torch.int64)
    else:
        # 旧的兼容性数据结构
        ext1_uv_tensor = torch.tensor(track_pairs["ext1_uv"], device=wrist_pose_enc.device, dtype=torch.float32)
        ext2_uv_tensor = torch.tensor(track_pairs["ext2_uv"], device=wrist_pose_enc.device, dtype=torch.float32)
        pair_type_tensor = torch.tensor(track_pairs["pair_type"], device=wrist_pose_enc.device, dtype=torch.int64)
    
    confidence_tensor = torch.tensor(track_pairs["confidence"], device=wrist_pose_enc.device, dtype=torch.float32)
    # Check if track_pairs has batch indices
    batch_indices = torch.tensor(track_pairs["batch_indices"], device=wrist_pose_enc.device, dtype=torch.int64)
    
    # Filter by confidence threshold
    valid_mask = confidence_tensor > track_confidence_threshold
    if not valid_mask.any():
        print("No valid tracks found in first frame")
        return {"loss_projection": torch.tensor(0.0, device=wrist_pose_enc.device)}
    
    wrist_uv_valid = wrist_uv_tensor[valid_mask]
    if not "ext_uv" in track_pairs:
        ext_uv_valid = ext1_uv_tensor[valid_mask]
        pair_type_valid = pair_type_tensor[valid_mask]
    else:
        ext_uv_valid = ext_uv_tensor[valid_mask]
        if pair_type_tensor.numel() == 0:
            # 如果缺少pair_type，则默认全部对应第0个ext
            pair_type_valid = torch.zeros(ext_uv_valid.shape[0], dtype=torch.int64, device=ext_uv_valid.device)
        else:
            pair_type_valid = pair_type_tensor[valid_mask]
    
    batch_indices_valid = batch_indices[valid_mask]
    
    
    # Batch process projection for both wrist+ext1 and wrist+ext2 pairs
    batch_losses = []
    valid_count = 0
    depth_loss_count = 0
    uv_loss_sum = 0.0
    depth_loss_sum = 0.0
    
    for i in range(len(wrist_uv_valid)):
        wrist_uv = wrist_uv_valid[i]
        batch_idx = batch_indices_valid[i].item()
        if not "ext_uv" in track_pairs:
            pair_type = pair_type_valid[i].item()
            if pair_type in [0, 1]:
                ext_uv = ext_uv_valid[i]
                world_points_seq = pair_type
            else:
                continue
        else:
            ext_uv = ext_uv_valid[i]
            # 使用pair_type来索引预测的world_points对应的ext序列
            world_points_seq = int(pair_type_valid[i].item())
        
        # Skip if ext_uv is invalid (contains -1)
        if ext_uv[0] < 0 or ext_uv[1] < 0:
            continue
        
        # Get 3D point using bilinear interpolation (no clamping, no rounding)
        # Use the corresponding batch's world points
        point_3d = _get_interpolated_3d_point(
            world_points[batch_idx, world_points_seq],  # [H, W, 3]
            ext_uv[0],  # u coordinate (float)
            ext_uv[1]   # v coordinate (float)
        )  # [3]
        
        # Project 3D point to wrist view (using GT wrist intrinsics)
        # Use the corresponding batch's wrist camera parameters
        projected_uv, depth, valid_mask = _project_3d_to_wrist(
            point_3d, 
            wrist_ext[batch_idx:batch_idx+1],  # [1, 1, 3, 4]
            wrist_intrinsics[batch_idx:batch_idx+1]  # [1, 1, 3, 3]
        )
        
        # Process the single batch result
        if valid_mask[0]: 
            # Case 2: Valid projection, compute MSE loss
            # 获取图像尺寸用于归一化
            image_height, image_width = batch["images"].shape[-2:]  # 获取图像高度和宽度
            image_diagonal = torch.tensor(image_height**2 + image_width**2, device=projected_uv.device, dtype=projected_uv.dtype)
            
            # 计算归一化的UV loss
            uv_loss = F.mse_loss(projected_uv[0], wrist_uv)/image_diagonal * 30
            batch_losses.append(uv_loss * weight)
            uv_loss_sum += uv_loss.item() * weight
            valid_count += 1
        else:
            # Case 1: Z is negative or invalid projection, encourage Z to become positive
            depth_loss = F.relu(-depth[0]) * depth_loss_weight
            batch_losses.append(depth_loss)
            depth_loss_sum += depth_loss.item()
            depth_loss_count += 1
    
    # Compute average loss
    if batch_losses:
        total_loss = torch.stack(batch_losses).mean()
    else:
        total_loss = torch.tensor(0.0, device=wrist_pose_enc.device, requires_grad=True)
    # if valid_count > 0:
    #     uv_loss_sum /= valid_count
    #     # uv_loss_sum *= 1000
    # if depth_loss_count > 0:
    #     depth_loss_sum /= depth_loss_count
    #     # depth_loss_su 
    # total_loss = uv_loss_sum + depth_loss_sum
    return {
        "loss_projection": total_loss,
        "valid_track_points": valid_count,
        "total_track_points": len(wrist_uv_valid),
        "depth_loss_count": depth_loss_count,
        "uv_loss_sum": uv_loss_sum,
        "depth_loss_sum": depth_loss_sum
    }


def _get_interpolated_3d_point(
    world_points_map: torch.Tensor,
    u: float,
    v: float
) -> torch.Tensor:
    """
    Get 3D point using bilinear interpolation without clamping coordinates.
    
    Args:
        world_points_map: 3D world points map [H, W, 3]
        u: U coordinate (float, can be outside image bounds)
        v: V coordinate (float, can be outside image bounds)
        
    Returns:
        3D point [3] using bilinear interpolation
    """
    H, W, _ = world_points_map.shape
    
    # Convert to tensor if needed
    if not isinstance(u, torch.Tensor):
        u = torch.tensor(u, device=world_points_map.device, dtype=world_points_map.dtype)
    if not isinstance(v, torch.Tensor):
        v = torch.tensor(v, device=world_points_map.device, dtype=world_points_map.dtype)
    
    # Get the four corner indices (no clamping)
    u0, u1 = torch.floor(u), torch.ceil(u)
    v0, v1 = torch.floor(v), torch.ceil(v)
    
    # Convert to long for indexing
    u0_idx, u1_idx = u0.long(), u1.long()
    v0_idx, v1_idx = v0.long(), v1.long()
    
    # Calculate interpolation weights
    wu = u - u0
    wv = v - v0
    
    # Handle edge cases where u0 == u1 or v0 == v1
    wu = torch.where(u1_idx == u0_idx, torch.tensor(0.0, device=u.device, dtype=u.dtype), wu)
    wv = torch.where(v1_idx == v0_idx, torch.tensor(0.0, device=v.device, dtype=v.dtype), wv)
    
    # Get the four corner points with proper boundary handling
    # Use torch.clamp to handle out-of-bounds indices gracefully
    p00 = world_points_map[torch.clamp(v0_idx, 0, H-1), torch.clamp(u0_idx, 0, W-1), :]
    p01 = world_points_map[torch.clamp(v0_idx, 0, H-1), torch.clamp(u1_idx, 0, W-1), :]
    p10 = world_points_map[torch.clamp(v1_idx, 0, H-1), torch.clamp(u0_idx, 0, W-1), :]
    p11 = world_points_map[torch.clamp(v1_idx, 0, H-1), torch.clamp(u1_idx, 0, W-1), :]
    
    # Bilinear interpolation
    point_3d = (1-wu)*(1-wv)*p00 + wu*(1-wv)*p01 + (1-wu)*wv*p10 + wu*wv*p11
    
    return point_3d


def _project_3d_to_wrist(
    point_3d: torch.Tensor,
    wrist_ext: torch.Tensor,
    wrist_intrinsics: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Project 3D point to wrist camera view with batch processing.
    
    Args:
        point_3d: 3D point in world coordinates [3]
        wrist_ext: Wrist camera extrinsic matrix [B, 1, 3, 4] - world2camera变换矩阵
        wrist_intrinsics: Wrist camera intrinsic matrix [B, 1, 3, 3]
        
    Returns:
        Tuple of (projected_uv, depth, valid_mask)
        - projected_uv: [B, 2] - projected UV coordinates
        - depth: [B] - depth values for each batch
        - valid_mask: [B] - boolean mask indicating valid projections
    """
    # Assert input format is correct
    assert point_3d.shape == (3,), f"point_3d must be [3], got {point_3d.shape}"
    assert wrist_ext.dim() == 4 and wrist_ext.shape[1:] == (1, 3, 4), f"wrist_ext must be [B, 1, 3, 4], got {wrist_ext.shape}"
    assert wrist_intrinsics.dim() == 4 and wrist_intrinsics.shape[1:] == (1, 3, 3), f"wrist_intrinsics must be [B, 1, 3, 3], got {wrist_intrinsics.shape}"
    assert wrist_ext.shape[0] == wrist_intrinsics.shape[0], f"wrist_ext and wrist_intrinsics must have same batch size, got {wrist_ext.shape[0]} vs {wrist_intrinsics.shape[0]}"
    
    B = wrist_ext.shape[0]
    
    world2camera_ext = wrist_ext[:,0,:,:]
    
    # Convert to homogeneous coordinates and expand to batch size
    # Use the same dtype as wrist_ext to avoid type mismatch (wrist_ext is from predictions, so it's safe to match its type)
    target_dtype = wrist_ext.dtype
    point_3d_matched = point_3d.to(target_dtype)  # Convert point_3d to match wrist_ext dtype
    point_homo = torch.cat([point_3d_matched, torch.ones(1, device=point_3d.device, dtype=target_dtype)])  # [4]
    point_homo = point_homo.unsqueeze(0).expand(B, 4)  # [B, 4]
    
    # Project to wrist camera coordinates
    # world2camera_ext: [B, 3, 4]
    point_cam = torch.bmm(world2camera_ext, point_homo.unsqueeze(-1)).squeeze(-1)  # [B, 3]
    
    # Get depth (Z coordinate) for all batches
    depth = point_cam[:, 2]  # [B]
    
    # Create valid mask for depth > 0
    depth_valid_mask = depth > 0  # [B]
    
    # Initialize projected_uv with zeros (matching dtype)
    projected_uv = torch.zeros(B, 2, device=point_3d.device, dtype=target_dtype)
    
    # Only process valid batches (depth > 0)
    if depth_valid_mask.any():
        # Project to image coordinates for valid batches
        point_2d = point_cam[:, :2] / depth.unsqueeze(-1)  # [B, 2]
        
        # Apply intrinsics
        # wrist_intrinsics: [B, 1, 3, 3] -> [B, 3, 3]
        wrist_intrinsics_squeezed = wrist_intrinsics.squeeze(1)  # [B, 3, 3]
        # Convert wrist_intrinsics to match point_2d dtype to avoid type mismatch
        wrist_intrinsics_squeezed = wrist_intrinsics_squeezed.to(point_2d.dtype)
        point_2d_homo = torch.cat([point_2d, torch.ones(B, 1, device=point_2d.device, dtype=point_2d.dtype)], dim=1)  # [B, 3]
        point_pixel = torch.bmm(wrist_intrinsics_squeezed, point_2d_homo.unsqueeze(-1)).squeeze(-1)  # [B, 3]
        point_pixel = point_pixel[:, :2]  # [B, 2]
        
        # Combine masks
        valid_mask = depth_valid_mask  # [B]
        
        # Assign valid projections - ensure both tensors have the same dtype
        projected_uv = projected_uv.to(point_pixel.dtype)
        projected_uv[valid_mask] = point_pixel[valid_mask]
    else:
        # No valid depth, all invalid
        valid_mask = torch.zeros(B, dtype=torch.bool, device=point_3d.device)
    
    return projected_uv, depth, valid_mask



