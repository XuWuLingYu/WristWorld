#!/usr/bin/env python3
"""
VGGT 训练验证可视化工具

提供训练过程中的验证可视化功能：
- 深度图可视化（ext1和ext2分别）
- 点云生成和保存（含wrist origin红球）
- wrist视角点云投影
"""

import os
from re import S
import sys
import numpy as np
import cv2
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F
import trimesh
import matplotlib.pyplot as plt
from pathlib import Path
import json
from typing import Tuple, List, Optional, Dict, Any
import logging
from datetime import datetime
from vggt.utils.pose_enc import extri_intri_to_pose_encoding,pose_encoding_to_extri_intri



class ValidationVisualizer:
    """
    验证阶段的可视化工具
    """
    
    def __init__(self, output_base_dir: str = "logs", rank: int = 0, experiment_name: str = None):
        """
        初始化可视化工具
        
        Args:
            output_base_dir: 输出基础目录
            rank: 分布式训练的rank
        """
        self.output_base_dir = Path(output_base_dir)
        self.rank = rank
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 只在rank 0上进行可视化，避免多进程冲突
        self.should_visualize = (rank == 0)
        
        # 🔧 修复：始终初始化目录路径，避免属性缺失错误
        # 🆕 生成时间戳目录，每次训练会话使用独立目录
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if experiment_name is None:
            self.val_vis_base = self.output_base_dir / "validation_visualizations"
        else:
            self.val_vis_base = self.output_base_dir / "validation_visualizations" / experiment_name
        self.val_vis_dir = self.val_vis_base / timestamp
        # self.depth_dir = self.val_vis_dir / "depth_maps"
        # self.pointcloud_dir = self.val_vis_dir / "pointclouds"
        # self.gt_pointcloud_dir = self.val_vis_dir / "gt_pointclouds"  # 🆕 GT点云目录
        self.projection_dir = self.val_vis_dir
        if self.should_visualize:
            # 创建所有必要目录
            for dir_path in [self.val_vis_dir]:
                dir_path.mkdir(parents=True, exist_ok=True)
                
            logging.info(f"✅ 验证可视化目录已创建: {self.val_vis_dir}")
            logging.info(f"📅 时间戳会话目录: {timestamp}")
        
        logging.info(f"🎨 ValidationVisualizer 初始化完成 (rank={rank}, visualize={self.should_visualize})")
    
    def visualize_validation_results(self, predictions: Dict, batch: Dict, epoch: int, batch_idx: int) -> Dict[str, str]:
        """
        完整的验证结果可视化
        
        Args:
            predictions: 模型预测结果
            batch: 输入batch数据
            epoch: 当前epoch
            batch_idx: 当前batch索引
            
        Returns:
            生成的文件路径字典
        """
        import cv2
        import json
        from pathlib import Path
        import gc
        
        # === 内存管理：开始前清理 ===
        gc.collect()
        if hasattr(torch, 'cuda') and torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # === 由于batch_size是24，只保留第0个sample进行可视化 ===
        # 对所有batch数据进行切片，只保留第0个样本
        for key in batch:
            if isinstance(batch[key], np.ndarray) and batch[key].ndim > 0:
                batch[key] = batch[key][:1]  # 只保留第0个样本
            if isinstance(batch[key], torch.Tensor) and batch[key].ndim > 0:
                batch[key] = batch[key][:1]  # 只保留第0个样本
            elif isinstance(batch[key], list) and len(batch[key]) > 0:
                batch[key] = batch[key][:1]  # 只保留第0个样本
        
        # === 严格的断言检查 - 确保唯一形状 ===
        # 1. 检查batch基本结构
        assert "images" in batch, "batch中缺少images"
        # assert "depths" in batch, "batch中缺少depths"
        # assert "depth_nan_masks" in batch, "batch中缺少depth_nan_masks"
        assert "extrinsics" in batch, "batch中缺少extrinsics"
        assert "intrinsics" in batch, "batch中缺少intrinsics"
        # assert "point_cloud" in batch, "batch中缺少point_cloud"
        # assert "point_colors" in batch, "batch中缺少point_colors"
        
        # 2. 检查图像数据形状 - 唯一形状
        images = batch["images"]
        # depths = batch["depths"]
        # depth_nan_masks = batch["depth_nan_masks"]
        
        # 检查batch维度（现在应该是单样本）
        assert images.ndim == 5, f"图像维度错误: {images.ndim}，期望5 (B,S,C,H,W)"
        # assert depths.ndim == 4, f"深度图维度错误: {depths.ndim}，期望4 (B,S,H,W)"
        # assert depth_nan_masks.ndim == 4, f"深度mask维度错误: {depth_nan_masks.ndim}，期望4 (B,S,H,W)"
        
        # 检查batch size（现在应该是1）
        assert images.shape[0] == 1, f"期望batch size为1，实际{images.shape[0]}"
        # assert depths.shape[0] == 1, f"期望batch size为1，实际{depths.shape[0]}"
        # assert depth_nan_masks.shape[0] == 1, f"期望batch size为1，实际{depth_nan_masks.shape[0]}"
        
        # 检查序列长度（相机数量）
        # assert images.shape[1] == 2, f"期望2个相机，实际{images.shape[1]}个"
        # assert depths.shape[1] == 2, f"期望2个深度图，实际{depths.shape[1]}个"
        # assert depth_nan_masks.shape[1] == 2, f"期望2个深度mask，实际{depth_nan_masks.shape[1]}个"
        
        # 检查图像尺寸 - 注意图像是(B,S,C,H,W)格式
        assert images.shape[2] == 3, f"图像形状错误: {images.shape}，期望3channel"
        # assert depths.shape[2:] == (294, 518), f"深度图形状错误: {depths.shape[2:]}，期望(294,518)"
        # assert depth_nan_masks.shape[2:] == (294, 518), f"深度mask形状错误: {depth_nan_masks.shape[2:]}，期望(294,518)"
        
        # 3. 检查相机参数形状 - 唯一形状（支持任意视角数S）
        extrinsics = batch["extrinsics"]  # (1, S, 3, 4)
        intrinsics = batch["intrinsics"]  # (1, S, 3, 3)
        wrist_extrinsics = batch["wrist_extrinsics"]  # (1, 3, 4)
        wrist_intrinsics = batch["wrist_intrinsics"]  # (1, 3, 3)
        
        # 转换为numpy数组（如果是tensor）
        if torch.is_tensor(extrinsics):
            extrinsics = extrinsics.cpu().numpy()
        if torch.is_tensor(intrinsics):
            intrinsics = intrinsics.cpu().numpy()
        if torch.is_tensor(wrist_extrinsics):
            wrist_extrinsics = wrist_extrinsics.cpu().numpy()
        if torch.is_tensor(wrist_intrinsics):
            wrist_intrinsics = wrist_intrinsics.cpu().numpy()
        
        wrist_extrinsics = wrist_extrinsics[0]
        wrist_intrinsics = wrist_intrinsics[0]
        assert extrinsics.ndim == 4 and extrinsics.shape[0] == 1 and extrinsics.shape[2:] == (3, 4), f"wrist外参形状错误: {extrinsics.shape}，期望(1,S,3,4)"
        assert intrinsics.ndim == 4 and intrinsics.shape[0] == 1 and intrinsics.shape[2:] == (3, 3), f"wrist内参形状错误: {intrinsics.shape}，期望(1,S,3,3)"
        assert wrist_extrinsics.shape == (1, 3, 4), f"wrist外参形状错误: {wrist_extrinsics.shape}，期望(1,3,4)"
        assert wrist_intrinsics.shape == (1, 3, 3), f"wrist内参形状错误: {wrist_intrinsics.shape}，期望(1,3,3)"
        
        # 4. 检查点云数据 - 唯一形状
        # 使用预测的点云而不是GT点云
        if "world_points" in predictions:
            points_3d = predictions["world_points"][0]
            if torch.is_tensor(points_3d):
                points_3d = points_3d.cpu().numpy()
        else:
            raise ValueError("predictions 缺少 world_points")
        points_3d = points_3d.reshape(-1, 3)  # 重塑为(N, 3)
        
        # 获取单个样本的图像数据
        images_sample = images[0]
        
        # 统一生成颜色：拼接所有视角RGB
        colors_list = []
        for cam_idx in range(images_sample.shape[0]):
            rgb = images_sample[cam_idx]
            if torch.is_tensor(rgb):
                rgb = rgb.cpu().numpy()
            rgb = np.transpose(rgb, (1, 2, 0))
            assert rgb.shape[-1] == 3, f"RGB形状错误: {rgb.shape}"
            colors_list.append(rgb.reshape(-1, 3))
        colors = np.concatenate(colors_list, axis=0)
        
        # 确保点云和颜色数量匹配
        assert len(points_3d) == len(colors), f"点云和颜色数量不匹配: {len(points_3d)} vs {len(colors)}"
        assert points_3d.ndim == 2, f"点云维度错误: {points_3d.ndim}，期望2"
        assert points_3d.shape[1] == 3, f"点云形状错误: {points_3d.shape}，期望(N,3)"
        assert colors.ndim == 2, f"颜色维度错误: {colors.ndim}，期望2"
        assert colors.shape[1] == 3, f"颜色形状错误: {colors.shape}，期望(N,3)"
        
        # 5. 检查predictions基本结构
        assert "pose_enc" in predictions, "predictions中缺少pose_enc"
        assert "wrist_pose_enc" in predictions, "predictions中缺少wrist_pose_enc"
        
        
        # === 创建输出目录 ===
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_dir = Path(self.projection_dir) / f"epoch_{epoch}_batch_{batch_idx}_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # === 可视化处理 ===
        # 使用第0个样本（现在batch中只有1个样本）
        sample_idx = 0
        
        # 提取第0个样本的数据
        images_sample = images[sample_idx]
        # depths_sample = depths[sample_idx]
        # depth_nan_masks_sample = depth_nan_masks[sample_idx]
        extrinsics_sample = extrinsics[sample_idx]
        intrinsics_sample = intrinsics[sample_idx]
        
        # === 1. 可视化相机视角（任意S） ===
        S = images_sample.shape[0]
        camera_names = [f"ext{i+1}" for i in range(S)]
        camera_indices = list(range(S))
        
        for i, (camera_name, camera_idx) in enumerate(zip(camera_names, camera_indices)):
            # 获取当前相机的数据
            camera_rgb = images_sample[camera_idx]  # (3, 294, 518)
            # camera_depth = depths_sample[camera_idx]  # (294, 518) - GT depth
            # camera_valid_depth = depth_nan_masks_sample[camera_idx]  # (294, 518)
            camera_extrinsic = extrinsics_sample[camera_idx]  # (3, 4)
            camera_intrinsic = intrinsics_sample[camera_idx]  # (3, 3)
            
            # 获取预测的depth
            assert "depth" in predictions, "predictions中缺少depth"
            pred_depth = predictions["depth"][0, camera_idx].cpu().numpy()  # 取第0个样本，第camera_idx个相机
            pred_depth = pred_depth.squeeze()  # 移除最后的维度，从(294, 518, 1)变为(294, 518)
            
            # 转换为numpy并处理维度
            if torch.is_tensor(camera_rgb):
                camera_rgb = camera_rgb.cpu().numpy()
            # if torch.is_tensor(camera_depth):
            #     camera_depth = camera_depth.cpu().numpy()
            # if torch.is_tensor(camera_valid_depth):
            #     camera_valid_depth = camera_valid_depth.cpu().numpy()
            if torch.is_tensor(camera_extrinsic):
                camera_extrinsic = camera_extrinsic.cpu().numpy()
            if torch.is_tensor(camera_intrinsic):
                camera_intrinsic = camera_intrinsic.cpu().numpy()
             
            # 转换RGB格式 (C, H, W) -> (H, W, C)
            camera_rgb = np.transpose(camera_rgb, (1, 2, 0))  # (294, 518, 3)
            camera_rgb = (camera_rgb * 255).astype(np.uint8)
            all_camera_rgb = images_sample.cpu().numpy()
            all_camera_rgb = np.transpose(all_camera_rgb, (0,2,3,1))
            all_camera_rgb = (all_camera_rgb*255).astype(np.uint8)
            # 创建2x2网格
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle(f'{camera_name.upper()} Visualization', fontsize=16)
            
            # 1. RGB图像
            axes[0, 0].imshow(camera_rgb)
            axes[0, 0].set_title('RGB Image')
            axes[0, 0].axis('off')
            
            # 2. Prediction Depth - 使用预测的depth
            depth_img = axes[0, 1].imshow(pred_depth, cmap='viridis')
            axes[0, 1].set_title('Prediction Depth')
            axes[0, 1].axis('off')
            plt.colorbar(depth_img, ax=axes[0, 1], fraction=0.046, pad=0.04)
            
            # 3. GT Depth (使用valid mask)
            # gt_depth_masked = camera_depth.copy()
            # gt_depth_masked[~camera_valid_depth] = np.nan
            # gt_depth_img = axes[1, 0].imshow(gt_depth_masked, cmap='viridis')
            # axes[1, 0].set_title('GT Depth (Valid)')
            # axes[1, 0].axis('off')
            # plt.colorbar(gt_depth_img, ax=axes[1, 0], fraction=0.046, pad=0.04)
            
            # 4. Point Cloud Projection - 使用预测点云和预测相机参数
            # 获取预测的点云（如果存在）
            assert "world_points" in predictions
            pred_points_3d = predictions["world_points"][0].cpu().numpy()  # 取第0个样本
            
            # 统一处理world_points维度
            if pred_points_3d.ndim == 3:
                pred_points_3d = pred_points_3d[None, ...]
            assert pred_points_3d.ndim == 4, f"Unexpected world_points shape: {pred_points_3d.shape}"
            pred_points_3d = pred_points_3d[camera_idx].reshape(-1, 3)
            pred_colors = all_camera_rgb[camera_idx].reshape(-1, 3)
            
            # 确保点云和颜色数量匹配
            assert len(pred_points_3d) == len(pred_colors), f"点云和颜色数量不匹配: {len(pred_points_3d)} vs {len(pred_colors)}"
            
            # 获取预测的相机参数（如果存在）
            assert "pose_enc_list" in predictions
            # pose_enc形状是[24, 2, 9]，其中2表示两个相机(ext1, ext2)
            # 我们需要根据当前相机索引camera_idx来获取对应的pose
            pred_pose_enc = predictions["pose_enc_list"][ -1][0, camera_idx].cpu()  # 取第0个样本，第camera_idx个相机
            extrinsics,intrinsics = pose_encoding_to_extri_intri(pred_pose_enc.unsqueeze(0).unsqueeze(0),image_size_hw=camera_rgb.shape[:2])
            pred_extrinsic = extrinsics[0,0].numpy()
            pred_intrinsic = intrinsics[0,0].numpy()
            
            
            # 投影点云
 
            projection = self.visualize_point_cloud_projection(
                points_3d=pred_points_3d,
                point_colors=pred_colors,
                camera_intrinsics=pred_intrinsic,
                camera_extrinsics=pred_extrinsic,
                image_shape=camera_rgb.shape[:2],
                need_inverse=False # 对于ext1/ext2，是world2camera，不需要求逆
            )
            axes[1, 1].imshow(projection)
            axes[1, 1].set_title('Point Cloud Projection')
            axes[1, 1].axis('off')
            
            # 保存2x2网格
            output_path = output_dir / f"{camera_name}_grid.png"
            plt.tight_layout()
            plt.savefig(str(output_path), dpi=150, bbox_inches='tight')
            plt.close()
        
        # 保存单独的projection
        
        # 保存单独的RGB和depth
        
        # === 2. 可视化GT wrist RGB ===
        if "wrist_image" in batch and batch["wrist_image"] is not None:
            wrist_rgb = batch["wrist_image"]
            
            # 处理batch维度
            if wrist_rgb.ndim == 4:  # (B, H, W, 3) 格式
                wrist_rgb = wrist_rgb[sample_idx]  # 取第0个样本
            elif wrist_rgb.ndim == 3:  # (H, W, 3) 格式，已经是单个样本
                pass
        else:
            raise ValueError(f"wrist_rgb维度错误: {wrist_rgb.ndim}，期望3或4")
        
        wrist_rgb = np.array(wrist_rgb.cpu())
        assert wrist_rgb.ndim == 3, f"wrist_rgb维度错误: {wrist_rgb.ndim}，期望3"
        assert wrist_rgb.shape[2] == 3, f"wrist_rgb形状错误: {wrist_rgb.shape}，期望(H,W,3)"
        
        # 确保颜色范围正确
        wrist_rgb = (wrist_rgb).astype(np.uint8)
        
        # 保存真实wrist RGB图像
        wrist_rgb_path = output_dir / "wrist_rgb.png"
        plt.imsave(str(wrist_rgb_path), wrist_rgb)  # 使用plt.imsave保持RGB格式
        
        # === 3.1. 可视化wrist投影 ===
        # 获取预测的wrist pose和点云
        if "wrist_pose_enc_list" in predictions and "world_points" in predictions:
            # 获取预测的wrist pose
            wrist_pose_enc = predictions["wrist_pose_enc_list"][-1][0].cpu()  # 取第0个样本
            # 🔥 NEW: wrist_head now outputs single wrist pose [B, 1, target_dim] instead of [B, S, target_dim]
            wrist_pose_enc = wrist_pose_enc[0]  # 取第一个（也是唯一的）wrist pose
            wrist_ext,wrist_intrinsics = pose_encoding_to_extri_intri(wrist_pose_enc.unsqueeze(0).unsqueeze(0),image_size_hw=wrist_rgb.shape[:2]) # camera2world
            wrist_ext = wrist_ext[0,0].numpy()
            # 使用GT wrist intrinsics而不是预测的intrinsics
            wrist_intrinsics_gt = batch["wrist_intrinsics"][0].cpu().numpy()  # GT intrinsic
            # 获取预测的点云
            pred_points_3d = predictions["world_points"][0].cpu().numpy()
            pred_points_3d = pred_points_3d.reshape(-1, 3)  
            
            # 确保点云和颜色数量匹配
            if len(pred_points_3d) != len(colors):
                logging.warning(f"Wrist projection: 点云和颜色数量不匹配: {len(pred_points_3d)} vs {len(colors)}")
                # 如果数量不匹配，截取到较小的数量
                min_count = min(len(pred_points_3d), len(colors))
                pred_points_3d = pred_points_3d[:min_count]
                colors = colors[:min_count]
            
            # 投影到wrist视角
            wrist_projection = self.visualize_point_cloud_projection(
                points_3d=pred_points_3d,
                point_colors=colors,  # 使用原始点云颜色
                camera_extrinsics= wrist_ext,
                camera_intrinsics= wrist_intrinsics_gt[0],  # 使用GT intrinsic
                image_shape=wrist_rgb.shape[:2],
                need_inverse=False # 对于wrist，是world2camera，不需要求逆
            )
            
            # 保存wrist投影
            wrist_projection_path = output_dir / "wrist_projection.png"
            cv2.imwrite(str(wrist_projection_path), cv2.cvtColor(wrist_projection, cv2.COLOR_RGB2BGR))
            # 创建wrist对比图（原图vs投影）
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            fig.suptitle('Wrist Camera: Original vs Projection', fontsize=16)
            
            axes[0].imshow(wrist_rgb)
            axes[0].set_title('Original Wrist RGB')
            axes[0].axis('off')
            
            axes[1].imshow(wrist_projection)
            axes[1].set_title('Point Cloud Projection')
            axes[1].axis('off')
            
            wrist_comparison_path = output_dir / "wrist_comparison.png"
            plt.tight_layout()
            plt.savefig(str(wrist_comparison_path), dpi=150, bbox_inches='tight')
            plt.close()
        
        # === 4. 可视化点云（带红球和绿球） ===
        # 创建带红球的点云（预测wrist pose）
        points_with_red_sphere = self._add_wrist_origin_sphere(
            predictions=predictions,  # 传入predictions而不是batch
            points_3d=points_3d,
            colors=colors
        )
        
        # 保存带红球的点云
        red_sphere_path = output_dir / "pointcloud_with_red_sphere.glb"
        self._save_point_cloud_as_glb(
            points=points_with_red_sphere["points"],
            colors=points_with_red_sphere["colors"],
            output_path=str(red_sphere_path)
        )
        
        # === 5. 保存元数据（统一，任意视角） ===
        metadata = {
            "epoch": epoch,
            "batch_idx": batch_idx,
            "timestamp": timestamp,
            "point_cloud_size": len(points_3d),
            "training_mode": f"multi_view_{images_sample.shape[0]}",
            "cameras": [f"ext{i+1}" for i in range(images_sample.shape[0])] + ["wrist"],
            "image_shapes": {f"ext{i+1}": images_sample[i].shape for i in range(images_sample.shape[0])},
        }
        
        metadata_path = output_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # === 内存管理：结束后清理 ===
        plt.close('all')
        gc.collect()
        if hasattr(torch, 'cuda') and torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 返回生成的文件路径（统一）
        result = {
            "output_dir": str(output_dir),
            "metadata": str(metadata_path),
            "wrist_rgb": str(output_dir / "wrist_rgb.png"),
            "pointcloud_with_red_sphere": str(red_sphere_path),
        }
        for i in range(images_sample.shape[0]):
            name = f"ext{i+1}"
            result[f"{name}_grid"] = str(output_dir / f"{name}_grid.png")
            result[f"{name}_rgb"] = str(output_dir / f"{name}_rgb.png")
            result[f"{name}_depth"] = str(output_dir / f"{name}_depth.png")
            result[f"{name}_projection"] = str(output_dir / f"{name}_projection.png")
        
        # 如果有wrist投影，添加相关路径
        if "wrist_pose_enc" in predictions and "world_points" in predictions:
            result["wrist_projection"] = str(output_dir / "wrist_projection.png")
            result["wrist_comparison"] = str(output_dir / "wrist_comparison.png")
        
        # 如果有GT wrist pose，添加绿球点云路径
        # if "wrist_extrinsics" in batch and batch["wrist_extrinsics"] is not None:
            # result["gt_pointcloud_with_green_sphere"] = str(green_sphere_path)
        
        # === 6. 新增：Projection可视化 ===
        if "track_pairs" in batch and "wrist_pose_enc_list" in predictions and "world_points" in predictions:
 
            projection_vis_result = self._visualize_projection_tracking(
                predictions=predictions,
                batch=batch,
                output_dir=output_dir,
            )
            result.update(projection_vis_result)
         
        return result
    
    def _pose_to_extrinsics(self, pose_enc: np.ndarray) -> np.ndarray:
        """
        将pose encoding转换为外参矩阵
        
        Args:
            pose_enc: pose encoding (6,) 或 (9,) - 支持6D和9D格式
                - 6D格式: [tx, ty, tz, rx, ry, rz]
                - 9D格式: [tx, ty, tz, qx, qy, qz, qw, fov_h, fov_w]
            
        Returns:
            外参camera to world矩阵 (3, 4)
        """
        assert pose_enc.shape == (9,), f"pose_enc形状错误: {pose_enc.shape}，期望(9,)"
        # 9D格式: [tx, ty, tz, qx, qy, qz, qw, fov_h, fov_w]
        translation = pose_enc[:3]  # [tx, ty, tz]
        quaternion = pose_enc[3:7]  # [qx, qy, qz, qw]
        
        # 将四元数转换为旋转矩阵
        import cv2
        rotation_matrix = cv2.Rodrigues(quaternion[:3])[0]  # 使用前3个分量作为旋转向量
        
        # 构建外参矩阵 [R|t]
        extrinsics = np.eye(4)
        extrinsics[:3, :3] = rotation_matrix
        extrinsics[:3, 3] = translation
        
        return extrinsics[:3, :]  # 返回 (3, 4)
    
    def visualize_point_cloud_projection(
        self,
        points_3d: np.ndarray,
        point_colors: np.ndarray,
        camera_extrinsics: np.ndarray,
        camera_intrinsics: np.ndarray,
        image_shape: Tuple[int, int],
        need_inverse: bool = False
    ) -> np.ndarray:
        """
        将3D点云投影到指定相机视角并可视化（按距离排序，远的先画，近的后画）
        
        Args:
            points_3d: 3D点云坐标 (N, 3) - 世界坐标系
            point_colors: 点云颜色 (N, 3)
            camera_extrinsics: 相机外参 (3, 4) - world2camera变换矩阵
            camera_intrinsics: 相机内参 (3, 3) - GT intrinsic
            image_shape: 输出图像形状 (H, W)
            need_inverse: 是否需要对外参求逆 (True for wrist, False for ext1/ext2)
            
        Returns:
            投影图像 (H, W, 3)
        """
        import cv2

        H, W = image_shape

        # 检查输入数据
        assert points_3d.ndim == 2, f"点云维度错误: {points_3d.ndim}，期望2"
        assert points_3d.shape[1] == 3, f"点云形状错误: {points_3d.shape}，期望(N,3)"
        assert point_colors.ndim == 2, f"颜色维度错误: {point_colors.ndim}，期望2"
        assert point_colors.shape[1] == 3, f"颜色形状错误: {point_colors.shape}，期望(N,3)"
        assert len(points_3d) == len(point_colors), f"点云和颜色数量不匹配: {len(points_3d)} vs {len(point_colors)}"
        assert camera_extrinsics.shape == (3, 4), f"相机外参形状错误: {camera_extrinsics.shape}，期望(3,4)"
        assert camera_intrinsics.shape == (3, 3), f"相机内参形状错误: {camera_intrinsics.shape}，期望(3,3)"

        # 根据need_inverse参数决定是否求逆
        if need_inverse:
            # wrist投影：camera_extrinsics是camera2world变换矩阵，需要求逆得到world2camera
            camera_ext_4x4 = np.vstack([camera_extrinsics, [0, 0, 0, 1]])  # 扩展为4x4齐次坐标矩阵
            world2camera_ext = np.linalg.inv(camera_ext_4x4)[:3, :4]  # 求逆得到world2camera变换
        else:
            # ext1/ext2投影：camera_extrinsics已经是world2camera变换矩阵，直接使用
            world2camera_ext = camera_extrinsics

        # 创建输出图像
        image = np.full((H, W, 3), (0, 0, 0), dtype=np.uint8)

        # 向量化处理所有3D点
        # 转换为齐次坐标 [N, 4]
        points_homo = np.concatenate([points_3d, np.ones((len(points_3d), 1))], axis=1)

        # 投影到相机坐标系 [N, 3]
        points_cam = (world2camera_ext @ points_homo.T).T

        # 深度过滤mask
        depth_mask = points_cam[:, 2] > 0.01
        if not np.any(depth_mask):
            return image

        # 应用深度过滤
        points_cam = points_cam[depth_mask]
        point_colors = point_colors[depth_mask]

        # 投影到图像平面 [N, 2]
        points_2d = points_cam[:, :2] / points_cam[:, 2:3]

        # 应用内参 [N, 2]
        points_2d_homo = np.concatenate([points_2d, np.ones((len(points_2d), 1))], axis=1)
        points_pixel = (camera_intrinsics @ points_2d_homo.T).T
        projected_uv = points_pixel[:, :2]

        # 边界检查mask
        u_mask = (projected_uv[:, 0] >= 0) & (projected_uv[:, 0] < W)
        v_mask = (projected_uv[:, 1] >= 0) & (projected_uv[:, 1] < H)
        boundary_mask = u_mask & v_mask

        if not np.any(boundary_mask):
            return image

        # 应用边界过滤
        projected_uv = projected_uv[boundary_mask]
        point_colors = point_colors[boundary_mask]
        points_cam = points_cam[boundary_mask]

        # 按距离排序（z越大越远，先画远的）
        z_vals = points_cam[:, 2]
        sort_idx = np.argsort(z_vals)[::-1]  # 从远到近（z大到z小）
        projected_uv = projected_uv[sort_idx]
        point_colors = point_colors[sort_idx]

        # 转换为整数坐标
        u_coords = projected_uv[:, 0].astype(int)
        v_coords = projected_uv[:, 1].astype(int)

        # 处理颜色格式（向量化）
        if point_colors.max() < 2:
            point_colors = (point_colors * 255).astype(np.uint8)

        # 按顺序画点（远的先画，近的后画）
        for i in range(len(u_coords)):
            u, v = u_coords[i], v_coords[i]
            color = point_colors[i].tolist()
            cv2.circle(image, (u, v), 2, color, -1)

        valid_count = len(u_coords)

        return image
    
    def _add_wrist_origin_sphere(self, predictions: Dict, points_3d: np.ndarray, colors: np.ndarray) -> Dict:
        """
        在点云中添加红球表示预测的wrist origin
        
        Args:
            predictions: 包含wrist pose预测的predictions数据
            points_3d: 原始点云坐标 (N, 3)
            colors: 原始点云颜色 (N, 3)
            
        Returns:
            包含红球的点云数据 {"points": ..., "colors": ...}
        """
        # 获取预测的wrist pose
        wrist_pose_enc = predictions.get("wrist_pose_enc")
        if wrist_pose_enc is None:
            return {"points": points_3d, "colors": colors}
        
        # 处理batch维度和pose encoding格式
        if torch.is_tensor(wrist_pose_enc):
            wrist_pose_enc = wrist_pose_enc.cpu().numpy()
        
        # 🔥 NEW: wrist_head now outputs single wrist pose [B, 1, target_dim] instead of [B, S, target_dim]
        wrist_pose_enc = wrist_pose_enc[0]  # 取第0个样本
        assert wrist_pose_enc.shape == (1, 9), f"wrist_pose_enc形状错误: {wrist_pose_enc.shape}，期望(1,9)"
        # 现在只有一个wrist pose，9D格式
        wrist_pose = wrist_pose_enc[0]  # 取唯一的wrist pose，9D格式
        
        # 转换为camera-to-world外参矩阵
        wrist_ext = self._pose_to_extrinsics(wrist_pose)
        
        # 提取wrist origin位置
        # wrist_ext是camera2world变换矩阵T_wc
        wrist_origin = wrist_ext[:3, 3]  # 取逆矩阵的平移部分
        
        # 生成红球点云
        sphere_points, sphere_colors = self._generate_sphere_points(
            center=wrist_origin,
            radius=0.05,  # 5cm半径
            color=(255, 0, 0),  # 红色
            num_points=100
        )
        
        
        # 合并原始点云和红球
        combined_points = np.vstack([points_3d, sphere_points])
        combined_colors = np.vstack([colors, sphere_colors])
        
        return {"points": combined_points, "colors": combined_colors}
    
    def _add_gt_wrist_origin_sphere(self, batch: Dict, points_3d: np.ndarray, colors: np.ndarray) -> Dict:
        """
        在点云中添加绿球表示GT wrist origin
        
        Args:
            batch: 包含GT wrist pose的batch数据
            points_3d: 原始点云坐标 (N, 3)
            colors: 原始点云颜色 (N, 3)
            
        Returns:
            包含绿球的点云数据 {"points": ..., "colors": ...}
        """
        # 获取GT wrist pose
        wrist_extrinsics = batch.get("wrist_extrinsics")
        if wrist_extrinsics is None:
            return {"points": points_3d, "colors": colors}
        
        wrist_ext = wrist_extrinsics[0][0].cpu().numpy()
        
        assert wrist_ext.shape == (3, 4), f"wrist外参形状错误: {wrist_ext.shape}，期望(3,4)"
        
        # 提取GT wrist origin位置
        # wrist_ext是world2camera变换矩阵T_wc
        # 要得到相机在世界坐标系中的位置，需要求逆：T_cw = inv(T_wc)
        # 相机位置 = T_cw * [0,0,0,1] = T_cw的平移部分
        wrist_ext_4x4 = np.vstack([wrist_ext, [0, 0, 0, 1]])  # 扩展为4x4齐次坐标矩阵
        wrist_ext_inv = np.linalg.inv(wrist_ext_4x4)  # 求逆得到camera2world变换
        gt_wrist_origin = wrist_ext_inv[:3, 3]  # 取逆矩阵的平移部分
        
        # 生成绿球点云
        sphere_points, sphere_colors = self._generate_sphere_points(
            center=gt_wrist_origin,
            radius=0.05,  # 5cm半径
            color=(0, 255, 0),  # 绿色
            num_points=100
        )
        
        
        # 合并原始点云和绿球
        combined_points = np.vstack([points_3d, sphere_points])
        combined_colors = np.vstack([colors, sphere_colors])
        
        return {"points": combined_points, "colors": combined_colors}
    
    def _generate_sphere_points(self, center: np.ndarray, radius: float, color: Tuple[int, int, int], num_points: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        生成球体点云
        
        Args:
            center: 球心坐标 (3,)
            radius: 球半径
            color: 球颜色 (R, G, B)
            num_points: 球体点数
            
        Returns:
            球体点云坐标和颜色
        """
        # 生成球面均匀分布的点
        phi = np.linspace(0, 2*np.pi, int(np.sqrt(num_points)))
        theta = np.linspace(0, np.pi, int(np.sqrt(num_points)))
        phi, theta = np.meshgrid(phi, theta)
        
        # 球坐标转笛卡尔坐标
        x = radius * np.sin(theta) * np.cos(phi)
        y = radius * np.sin(theta) * np.sin(phi)
        z = radius * np.cos(theta)
        
        # 展平并添加球心偏移
        sphere_points = np.concatenate([x.flatten(), y.flatten(), z.flatten()]).reshape(-1, 3)
        sphere_points = sphere_points + center  # 使用numpy广播机制
        
        # 生成颜色
        sphere_colors = np.full((len(sphere_points), 3), color, dtype=np.uint8)
        
        return sphere_points, sphere_colors
    
    def _save_point_cloud_as_glb(self, points: np.ndarray, colors: np.ndarray, output_path: str):
        """
        将点云保存为GLB格式
        
        Args:
            points: 点云坐标 (N, 3)
            colors: 点云颜色 (N, 3)
            output_path: 输出文件路径
        """
        # 创建trimesh点云对象
        point_cloud = trimesh.PointCloud(
            vertices=points,
            colors=colors
        )
        
        # 导出为GLB格式
        point_cloud.export(output_path)
    
    def _visualize_projection_tracking(
        self,
        predictions: Dict,
        batch: Dict,
        output_dir: Path,
    ) -> Dict[str, str]:
        """
        可视化projection tracking结果
        
        Args:
            predictions: 模型预测结果
            batch: 输入batch数据
            output_dir: 输出目录
            
        Returns:
            生成的文件路径字典
        """
        import cv2
        import numpy as np
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri
        is_single_view_data = False
        if "single_view_training" in batch:
            is_single_view_data = batch["single_view_training"][0].item() if torch.is_tensor(batch["single_view_training"]) else batch["single_view_training"][0]
        
        # 只处理batch_index=0的数据
        track_pairs = batch["track_pairs"]
        if len(track_pairs.get("wrist_uv", [])) == 0:
            print("no track pairs")
            return {}
        
        # 过滤出batch_index=0的track pairs
        if "batch_indices" in track_pairs:
            batch_indices = track_pairs["batch_indices"]
            batch_0_mask = [i == 0 for i in batch_indices]
            
            # 提取batch_0的数据
            wrist_uv_batch0 = [track_pairs["wrist_uv"][i] for i, mask in enumerate(batch_0_mask) if mask]
            
            # 🎯 新的数据结构：使用统一的ext_uv字段
            if "ext_uv" in track_pairs:
                ext_uv_batch0 = [track_pairs["ext_uv"][i] for i, mask in enumerate(batch_0_mask) if mask]
            else:
                # 旧的兼容性数据结构
                ext1_uv_batch0 = [track_pairs["ext1_uv"][i] for i, mask in enumerate(batch_0_mask) if mask]
                ext2_uv_batch0 = [track_pairs["ext2_uv"][i] for i, mask in enumerate(batch_0_mask) if mask]
            pair_type_batch0 = [track_pairs["pair_type"][i] for i, mask in enumerate(batch_0_mask) if mask]
            
            confidence_batch0 = [track_pairs["confidence"][i] for i, mask in enumerate(batch_0_mask) if mask]
        # print(len(wrist_uv_batch0),len(track_pairs["wrist_uv"]))
        if len(wrist_uv_batch0) == 0:
            return {}
        
        # 获取wrist RGB图像
        wrist_rgb = batch["wrist_image"][0].cpu().numpy()  # 取batch_0
        wrist_rgb = wrist_rgb.astype(np.uint8)
        
        # Resize wrist_rgb从1280x720到518x294
        # print(wrist_rgb.shape)
        # if wrist_rgb.shape[1] == 1280:
        wrist_rgb = cv2.resize(wrist_rgb, (518, 294))
        # else:
        #     wrist_rgb = cv2.resize(wrist_rgb, (518, 518))
        
        H, W = wrist_rgb.shape[:2]
        
        # 获取预测的wrist pose和GT wrist intrinsics
        wrist_pose_enc = predictions["wrist_pose_enc_list"][-1][0]  # [1, 9] - batch_0
        wrist_pose_enc = wrist_pose_enc[0]  # [9] - 取唯一的wrist pose
        
        # 转换为extrinsic和intrinsic
        wrist_ext_pred, _ = pose_encoding_to_extri_intri(
            wrist_pose_enc.unsqueeze(0).unsqueeze(0),  # [1, 1, 9]
            image_size_hw=(H, W),
            build_intrinsics=False
        )
        wrist_ext_pred = wrist_ext_pred[0, 0].cpu().numpy()  # [3, 4]
        
        # 使用GT wrist intrinsics
        wrist_intrinsics_gt = batch["wrist_intrinsics"][0].cpu().numpy()  # [3, 3] - batch_0
        
        # 获取预测的world points
        world_points = predictions["world_points"][0]  # [2, H, W, 3] - batch_0
        
        # 创建可视化图像
        # 1. 真实wrist view + track点
        wrist_with_tracks = wrist_rgb.copy()
        
        # 2. 生成wrist point cloud projection
        wrist_projection = self._generate_wrist_point_cloud_projection(
            predictions=predictions,
            batch=batch,
            image_shape=(H, W)
        )
        
        # 3. 对比图像（左右拼接）
        comparison_img = np.zeros((H, W*2, 3), dtype=np.uint8)
        # 处理每个track pair
        valid_projections = 0
        total_pairs = len(wrist_uv_batch0)
        comparison_img[:, :W] = wrist_with_tracks  # 左半边：真实wrist view + track点
        comparison_img[:, W:] = wrist_projection   # 右半边：wrist point cloud projection
        
        for i in range(total_pairs):
            wrist_uv = wrist_uv_batch0[i]
            pair_type = pair_type_batch0[i]
            confidence = confidence_batch0[i]
            
            # 跳过低置信度的点
            if confidence < 0.1:
                continue
            
            # 在真实wrist view上画点
            wrist_u, wrist_v = int(wrist_uv[0]), int(wrist_uv[1])
            if 0 <= wrist_u < W and 0 <= wrist_v < H:
                cv2.circle(comparison_img, (wrist_u, wrist_v), 3, (0, 255, 0), -1)  # 绿色点
            ext_uv = ext_uv_batch0[i]
            world_points_seq = pair_type
            
            # 跳过无效的ext UV坐标
            if ext_uv[0] < 0 or ext_uv[1] < 0:
                continue
            
            # 从world points获取3D点
            try:
                point_3d = self._get_interpolated_3d_point_numpy(
                    world_points[world_points_seq].cpu().numpy(),  # [H, W, 3]
                    ext_uv[0],  # u coordinate
                    ext_uv[1]   # v coordinate
                )
                
                # 投影到wrist view
                projected_uv, depth, is_valid = self._project_3d_to_wrist_numpy(
                    point_3d,
                    wrist_ext_pred,
                    wrist_intrinsics_gt
                )
                if is_valid:
                    
                    # print(is_valid,projected_uv)
                    # 在投影图像上画点
                    proj_u, proj_v = int(projected_uv[0][0]), int(projected_uv[0][1])
                    if 0 <= proj_u < W and 0 <= proj_v < H:
                        cv2.circle(comparison_img, (W + proj_u, proj_v), 3, (255, 0, 0), -1)  # 红色点
                        cv2.circle(comparison_img, (wrist_u, wrist_v), 3, (0, 255, 0), -1)  # 绿色点
                        cv2.line(comparison_img, (wrist_u, wrist_v), (W + proj_u, proj_v), (255, 255, 255), 1)
                        
                        valid_projections += 1
                
            except Exception as e:
                print(e)
                import traceback
                traceback.print_exc()
                continue
        
        # 保存图像
        comparison_path = output_dir / "wrist_tracking_comparison.png"
        
        cv2.imwrite(str(comparison_path), cv2.cvtColor(comparison_img, cv2.COLOR_RGB2BGR))
        
        return {
            "wrist_tracking_comparison": str(comparison_path),
            "valid_projections": valid_projections,
            "total_track_pairs": total_pairs
        }
    
    def _get_interpolated_3d_point_numpy(
        self,
        world_points_map: np.ndarray,
        u: float,
        v: float
    ) -> np.ndarray:
        """
        使用numpy进行双线性插值获取3D点
        
        Args:
            world_points_map: 3D世界点云图 [H, W, 3]
            u: U坐标（浮点数）
            v: V坐标（浮点数）
            
        Returns:
            3D点 [3]
        """
        H, W, _ = world_points_map.shape
        
        # 获取四个角点索引
        u0, u1 = int(np.floor(u)), int(np.ceil(u))
        v0, v1 = int(np.floor(v)), int(np.ceil(v))
        
        # 计算插值权重
        wu = u - u0
        wv = v - v0
        
        # 处理边界情况
        if u0 == u1:
            wu = 0.0
        if v0 == v1:
            wv = 0.0
        
        # 获取四个角点（处理边界）
        u0_clamped = max(0, min(u0, W-1))
        u1_clamped = max(0, min(u1, W-1))
        v0_clamped = max(0, min(v0, H-1))
        v1_clamped = max(0, min(v1, H-1))
        
        p00 = world_points_map[v0_clamped, u0_clamped, :]
        p01 = world_points_map[v0_clamped, u1_clamped, :]
        p10 = world_points_map[v1_clamped, u0_clamped, :]
        p11 = world_points_map[v1_clamped, u1_clamped, :]
        
        # 双线性插值
        point_3d = (1-wu)*(1-wv)*p00 + wu*(1-wv)*p01 + (1-wu)*wv*p10 + wu*wv*p11
        
        return point_3d
    
    def _project_3d_to_wrist_numpy(
        self,
        point_3d: np.ndarray,
        wrist_ext: np.ndarray,
        wrist_intrinsics: np.ndarray
    ) -> Tuple[np.ndarray, float, bool]:
        """
        使用numpy将3D点投影到wrist相机视角
        
        Args:
            point_3d: 世界坐标系中的3D点 [3]
            wrist_ext: wrist相机外参矩阵 [3, 4] - world2camera变换矩阵
            wrist_intrinsics: wrist相机内参矩阵 [3, 3]
            
        Returns:
            Tuple of (projected_uv, depth, is_valid)
            - projected_uv: [2] - 投影的UV坐标
            - depth: 深度值
            - is_valid: 是否有效投影
        """
        # 对外参求逆：从camera2world变为world2camera
        wrist_ext_4x4 = np.vstack([wrist_ext, [0, 0, 0, 1]])  # 扩展为4x4齐次坐标矩阵
        world2camera_ext = wrist_ext_4x4[:3, :4]  # 求逆得到world2camera变换
        
        # 转换为齐次坐标
        point_homo = np.append(point_3d, 1.0)  # [4]
        
        # 投影到相机坐标系
        point_cam = world2camera_ext @ point_homo  # [3]
        
        # 检查深度是否为正
        depth = point_cam[2]
        if depth <= 0.01:
            return np.array([0, 0]), depth, False
        
        # 投影到图像平面
        point_2d = point_cam[:2] / depth  # [2]
        
        # 应用内参
        point_2d_homo = np.append(point_2d, 1.0)  # [3]
        point_pixel = wrist_intrinsics @ point_2d_homo  # [3]
        projected_uv = point_pixel[:2]  # [2]
        
        return projected_uv, depth, True
    
    def _generate_wrist_point_cloud_projection(
        self,
        predictions: Dict,
        batch: Dict,
        image_shape: Tuple[int, int]
    ) -> np.ndarray:
        """
        生成wrist point cloud projection图像
        
        Args:
            predictions: 模型预测结果
            batch: 输入batch数据
            image_shape: 输出图像形状 (H, W)
            
        Returns:
            wrist point cloud projection图像 (H, W, 3)
        """
        H, W = image_shape
        
        # 获取预测的wrist pose和world points
        wrist_pose_enc = predictions["wrist_pose_enc_list"][-1][0]  # [1, 9] - batch_0
        wrist_pose_enc = wrist_pose_enc[0]  # [9] - 取唯一的wrist pose
        
        # 转换为extrinsic和intrinsic
        wrist_ext_pred, _ = pose_encoding_to_extri_intri(
            wrist_pose_enc.unsqueeze(0).unsqueeze(0),  # [1, 1, 9]
            image_size_hw=(H, W),
            build_intrinsics=False
        )
        wrist_ext_pred = wrist_ext_pred[0, 0].cpu().numpy()  # [3, 4]
        
        # 使用GT wrist intrinsics
        wrist_intrinsics_gt = batch["wrist_intrinsics"][0].cpu().numpy()  # [3, 3] - batch_0
        
        # 获取预测的world points
        world_points = predictions["world_points"][0]  # [2, H, W, 3] - batch_0
        
        # 合并所有视角的world points
        all_world_points = world_points.reshape(-1, 3)  # [N, 3]
        
        # 生成颜色（使用所有视角的RGB）
        images = batch["images"][0]  # [S, C, H, W] - batch_0
        colors_list = []
        for cam_idx in range(images.shape[0]):
            rgb = images[cam_idx]
            if torch.is_tensor(rgb):
                rgb = rgb.cpu().numpy()
            rgb = np.transpose(rgb, (1, 2, 0))
            assert rgb.shape[-1] == 3, f"RGB形状错误: {rgb.shape}"
            colors_list.append(rgb.reshape(-1, 3))
        colors = np.concatenate(colors_list, axis=0)
        
        # 确保点云和颜色数量匹配
        if len(all_world_points) != len(colors):
            logging.warning(f"Wrist projection: 点云和颜色数量不匹配: {len(all_world_points)} vs {len(colors)}")
            # 如果数量不匹配，截取到较小的数量
            min_count = min(len(all_world_points), len(colors))
            all_world_points = all_world_points[:min_count]
            colors = colors[:min_count]
        
        # 投影到wrist视角
        wrist_projection = self.visualize_point_cloud_projection(
            points_3d=all_world_points.cpu().numpy(),
            point_colors=colors,
            camera_extrinsics=wrist_ext_pred,
            camera_intrinsics=wrist_intrinsics_gt[0],  # 使用GT intrinsic
            image_shape=(H, W),
            need_inverse=False  # 对于wrist，是world2camera，不需要求逆
        )
        
        return wrist_projection