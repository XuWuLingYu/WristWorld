#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VGGT点云可视化脚本

输入：
- VGGT checkpoint路径
- ext1视频路径
- ext2视频路径

功能：
1. 加载VGGT模型（使用与prepare_condition_clips.py相同的参数）
2. 推理首帧点云
3. 使用ext1、ext2外参的球面插值均值
4. 使用ext1内参
5. 渲染1920x1080高清图像，背景透明
6. 保存PNG

球面插值说明：
- 将旋转矩阵转换为四元数
- 对四元数进行球面线性插值（SLERP）
- 对平移向量进行线性插值
- 重新组合为外参矩阵
"""

import os
import sys
import argparse
import numpy as np
import cv2
from PIL import Image
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Tuple, List, Optional
import json
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
import trimesh

# 添加VGGT路径
sys.path.append('/mnt/zezhong/vggt_training')
sys.path.append('/mnt/zezhong/vggt_training/vggt')
sys.path.append('/mnt/zezhong/vggt_training/training')

try:
    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri, extri_intri_to_pose_encoding
    from vggt.utils.geometry import unproject_depth_map_to_point_map
except ImportError as e:
    print(f"导入VGGT模块失败: {e}")
    print("请确保脚本在正确的环境中运行，并且路径设置正确")
    sys.exit(1)


class VGGTPointCloudVisualizer:
    def __init__(self, checkpoint_path: str, device: str = "cuda"):
        """
        初始化VGGT点云可视化器
        
        Args:
            checkpoint_path: 模型checkpoint路径
            device: 推理设备
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.checkpoint_path = checkpoint_path
        self.model = self._load_model()
        
        print(f"✅ VGGT点云可视化器初始化完成，设备: {self.device}")
    
    def _load_model(self) -> VGGT:
        """
        加载VGGT模型（与prepare_condition_clips.py完全一致）
        
        Returns:
            加载的VGGT模型
        """
        print(f"📦 正在加载模型checkpoint: {self.checkpoint_path}")
        
        # 与prepare_condition_clips.py完全一致的模型配置
        model = VGGT(
            img_size=518,
            patch_size=14,
            embed_dim=1024,
            enable_camera=True,      # 需要相机参数预测
            enable_depth=True,       # 需要深度预测
            enable_point=True,       # 需要点云预测
            enable_track=False,      # 不需要track预测
            enable_wrist=True,       # 需要wrist pose预测
            pretrained="facebook/VGGT-1B",
            use_lora=False,          # 不使用LoRA
            lora_rank=16,
            lora_alpha=32
        )
        
        # 禁用track_head参数的梯度（与训练配置一致）
        for name, param in model.named_parameters():
            if "track_head" in name:
                param.requires_grad = False
        
        # 加载checkpoint
        try:
            checkpoint = torch.load(self.checkpoint_path, map_location="cpu")
            
            if "model" in checkpoint:
                model_state_dict = checkpoint["model"]
            else:
                model_state_dict = checkpoint
                
            missing_keys, unexpected_keys = model.load_state_dict(model_state_dict, strict=False)
            
            if missing_keys:
                print(f"⚠️ 缺失的键: {missing_keys}")
            if unexpected_keys:
                print(f"⚠️ 未预期的键: {unexpected_keys}")
                
            print("✅ 模型加载成功")
            
        except Exception as e:
            print(f"❌ 加载checkpoint失败: {e}")
            print("🔄 尝试加载预训练模型...")
            try:
                model = VGGT.from_pretrained("facebook/VGGT-1B")
                print("✅ 预训练模型加载成功")
            except Exception as e2:
                print(f"❌ 预训练模型加载也失败: {e2}")
                sys.exit(1)
        
        model.eval()
        model.to(self.device)
        return model
    
    def _get_video_frame_count(self, video_path: str) -> int:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件: {video_path}")
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return max(total, 0)

    def extract_first_frame(self, video_path: str,frame_num=0) -> np.ndarray:
        """
        提取视频首帧
        
        Args:
            video_path: 视频文件路径
            
        Returns:
            首帧RGB图像
        """
        print(f"🎬 提取首帧: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件: {video_path}")
        
        for i in range(frame_num):
            ret, frame = cap.read()
            if not ret:
                raise ValueError(f"无法读取视频首帧: {video_path}")
        
        ret, frame = cap.read()
        if not ret:
            raise ValueError(f"无法读取视频首帧: {video_path}")
        
        # 转换BGR到RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cap.release()
        
        print(f"✅ 首帧提取完成，尺寸: {frame_rgb.shape}")
        return frame_rgb
    
    def preprocess_frames(self, frame1: np.ndarray, frame2: np.ndarray) -> torch.Tensor:
        """
        预处理帧对（与prepare_condition_clips.py一致）
        
        Args:
            frame1: 第一帧
            frame2: 第二帧
        """
        import tempfile, shutil
        temp_dir = Path(tempfile.mkdtemp(prefix=f"vggt_temp_{os.getpid()}_", dir=str(Path.cwd())))
        try:
            frame1_path = temp_dir / "frame1.jpg"
            frame2_path = temp_dir / "frame2.jpg"
            Image.fromarray(frame1).save(frame1_path)
            Image.fromarray(frame2).save(frame2_path)
            
            try:
                images = load_and_preprocess_images([str(frame1_path), str(frame2_path)])
            except ImportError:
                images = self._simple_preprocess_images([str(frame1_path), str(frame2_path)])
            
            return images.to(self.device)
        finally:
            try:
                shutil.rmtree(temp_dir, ignore_errors=True)
            except Exception:
                pass
    
    def _simple_preprocess_images(self, image_paths: List[str]) -> torch.Tensor:
        """
        简化的图像预处理（与prepare_condition_clips.py一致）
        """
        import torchvision.transforms as TF
        
        images = []
        to_tensor = TF.ToTensor()
        
        for image_path in image_paths:
            img = Image.open(image_path)
            if img.mode == "RGBA":
                background = Image.new("RGBA", img.size, (255, 255, 255, 255))
                img = Image.alpha_composite(background, img)
            img = img.convert("RGB")
            
            img = img.resize((518, 518), Image.Resampling.BICUBIC)
            img_tensor = to_tensor(img)
            images.append(img_tensor)
        
        return torch.stack(images)
    
    def run_inference(self, frame1: np.ndarray, frame2: np.ndarray) -> dict:
        """
        运行VGGT推理
        
        Args:
            frame1: 第一帧
            frame2: 第二帧
            
        Returns:
            推理结果
        """
        print("🔄 运行VGGT推理...")
        
        # 预处理
        images = self.preprocess_frames(frame1, frame2)
        images = images.unsqueeze(0)  # 添加batch维度
        
        # 推理
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=dtype):
                predictions = self.model(images)
        
        # 转换pose编码为extrinsic/intrinsic矩阵
        if "pose_enc" in predictions:
            extrinsic, intrinsic = pose_encoding_to_extri_intri(
                predictions["pose_enc"], 
                image_size_hw=(294, 518),  # 与训练时的图像尺寸一致
                build_intrinsics=True
            )
            predictions["extrinsic"] = extrinsic
            predictions["intrinsic"] = intrinsic
        
        # 转换wrist pose编码（如果存在）
        if "wrist_pose_enc" in predictions:
            wrist_extrinsic, wrist_intrinsic = pose_encoding_to_extri_intri(
                predictions["wrist_pose_enc"], 
                image_size_hw=(294, 518),  # 与训练时的图像尺寸一致
                build_intrinsics=True
            )
            predictions["wrist_extrinsic"] = wrist_extrinsic
            predictions["wrist_intrinsic"] = wrist_intrinsic
        
        # 添加原始images
        predictions["images"] = images.cpu().numpy()
        
        # 转换为numpy
        for key in predictions.keys():
            if isinstance(predictions[key], torch.Tensor):
                predictions[key] = predictions[key].cpu().numpy()
        
        print("✅ VGGT推理完成")
        return predictions
    
    def spherical_interpolate_extrinsics(self, ext1: np.ndarray, ext2: np.ndarray, t: float = 0.5) -> np.ndarray:
        """
        使用球面插值计算两个外参矩阵的插值（支持外推）
        
        Args:
            ext1: 第一个外参矩阵 (3, 4)
            ext2: 第二个外参矩阵 (3, 4)
            t: 插值参数，0.5表示中点，>1表示外推，<0表示反向外推
            
        Returns:
            插值后的外参矩阵 (3, 4)
        """
        # 提取旋转矩阵和平移向量
        R1 = ext1[:3, :3]
        t1 = ext1[:3, 3]
        R2 = ext2[:3, :3]
        t2 = ext2[:3, 3]
        
        # 将旋转矩阵转换为四元数
        r1 = R.from_matrix(R1)
        r2 = R.from_matrix(R2)
        
        # 处理外推插值
        if t < 0 or t > 1:
            # 对于外推，我们需要计算从r1到r2的旋转增量
            # 然后按比例应用这个增量
            r1_to_r2 = r2 * r1.inv()  # 从r1到r2的相对旋转
            
            # 计算旋转增量
            r_increment = r1_to_r2
            r_interp = r1 * (r_increment ** t)  # 应用t倍的旋转增量
        else:
            # 正常插值
            key_rots = R.concatenate([r1, r2])
            key_times = [0, 1]
            slerp = Slerp(key_times, key_rots)
            r_interp = slerp(t)
        
        # 线性插值平移向量（支持外推）
        t_interp = (1 - t) * t1 + t * t2
        
        # 重新组合外参矩阵
        ext_interp = np.eye(4)
        ext_interp[:3, :3] = r_interp.as_matrix()
        ext_interp[:3, 3] = t_interp
        
        print(f"🔄 球面插值信息:")
        print(f"  插值参数 t: {t}")
        print(f"  插值类型: {'外推' if t < 0 or t > 1 else '内插'}")
        print(f"  旋转角度差: {np.linalg.norm(R1 - R2):.3f}")
        print(f"  平移距离差: {np.linalg.norm(t1 - t2):.3f}")
        
        return ext_interp[:3, :4]  # 返回3x4格式
    
    def _project_world_points_to_view(self,
                                      world_points: np.ndarray,
                                      view_extrinsic_w2c: np.ndarray,
                                      view_intrinsic: np.ndarray,
                                      img_size: Tuple[int, int]) -> np.ndarray:
        """
        将世界坐标点投影到当前视角图像
        Args:
            world_points: (N, 3)
            view_extrinsic_w2c: (3, 4) world->camera
            view_intrinsic: (3, 3)
            img_size: (H, W)
        Returns:
            像素坐标 (N, 2)，不可见点返回NaN
        """
        H, W = img_size
        ones = np.ones((world_points.shape[0], 1), dtype=world_points.dtype)
        pts_h = np.concatenate([world_points, ones], axis=1).T  # (4, N)

        T_view = np.vstack([view_extrinsic_w2c, [0, 0, 0, 1]])  # (4,4)
        cam_pts = (T_view @ pts_h)[:3].T  # (N,3)

        # 仅投影到前方
        z = cam_pts[:, 2]
        valid = z > 0
        pixels = np.full((world_points.shape[0], 2), np.nan, dtype=np.float32)
        if not np.any(valid):
            return pixels

        uvw = (view_intrinsic @ cam_pts[valid].T).T  # (M,3)
        uv = uvw[:, :2] / uvw[:, 2:3]
        # 边界检查
        in_bounds = (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
        valid_idx = np.where(valid)[0]
        pixels[valid_idx[in_bounds]] = uv[in_bounds]
        return pixels

    def _draw_wrist_frustum(self,
                            image_rgba: np.ndarray,
                            wrist_extrinsic_w2c: np.ndarray,
                            wrist_intrinsic: np.ndarray,
                            view_extrinsic_w2c: np.ndarray,
                            view_intrinsic: np.ndarray,
                            img_size: Tuple[int, int] = (1080, 1920),
                            frustum_length: float = 0.2,
                            color_bgra: Tuple[int, int, int, int] = (0, 255, 0, 255),
                            thickness: int = 2) -> np.ndarray:
        """
        在投影图像上绘制基于wrist相机位姿的相机光锥示意图。
        - 使用wrist内参在其成像平面四角发射射线，长度为frustum_length（相机坐标系单位）。
        - 将这些点从wrist相机坐标变换到世界坐标，再用当前视角投影到图像上绘线。
        """
        H_in, W_in = 294, 518  # 与上游生成wrist_intrinsic时的大小一致
        # 计算wrist相机的相机到世界变换
        T_wrist = np.vstack([wrist_extrinsic_w2c, [0, 0, 0, 1]])  # world->wrist_cam
        T_wrist_inv = np.linalg.inv(T_wrist)  # wrist_cam->world

        # 构造四个角点的像素坐标以及相机原点
        corners_px = np.array([
            [0, 0, 1],
            [W_in - 1, 0, 1],
            [W_in - 1, H_in - 1, 1],
            [0, H_in - 1, 1],
        ], dtype=np.float32).T  # (3,4)

        K_inv = np.linalg.inv(wrist_intrinsic)
        rays_cam = K_inv @ corners_px  # (3,4)
        # 归一化方向
        rays_cam = rays_cam / np.linalg.norm(rays_cam, axis=0, keepdims=True)

        # 相机原点与角点(相机坐标系)
        origin_cam = np.array([[0, 0, 0, 1]], dtype=np.float32).T  # (4,1)
        corners_cam = np.vstack([rays_cam * frustum_length, np.ones((1, 4), dtype=np.float32)])  # (4,4)

        # 变换到世界坐标
        origin_world = (T_wrist_inv @ origin_cam)[:3].T  # (1,3)
        corners_world = (T_wrist_inv @ corners_cam)[:3].T  # (4,3)

        # 投影到当前视角图像
        pts_world = np.vstack([origin_world, corners_world])  # (5,3)
        uv = self._project_world_points_to_view(pts_world, view_extrinsic_w2c, view_intrinsic, img_size)

        # 转换到BGRA以便cv2绘制
        img_bgra = cv2.cvtColor(image_rgba, cv2.COLOR_RGBA2BGRA)

        # 连接原点到四个角
        for j in range(1, 5):
            p0, p1 = uv[0], uv[j]
            if not (np.any(np.isnan(p0)) or np.any(np.isnan(p1))):
                cv2.line(img_bgra,
                         (int(round(p0[0])), int(round(p0[1]))),
                         (int(round(p1[0])), int(round(p1[1]))),
                         color_bgra, thickness)

        # 连接角点之间以形成锥体边框
        edges = [(1, 2), (2, 3), (3, 4), (4, 1)]
        for a, b in edges:
            p0, p1 = uv[a], uv[b]
            if not (np.any(np.isnan(p0)) or np.any(np.isnan(p1))):
                cv2.line(img_bgra,
                         (int(round(p0[0])), int(round(p0[1]))),
                         (int(round(p1[0])), int(round(p1[1]))),
                         color_bgra, thickness)

        return cv2.cvtColor(img_bgra, cv2.COLOR_BGRA2RGBA)

    def _adjust_intrinsic_for_larger_fov(self, intrinsic: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """
        调整内参以增大视野（焦距减半，主点调整到目标图像中心）
        
        Args:
            intrinsic: 原始内参矩阵 (3, 3)
            target_size: 目标图像尺寸 (H, W)
            
        Returns:
            调整后的内参矩阵 (3, 3)
        """
        H, W = target_size
        
        # 创建新的内参矩阵
        new_intrinsic = intrinsic.copy()
        
        # 焦距减半（增大一倍视野）
        new_intrinsic[0, 0] = intrinsic[0, 0] * 3  # fx
        new_intrinsic[1, 1] = intrinsic[1, 1] * 3  # fy
        
        # 主点调整到目标图像中心
        new_intrinsic[0, 2] = W / 2.0-400  # cx
        new_intrinsic[1, 2] = H / 2.0-200  # cy
        
        print(f"🔧 内参调整:")
        print(f"  原始焦距: fx={intrinsic[0, 0]:.1f}, fy={intrinsic[1, 1]:.1f}")
        print(f"  原始主点: cx={intrinsic[0, 2]:.1f}, cy={intrinsic[1, 2]:.1f}")
        print(f"  调整后焦距: fx={new_intrinsic[0, 0]:.1f}, fy={new_intrinsic[1, 1]:.1f}")
        print(f"  调整后主点: cx={new_intrinsic[0, 2]:.1f}, cy={new_intrinsic[1, 2]:.1f}")
        
        return new_intrinsic
    
    def _render_point(self, image: np.ndarray, u: int, v: int, color: np.ndarray, radius: int = 2):
        """
        在图像上渲染一个点（小圆形）
        
        Args:
            image: 目标图像 (H, W, 4)
            u, v: 像素坐标
            color: 颜色 (R, G, B)
            radius: 点的半径
        """
        H, W = image.shape[:2]
        
        # 绘制小圆形
        for du in range(-radius, radius + 1):
            for dv in range(-radius, radius + 1):
                if du*du + dv*dv <= radius*radius:
                    new_u, new_v = u + du, v + dv
                    if 0 <= new_u < W and 0 <= new_v < H:
                        # 使用alpha混合
                        alpha = 255
                        image[new_v, new_u] = [color[0], color[1], color[2], alpha]
    
    def generate_point_cloud(self, predictions: dict) -> Tuple[np.ndarray, np.ndarray]:
        """
        生成点云坐标和颜色
        
        Args:
            predictions: 推理结果
            
        Returns:
            点云坐标和颜色
        """
        if "world_points" not in predictions:
            raise ValueError("预测结果中未包含world_points")
        
        # 使用预测的点云
        points_3d = predictions["world_points"]
        if torch.is_tensor(points_3d):
            points_3d = points_3d.cpu().numpy()
        
        # 获取图像数据
        images_sample = predictions["images"][0]  # 移除batch维度
        
        # 双视角模式：images_sample形状为 (2, 3, 294, 518)
        ext1_rgb = images_sample[0]  # (3, 294, 518)
        ext2_rgb = images_sample[1]  # (3, 294, 518)
        
        if torch.is_tensor(ext1_rgb):
            ext1_rgb = ext1_rgb.cpu().numpy()
        if torch.is_tensor(ext2_rgb):
            ext2_rgb = ext2_rgb.cpu().numpy()
        
        # 转置为(H, W, C)格式
        ext1_rgb = np.transpose(ext1_rgb, (1, 2, 0))  # (294, 518, 3)
        ext2_rgb = np.transpose(ext2_rgb, (1, 2, 0))  # (294, 518, 3)
        
        # 将两个相机的RGB图像合并为一个大的颜色数组
        colors = np.concatenate([ext1_rgb.reshape(-1, 3), ext2_rgb.reshape(-1, 3)], axis=0)  # (2*294*518, 3)
        
        # 确保点云和颜色数量匹配
        points_3d = points_3d.reshape(-1, 3)
        assert len(points_3d) == len(colors), f"点云和颜色数量不匹配: {len(points_3d)} vs {len(colors)}"
        
        return points_3d, colors
    
    def project_points_to_camera(self, points_3d: np.ndarray, colors: np.ndarray, 
                                extrinsic: np.ndarray, intrinsic: np.ndarray,
                                img_size: Tuple[int, int] = (1080, 1920)) -> np.ndarray:
        """
        将点云投影到相机视角
        
        Args:
            points_3d: 3D点云坐标 (N, 3) - 世界坐标系
            colors: 点云颜色 (N, 3)
            extrinsic: 外参 (3, 4) - world2camera变换矩阵
            intrinsic: 内参 (3, 3)
            img_size: 输出图像尺寸 (H, W) - 默认(1080, 1920)
            
        Returns:
            投影图像 (H, W, 4) - RGBA格式，背景透明
        """
        H, W = img_size
        
        # 创建白色背景 (RGBA 全 255)
        image = np.full((H, W, 4), 255, dtype=np.uint8)
        
        # 将点云从世界坐标转换到相机坐标
        points_homo = np.concatenate([points_3d, np.ones((points_3d.shape[0], 1))], axis=1)  # (N, 4)
        points_cam = (extrinsic @ points_homo.T).T  # (N, 3)
        
        # 过滤掉相机后面的点
        valid_mask = points_cam[:, 2] > 0 
        points_cam = points_cam[valid_mask]
        colors = colors[valid_mask]
        valid_mask = points_cam[:, 2] <1
        points_cam = points_cam[valid_mask]
        colors = colors[valid_mask]
        
        if len(points_cam) == 0:
            print("⚠️ 没有有效的投影点")
            return image
        
        # 投影到图像平面
        points_2d = (intrinsic @ points_cam.T).T  # (N, 3)
        points_2d = points_2d[:, :2] / points_2d[:, 2:3]  # 透视除法
        
        # 过滤掉图像边界外的点
        valid_mask = ((points_2d[:, 0] >= 0) & (points_2d[:, 0] < W) & 
                     (points_2d[:, 1] >= 0) & (points_2d[:, 1] < H))
        points_2d = points_2d[valid_mask]
        colors = colors[valid_mask]
        points_cam = points_cam[valid_mask]  # 保持相机坐标用于深度排序
        
        if len(points_2d) == 0:
            print("⚠️ 没有投影到图像内的点")
            return image
        
        # 按照深度（Z坐标）排序，实现正确的3D遮挡关系
        # 深度值越小（越近）的点排在后面，这样会被先渲染，近处的点会覆盖远处的点
        
        depth_values = points_cam[:, 2]  # Z坐标作为深度
        sort_indices = np.argsort(depth_values)[::-1]  # 降序排列，深度大的（远的）先渲染
        
        points_2d_sorted = points_2d[sort_indices]
        colors_sorted = colors[sort_indices]
        depth_sorted = depth_values[sort_indices]
        
        print(f"🔍 深度排序信息:")
        print(f"  最近点深度: {depth_sorted[-1]:.3f}")
        print(f"  最远点深度: {depth_sorted[0]:.3f}")
        print(f"  深度范围: {depth_sorted[0] - depth_sorted[-1]:.3f}")
        
        # 将点投影到图像上（按深度排序渲染）
        for i, (u, v) in enumerate(points_2d_sorted):
            u_int, v_int = int(round(u)), int(round(v))
            if 0 <= u_int < W and 0 <= v_int < H:
                # 确保颜色值在有效范围内
                color = np.clip(colors_sorted[i] * 255, 0, 255).astype(np.uint8)
                # 渲染点（可以扩展为小圆形以提高可见性）
                self._render_point(image, u_int, v_int, color, radius=4)
        
        print(f"✅ 投影完成，有效点数: {len(points_2d)}")
        return image
    
    def save_point_cloud_as_glb(self, points: np.ndarray, colors: np.ndarray, output_path: str):
        """
        将点云保存为GLB格式
        
        Args:
            points: 点云坐标 (N, 3)
            colors: 点云颜色 (N, 3)
            output_path: 输出文件路径
        """
        print(f"💾 保存点云为GLB格式: {output_path}")
        
        # 确保颜色值在正确范围内
        if colors.max() <= 1.0:
            colors = (colors * 255).astype(np.uint8)
        else:
            colors = colors.astype(np.uint8)
        
        # 创建trimesh点云对象
        point_cloud = trimesh.PointCloud(
            vertices=points,
            colors=colors
        )
        
        # 导出为GLB格式
        point_cloud.export(output_path)
        print(f"✅ GLB点云保存完成: {output_path}")
    
    def add_wrist_origin_sphere(self, points_3d: np.ndarray, colors: np.ndarray, 
                               wrist_extrinsic: np.ndarray, radius: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
        """
        在点云中添加红球表示wrist origin
        
        Args:
            points_3d: 原始点云坐标 (N, 3)
            colors: 原始点云颜色 (N, 3)
            wrist_extrinsic: wrist相机外参矩阵 (3, 4) - world2camera变换矩阵
            radius: 球体半径
            
        Returns:
            包含红球的点云坐标和颜色
        """
        # 提取wrist origin位置
        # wrist_extrinsic是world2camera变换矩阵T_wc
        # 要得到相机在世界坐标系中的位置，需要求逆：T_cw = inv(T_wc)
        # 相机位置 = T_cw * [0,0,0,1] = T_cw的平移部分
        wrist_ext_4x4 = np.vstack([wrist_extrinsic, [0, 0, 0, 1]])  # 扩展为4x4齐次坐标矩阵
        wrist_ext_inv = np.linalg.inv(wrist_ext_4x4)  # 求逆得到camera2world变换
        wrist_origin = wrist_ext_inv[:3, 3]  # 取逆矩阵的平移部分
        
        # 生成红球点云
        sphere_points, sphere_colors = self._generate_sphere_points(
            center=wrist_origin,
            radius=radius,
            color=(255, 0, 0),  # 红色
            num_points=100
        )
        
        # 合并原始点云和红球
        combined_points = np.vstack([points_3d, sphere_points])
        combined_colors = np.vstack([colors, sphere_colors])
        
        print(f"✅ 添加wrist origin红球，球心位置: {wrist_origin}")
        return combined_points, combined_colors
    
    def _generate_sphere_points(self, center: np.ndarray, radius: float, 
                               color: Tuple[int, int, int], num_points: int = 100) -> Tuple[np.ndarray, np.ndarray]:
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
    
    def visualize_point_cloud(self, ext1_video_path: str, ext2_video_path: str, 
                             output_path: str, save_glb: bool = True) -> dict:
        """
        完整的点云可视化流程
        
        Args:
            ext1_video_path: ext1视频路径
            ext2_video_path: ext2视频路径
            output_path: 输出PNG路径
            save_glb: 是否保存GLB格式点云
            
        Returns:
            结果统计信息
        """
        print("🎯 开始点云可视化流程...")
        # 计算两路视频的总帧数，取最小以保持同步
        total_frames_ext1 = self._get_video_frame_count(ext1_video_path)
        total_frames_ext2 = self._get_video_frame_count(ext2_video_path)
        total_frames = min(total_frames_ext1, total_frames_ext2)

        output_dir = Path("./pointcloud_vis")
        output_dir.mkdir(parents=True, exist_ok=True)

        for i in range(0, total_frames, 1):
            # 1. 提取首帧
            frame1 = self.extract_first_frame(ext1_video_path,frame_num=i)
            frame2 = self.extract_first_frame(ext2_video_path,frame_num=i)
            
            # 2. 运行VGGT推理
            predictions = self.run_inference(frame1, frame2)
            
            # 3. 生成点云
            points_3d, colors = self.generate_point_cloud(predictions)
            
            # 4. 获取相机参数
            ext1_extrinsic = predictions["extrinsic"][0, 0]  # (3, 4)
            ext2_extrinsic = predictions["extrinsic"][0, 1]  # (3, 4)
            ext1_intrinsic = predictions["intrinsic"][0, 0]  # (3, 3)
            
            # 5. 球面插值计算平均外参
            avg_extrinsic = self.spherical_interpolate_extrinsics(ext1_extrinsic, ext2_extrinsic, t=0.5)
            
            # 6. 调整内参：增大一倍视野（焦距减半，主点调整到图像中心）
            adjusted_intrinsic = self._adjust_intrinsic_for_larger_fov(ext1_intrinsic, target_size=(1080, 1920))
            
            # 7. 投影到1920x1080图像
            projection_image = self.project_points_to_camera(
                points_3d, colors, ext1_extrinsic, adjusted_intrinsic, img_size=(1080, 1920)
            )
            
            # 8. 叠加wrist相机光锥（如果有）
            if "wrist_extrinsic" in predictions and "wrist_intrinsic" in predictions:
                wrist_extrinsic = predictions["wrist_extrinsic"][0, 0]  # (3,4)
                wrist_intrinsic = predictions["wrist_intrinsic"][0, 0]  # (3,3)
                projection_image = self._draw_wrist_frustum(
                    projection_image,
                    wrist_extrinsic,
                    wrist_intrinsic,
                    ext1_extrinsic,
                    adjusted_intrinsic,
                    img_size=(1080, 1920),
                    frustum_length=0.2,
                    color_bgra=(0, 255, 0, 255),
                    thickness=2,
                )
            
            # 9. 保存每帧可视化到 ./pointcloud_vis/{frame_id}.png
            per_frame_path = output_dir / f"{i}.png"
            pil_image = Image.fromarray(projection_image, 'RGBA')
            pil_image.save(per_frame_path, 'PNG')
            print(f"✅ 第{i}帧可视化完成，保存到: {per_frame_path}")
            
            # 10. 保存GLB格式点云（如果启用）
            glb_files = {}
            if save_glb:
                # 保存原始点云
                glb_output_dir = output_dir / "pointclouds"
                glb_output_dir.mkdir(exist_ok=True)
                
                # 原始点云GLB
                original_glb_path = glb_output_dir / f"frame_{i}_original.glb"
                self.save_point_cloud_as_glb(points_3d, colors, str(original_glb_path))
                glb_files["original_pointcloud"] = str(original_glb_path)
                
                # 带wrist origin红球的点云GLB（如果有wrist pose预测）
                # if "wrist_extrinsic" in predictions:
                #     wrist_extrinsic = predictions["wrist_extrinsic"][0, 0]  # (3, 4)
                #     points_with_sphere, colors_with_sphere = self.add_wrist_origin_sphere(
                #         points_3d, colors, wrist_extrinsic
                #     )
                #     sphere_glb_path = glb_output_dir / f"{output_path.stem}_with_wrist_sphere_{i}.glb"
                #     self.save_point_cloud_as_glb(points_with_sphere, colors_with_sphere, str(sphere_glb_path))
                #     glb_files["pointcloud_with_wrist_sphere"] = str(sphere_glb_path)
                # else:
                #     print("⚠️  未检测到wrist pose预测，跳过wrist origin红球添加")
            
            # 11. 生成结果统计
            results = {
                "input_videos": {
                    "ext1": ext1_video_path,
                    "ext2": ext2_video_path
                },
                "output_file": str(per_frame_path),
                "glb_files": glb_files,
                "statistics": {
                    "total_points": len(points_3d),
                    "valid_projection_points": len([p for p in points_3d if p[2] > 0]),
                    "image_resolution": "1920x1080",
                    "background_transparent": False,
                    "extrinsic_interpolation": "spherical_slerp",
                    "intrinsic_source": "ext1",
                    "glb_saved": save_glb
                }
            }
            
        return results


def main():
    parser = argparse.ArgumentParser(description="VGGT点云可视化脚本")
    parser.add_argument("--checkpoint", required=True, help="VGGT checkpoint路径")
    parser.add_argument("--ext1", required=True, help="ext1视频路径")
    parser.add_argument("--ext2", required=True, help="ext2视频路径")
    parser.add_argument("--output", required=True, help="输出PNG路径")
    parser.add_argument("--device", default="cuda", help="推理设备 (cuda/cpu)")
    parser.add_argument("--save-glb", action="store_true", default=True, help="是否保存GLB格式点云（默认：True）")
    parser.add_argument("--no-glb", action="store_true", help="禁用GLB保存")
    
    args = parser.parse_args()
    
    # 处理GLB保存选项
    save_glb = args.save_glb and not args.no_glb
    
    # 验证输入文件
    for video_path, name in [(args.ext1, "ext1"), (args.ext2, "ext2")]:
        if not os.path.exists(video_path):
            print(f"❌ 视频文件不存在: {video_path}")
            return
    
    if not os.path.exists(args.checkpoint):
        print(f"❌ Checkpoint文件不存在: {args.checkpoint}")
        return
    
    try:
        # 创建可视化器
        visualizer = VGGTPointCloudVisualizer(args.checkpoint, args.device)
        
        # 运行可视化
        results = visualizer.visualize_point_cloud(args.ext1, args.ext2, args.output, save_glb=save_glb)
        
        print("\n" + "="*80)
        print("📋 点云可视化结果摘要:")
        print(f"  输入视频: {args.ext1}, {args.ext2}")
        print(f"  模型checkpoint: {args.checkpoint}")
        print(f"  输出文件: {args.output}")
        print(f"  总点数: {results['statistics']['total_points']}")
        print(f"  有效投影点数: {results['statistics']['valid_projection_points']}")
        print(f"  图像分辨率: {results['statistics']['image_resolution']}")
        print(f"  背景透明: {results['statistics']['background_transparent']}")
        print(f"  外参插值: {results['statistics']['extrinsic_interpolation']}")
        print(f"  内参来源: {results['statistics']['intrinsic_source']}")
        print(f"  GLB保存: {results['statistics']['glb_saved']}")
        
        if results['glb_files']:
            print("\n📦 GLB文件:")
            for name, path in results['glb_files'].items():
                print(f"  {name}: {path}")
        
        print("="*80)
        
    except Exception as e:
        print(f"❌ 点云可视化过程中发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
