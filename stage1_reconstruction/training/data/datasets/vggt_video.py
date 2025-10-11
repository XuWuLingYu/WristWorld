# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import os.path as osp
import logging
import random
import numpy as np
import cv2
from PIL import Image
from typing import Dict, List, Tuple, Optional

from data.dataset_util import *
from data.base_dataset import BaseDataset


class VggtVideoDataset(BaseDataset):
    """
    VGGT Video Dataset for training with video input.
    
    This dataset implements:
    1. Loads frames from three video files: ext1, ext2, wrist
    2. Uses all frames from the videos as both train and val sets
    3. Generates zeros for GT data (depth, point clouds, etc.)
    4. Supports wrist pose prediction
    """
    
    def __init__(
        self,
        common_config,
        split: str = "train",
        ext1_video_path: str = "",
        ext2_video_path: str = "",
        wrist_video_path: str = "",
        enable_wrist_prediction: bool = True,
    ):
        """
        Initialize the VggtVideoDataset.

        Args:
            common_config: Configuration object with common settings.
            split (str): Dataset split, either 'train' or 'test' (both use same data).
            ext1_video_path (str): Path to ext1 video file.
            ext2_video_path (str): Path to ext2 video file.
            wrist_video_path (str): Path to wrist video file.
            enable_wrist_prediction (bool): Whether to enable wrist pose prediction.
        """
        super().__init__(common_conf=common_config)

        self.debug = common_config.debug
        self.training = common_config.training
        self.get_nearby = common_config.get_nearby
        self.load_depth = common_config.load_depth
        self.inside_random = common_config.inside_random
        self.allow_duplicate_img = common_config.allow_duplicate_img
        self.enable_wrist_prediction = enable_wrist_prediction

        # 验证视频文件路径
        if not ext1_video_path or not ext2_video_path or not wrist_video_path:
            raise ValueError("所有三个视频路径都必须提供: ext1_video_path, ext2_video_path, wrist_video_path")
        
        if not osp.exists(ext1_video_path):
            raise FileNotFoundError(f"ext1视频文件不存在: {ext1_video_path}")
        if not osp.exists(ext2_video_path):
            raise FileNotFoundError(f"ext2视频文件不存在: {ext2_video_path}")
        if not osp.exists(wrist_video_path):
            raise FileNotFoundError(f"wrist视频文件不存在: {wrist_video_path}")

        self.ext1_video_path = ext1_video_path
        self.ext2_video_path = ext2_video_path
        self.wrist_video_path = wrist_video_path

        # 加载视频帧
        self.ext1_frames = self._load_video_frames(ext1_video_path)
        self.ext2_frames = self._load_video_frames(ext2_video_path)
        self.wrist_frames = self._load_video_frames(wrist_video_path)

        # 检查帧数一致性
        frame_counts = [len(self.ext1_frames), len(self.ext2_frames), len(self.wrist_frames)]
        if not all(count == frame_counts[0] for count in frame_counts):
            logging.warning(f"视频帧数不一致: ext1={frame_counts[0]}, ext2={frame_counts[1]}, wrist={frame_counts[2]}")
            # 使用最小帧数
            min_frames = min(frame_counts)
            self.ext1_frames = self.ext1_frames[:min_frames]
            self.ext2_frames = self.ext2_frames[:min_frames]
            self.wrist_frames = self.wrist_frames[:min_frames]

        self.total_frames = len(self.ext1_frames)
        
        # 获取视频信息
        self.video_info = self._get_video_info(ext1_video_path)
        
        status = "Training" if self.training else "Test"
        logging.info(f"{status}: VGGT-Video Dataset - 总帧数: {self.total_frames}")
        logging.info(f"{status}: 视频分辨率: {self.video_info['width']}x{self.video_info['height']}")
        logging.info(f"{status}: 视频FPS: {self.video_info['fps']}")

    def _load_video_frames(self, video_path: str) -> List[np.ndarray]:
        """
        加载视频帧
        
        Args:
            video_path: 视频文件路径
            
        Returns:
            帧列表 (RGB格式)
        """
        logging.info(f"正在加载视频帧: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件: {video_path}")
        
        frames = []
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        for i in range(frame_count):
            ret, frame = cap.read()
            if not ret:
                break
            
            # 转换BGR到RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
        
        cap.release()
        logging.info(f"✅ 加载完成，帧数: {len(frames)}")
        return frames

    def _get_video_info(self, video_path: str) -> Dict:
        """
        获取视频信息
        
        Args:
            video_path: 视频文件路径
            
        Returns:
            视频信息字典
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件: {video_path}")
        
        info = {
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        }
        
        cap.release()
        return info

    def _generate_zeros_depth(self, height: int, width: int) -> np.ndarray:
        """
        生成zeros深度图
        
        Args:
            height: 图像高度
            width: 图像宽度
            
        Returns:
            zeros深度图
        """
        return np.zeros((height, width, 1), dtype=np.float32)

    def _generate_zeros_point_cloud(self, height: int, width: int) -> np.ndarray:
        """
        生成zeros点云
        
        Args:
            height: 图像高度
            width: 图像宽度
            
        Returns:
            zeros点云 (H*W, 3)
        """
        return np.zeros((height * width, 3), dtype=np.float32)

    def _generate_zeros_camera_params(self) -> Dict:
        """
        生成合理的相机参数
        
        Returns:
            相机参数字典
        """
        # 使用合理的相机内参（基于DROID数据集）
        # 假设图像尺寸为294x518，焦距约为图像宽度的一半
        fx = fy = 259.0  # 焦距约为图像宽度的一半
        cx = 259.0  # 主点x坐标（图像宽度的一半）
        cy = 147.0  # 主点y坐标（图像高度的一半）
        
        intrinsic = np.array([
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)
        
        # 外参使用单位矩阵（假设相机在世界坐标系原点）
        extrinsic = np.eye(4, dtype=np.float32)
        
        return {
            'extrinsic': extrinsic,
            'intrinsic': intrinsic
        }

    def _preprocess_frame(self, frame: np.ndarray, target_size: Tuple[int, int] = (294, 518)) -> np.ndarray:
        """
        预处理帧
        
        Args:
            frame: 输入帧 (H, W, 3)
            target_size: 目标尺寸 (H, W) - 固定为(294, 518)以匹配DROID数据集比例
            
        Returns:
            预处理后的帧
        """
        # 调整尺寸
        frame_pil = Image.fromarray(frame)
        frame_resized = frame_pil.resize(target_size, Image.Resampling.BICUBIC)
        frame_array = np.array(frame_resized)
        
        # 归一化到[0, 1]
        frame_normalized = frame_array.astype(np.float32) / 255.0
        
        return frame_normalized

    def get_data(
        self,
        seq_index: int = None,
        img_per_seq: int = 2,  # 固定为2 (ext1 + ext2)
        seq_name: str = None,
        ids: list = None,
        aspect_ratio: float = 1.0,
    ) -> dict:
        """
        获取数据
        
        Args:
            seq_index: 序列索引（这里用作帧索引）
            img_per_seq: 每个序列的图像数量（固定为2）
            seq_name: 序列名称（未使用）
            ids: ID列表（未使用）
            aspect_ratio: 宽高比（未使用）
            
        Returns:
            数据字典
        """
        if seq_index is None:
            # 随机选择一帧
            seq_index = random.randint(0, self.total_frames - 1)
        
        # 确保索引在有效范围内
        seq_index = seq_index % self.total_frames
        
        # 获取对应帧
        ext1_frame = self.ext1_frames[seq_index]
        ext2_frame = self.ext2_frames[seq_index]
        wrist_frame = self.wrist_frames[seq_index]
        
        # 预处理帧
        target_size = (294, 518)  # 固定尺寸，匹配DROID数据集比例
        ext1_processed = self._preprocess_frame(ext1_frame, target_size)
        ext2_processed = self._preprocess_frame(ext2_frame, target_size)
        wrist_processed = self._preprocess_frame(wrist_frame, target_size)
        
        # 生成GT数据（zeros）
        height, width = target_size
        zeros_depth = self._generate_zeros_depth(height, width)
        zeros_point_cloud = self._generate_zeros_point_cloud(height, width)
        zeros_camera_params = self._generate_zeros_camera_params()
        
        # 构建数据字典
        data = {
            'images': np.stack([ext1_processed, ext2_processed], axis=0),  # (2, H, W, 3)
            'wrist_image': wrist_processed[np.newaxis, ...],  # (1, H, W, 3)
            'depth': zeros_depth[np.newaxis, ...],  # (1, H, W, 1)
            'world_points': zeros_point_cloud[np.newaxis, ...],  # (1, H*W, 3)
            # 🔥 添加训练器期望的字段
            'extrinsics': zeros_camera_params['extrinsic'][np.newaxis, ...],  # (1, 4, 4)
            'intrinsics': zeros_camera_params['intrinsic'][np.newaxis, ...],  # (1, 3, 3)
            'depths': zeros_depth[np.newaxis, ...],  # (1, H, W, 1) - 训练器期望的字段名
            'cam_points': zeros_point_cloud[np.newaxis, ...],  # (1, H*W, 3) - 相机坐标系点云
            'point_masks': np.ones((1, height * width), dtype=np.bool_),  # (1, H*W) - 点云掩码
            'camera_params': {
                'extrinsic': zeros_camera_params['extrinsic'][np.newaxis, ...],  # (1, 4, 4)
                'intrinsic': zeros_camera_params['intrinsic'][np.newaxis, ...],  # (1, 3, 3)
            },
            'frame_index': seq_index,
            'total_frames': self.total_frames,
            'video_info': self.video_info,
        }
        
        if self.debug:
            logging.debug(f"生成数据 - 帧索引: {seq_index}, 图像形状: {data['images'].shape}")
        
        return data

    def __len__(self):
        """返回数据集长度（总帧数）"""
        return self.total_frames