# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import os.path as osp
import glob
import logging
import pickle
import random
import numpy as np
import trimesh
import cv2
from PIL import Image

from data.dataset_util import *
from data.base_dataset import BaseDataset


class VggtDroidSingleViewDataset(BaseDataset):
    """
    VGGT DROID Dataset for single-view training with random selection of ext1 or ext2.
    
    This dataset implements:
    1. Loads point clouds, camera parameters, RGB images, and depth maps
    2. Transforms ext1 to identity matrix (world coordinate system)
    3. Randomly selects either ext1 or ext2 as the single input camera
    4. Predicts wrist camera extrinsics as additional supervision
    """
    
    def __init__(
        self,
        common_config,  # 修改为common_config以匹配DynamicTorchDataset的调用
        split: str = "train",
        DROID_DIR: str = None,
        min_num_frames: int = 1,
        len_train: int = 100000,
        len_test: int = 10000,
        enable_wrist_prediction: bool = True,
        single_view_training: bool = True,
        random_view_selection: bool = True,
        fixed_view: str = None,  # "ext1" 或 "ext2"，null表示随机选择
    ):
        """
        Initialize the VggtDroidSingleViewDataset.

        Args:
            common_conf: Configuration object with common settings.
            split (str): Dataset split, either 'train' or 'test'.
            DROID_DIR (str): Directory path to DROID data (./data_vggt).
            min_num_frames (int): Minimum number of frames per sequence.
            len_train (int): Length of the training dataset.
            len_test (int): Length of the test dataset.
            enable_wrist_prediction (bool): Whether to enable wrist pose prediction.
            single_view_training (bool): Whether to enable single view training.
            random_view_selection (bool): Whether to randomly select view.
            fixed_view (str): Fixed view to use if random_view_selection is False.
        """
        super().__init__(common_conf=common_config)

        self.debug = common_config.debug
        self.training = common_config.training
        self.get_nearby = common_config.get_nearby
        self.load_depth = common_config.load_depth
        self.inside_random = common_config.inside_random
        self.allow_duplicate_img = common_config.allow_duplicate_img
        self.enable_wrist_prediction = enable_wrist_prediction
        
        # 🎯 单视角训练配置
        self.single_view_training = single_view_training
        self.random_view_selection = random_view_selection
        self.fixed_view = fixed_view
        
        if DROID_DIR is None:
            DROID_DIR = "./data_vggt"
            
        self.DROID_DIR = DROID_DIR
        self.min_num_frames = min_num_frames

        # Discover all sequences in the data directory
        self.sequences = self._discover_sequences()
        
        if split == "train":
            # Use 80% for training
            split_idx = int(0.8 * len(self.sequences))
            self.sequences = self.sequences[:split_idx]
        elif split == "test":
            # Use 20% for testing
            split_idx = int(0.8 * len(self.sequences))
            self.sequences = self.sequences[split_idx:]
        else:
            raise ValueError(f"Invalid split: {split}")

        self.sequence_list_len = len(self.sequences)
        
        # 计算真实的样本数量 - 所有序列中的总帧数
        self.len_train = self._calculate_total_frames()
        
        status = "Training" if self.training else "Test"
        logging.info(f"{status}: VGGT-DROID Single View Data sequences: {self.sequence_list_len}")
        logging.info(f"{status}: VGGT-DROID Single View Dataset length: {self.len_train}")
        logging.info(f"{status}: Single view training: {self.single_view_training}")
        logging.info(f"{status}: Random view selection: {self.random_view_selection}")
        if not self.random_view_selection:
            logging.info(f"{status}: Fixed view: {self.fixed_view}")

    def _discover_sequences(self):
        """Discover all valid sequences in the data directory."""
        sequences = []
        
        if not osp.exists(self.DROID_DIR):
            logging.warning(f"DROID_DIR does not exist: {self.DROID_DIR}")
            return []
        
        # 🎯 修复：使用与多视角dataset相同的数据发现逻辑
        pattern = osp.join(self.DROID_DIR, "*")
        seq_dirs = glob.glob(pattern)
        for seq_dir in seq_dirs:
            if not osp.isdir(seq_dir):
                continue
                
            seq_name = osp.basename(seq_dir)
            
            # 🎯 修复：使用camera_parameter文件来发现帧
            frame_pattern = osp.join(seq_dir, "camera_parameter_*.pkl")
            frame_files = glob.glob(frame_pattern)
            frame_ids = []
            for frame_file in frame_files:
                frame_id = osp.basename(frame_file).replace("camera_parameter_", "").replace(".pkl", "")
                
                if self._check_frame_files(seq_dir, frame_id, verbose=True):
                    frame_ids.append(frame_id)
                
            if len(frame_ids) >= self.min_num_frames:
                sequences.append({
                    'seq_dir': seq_dir,
                    'seq_name': seq_name,
                    'frame_ids': frame_ids
                })
        
        logging.info(f"Discovered {len(sequences)} valid sequences")
        
        if len(sequences) == 0:
            logging.warning("No valid sequences found in DROID_DIR")
        
        return sequences
    
    def _check_frame_files(self, seq_dir, frame_id, verbose=False):
        """Check if all required files exist for a frame."""
        required_files = [
            f"camera_parameter_{frame_id}.pkl",
            # f"point_cloud_{frame_id}.glb",
            f"ext1_color_{frame_id}.png",
            f"ext2_color_{frame_id}.png",
        ]
        
        if self.load_depth:
            required_files.extend([
                f"ext1_depth_{frame_id}.pkl",
                f"ext2_depth_{frame_id}.pkl",
            ])
        
        for file_name in required_files:
            file_path = osp.join(seq_dir, file_name)
            if not osp.exists(file_path):
                if verbose:
                    logging.debug(f"Missing file: {file_path}")
                return False
        
        return True
    
    def _validate_camera_params(self, cam_params):
        """Validate camera parameters to prevent NaN/Inf issues."""
        # 验证外参
        for cam_name in ['ext1', 'ext2', 'wrist']:
            if cam_name in cam_params.get('extrinsics', {}):
                extri = np.array(cam_params['extrinsics'][cam_name], dtype=np.float32)
                if np.isnan(extri).any() or np.isinf(extri).any():
                    cam_params['extrinsics'][cam_name] = np.eye(4)[:3, :4].astype(np.float32)
                    
        # 验证内参
        for cam_name in ['ext1', 'ext2', 'wrist']:
            if cam_name in cam_params.get('intrinsics', {}):
                intri = np.array(cam_params['intrinsics'][cam_name], dtype=np.float32)
                if np.isnan(intri).any() or np.isinf(intri).any():
                    logging.warning(f"Found NaN/Inf in {cam_name} intrinsics")
                    
        return cam_params
    
    def _validate_and_clean_depth(self, depth_map):
        """
        Validate and clean depth map to prevent NaN/Inf issues.
        Returns both the cleaned depth map and a mask indicating original NaN/Inf locations.
        
        Returns:
            tuple: (cleaned_depth_map, nan_mask)
                - cleaned_depth_map: 深度图，NaN/Inf被设置为0
                - nan_mask: bool数组，True表示原始NaN/Inf位置
        """
        # 检查输入
        if depth_map is None:
            return None,None
        
        depth_map = np.array(depth_map, dtype=np.float32)
        
        # 创建NaN/Inf位置的mask - 这些位置在监督时会被忽略
        nan_inf_mask = np.isnan(depth_map) | np.isinf(depth_map)
        
        if nan_inf_mask.any():
            depth_map[nan_inf_mask] = 0.0
        
        return depth_map, nan_inf_mask
    
    def _generate_synthetic_depth(self):
        """Generate synthetic depth map for fallback."""
        return np.ones((294, 518), dtype=np.float32) * 0.1
    
    def _calculate_total_frames(self):
        """Calculate total number of frames across all sequences."""
        total_frames = 0
        for sequence in self.sequences:
            total_frames += len(sequence['frame_ids'])
        return total_frames

    def _global_to_seq_frame_index(self, global_idx):
        """Convert global sample index to sequence and frame index."""
        remaining_idx = global_idx
        for seq_idx, sequence in enumerate(self.sequences):
            if remaining_idx < len(sequence['frame_ids']):
                return seq_idx, remaining_idx
            remaining_idx -= len(sequence['frame_ids'])
        # Fallback to last sequence, last frame
        return len(self.sequences) - 1, len(self.sequences[-1]['frame_ids']) - 1

    def _transform_to_selected_view_coordinate_system(self, points_world, cam_params, selected_view):
        """
        🎯 变换到选中视角的坐标系系统
        选中的视角（ext1或ext2）成为原点（单位矩阵）
        """
        # Get camera extrinsics
        ext1_w2c_4x4 = np.eye(4)
        ext1_w2c_4x4[:, :] = cam_params['extrinsics']['ext1']  # ext1 world-to-camera
        
        ext2_w2c_4x4 = np.eye(4)
        ext2_w2c_4x4[:, :] = cam_params['extrinsics']['ext2']  # ext2 world-to-camera
        
        wrist_w2c_4x4 = np.eye(4)
        wrist_w2c_4x4[:, :] = cam_params['extrinsics']['wrist']  # wrist world-to-camera
        
        # 根据选中的视角确定参考坐标系
        if selected_view == "ext1":
            # ext1作为参考坐标系
            reference_w2c_4x4 = ext1_w2c_4x4
            reference_c2w = np.linalg.inv(ext1_w2c_4x4)  # ext1 camera-to-world
        elif selected_view == "ext2":
            # ext2作为参考坐标系
            reference_w2c_4x4 = ext2_w2c_4x4
            reference_c2w = np.linalg.inv(ext2_w2c_4x4)  # ext2 camera-to-world
        else:
            raise ValueError(f"Unknown selected view: {selected_view}")
        if points_world is not None:
            # Transform all points to the selected view's coordinate system
            points_homogeneous = np.concatenate([points_world, np.ones((points_world.shape[0], 1))], axis=1)
            points_new_world = (reference_c2w @ points_homogeneous.T).T[:, :3]
        else:
            points_new_world = None
        # Transform camera parameters
        ext1_new_4x4 = ext1_w2c_4x4 @ reference_c2w
        ext2_new_4x4 = ext2_w2c_4x4 @ reference_c2w
        wrist_new_4x4 = wrist_w2c_4x4 @ reference_c2w
        
        new_cam_params = {
            'intrinsics': cam_params['intrinsics'].copy(),
            'extrinsics': {
                'ext1': ext1_new_4x4[:3, :],  # 从新世界坐标到ext1的变换
                'ext2': ext2_new_4x4[:3, :],  # 从新世界坐标到ext2的变换
                'wrist': wrist_new_4x4[:3, :]  # 从新世界坐标到wrist的变换
            }
        }
        
        # 🎯 关键：将选中的视角的外参设置为单位矩阵
        if selected_view == "ext1":
            # 确认ext1的外参为单位矩阵（3x4），即前三行为[1,0,0,0],[0,1,0,0],[0,0,1,0]
            assert np.allclose(new_cam_params['extrinsics']['ext1'], np.eye(4)[:3, :]), \
                f"ext1外参不是单位矩阵，当前值: {new_cam_params['extrinsics']['ext1']}"
        elif selected_view == "ext2":
            assert np.allclose(new_cam_params['extrinsics']['ext2'], np.eye(4)[:3, :]), \
                f"ext1外参不是单位矩阵，当前值: {new_cam_params['extrinsics']['ext2']}"
        
        return points_new_world, new_cam_params

    def _load_camera_parameters(self, seq_dir, frame_id):
        """Load camera parameters for a specific frame."""
        param_file = osp.join(seq_dir, f"camera_parameter_{frame_id}.pkl")
        try:
            with open(param_file, 'rb') as f:
                cam_param = pickle.load(f)
            
            # 🎯 验证和清理相机参数（与多视角dataset保持一致）
            cam_param = self._validate_camera_params(cam_param)
            return cam_param
            
        except Exception as e:
            logging.warning(f"Failed to load camera parameters {param_file}: {e}. Using defaults.")
            # 返回默认的相机参数
            return None

    def _load_point_cloud(self, seq_dir, frame_id):
        """Load and process point cloud."""
        
        pcd_file = osp.join(seq_dir, f"point_cloud_{frame_id}.glb")
        if not os.path.exists(pcd_file):
            # logging.warning(f"Point cloud file not found: {pcd_file}")
            return None, None
        try:
            scene = trimesh.load(pcd_file)
            pcd = list(scene.geometry.values())[0]
            
            points_world = pcd.vertices  # (N, 3)
            colors_raw = pcd.colors[:, :3]  # RGBA -> RGB
            
            # Apply color enhancement
            colors_enhanced = self._enhance_colors(colors_raw)
            
            return points_world, colors_enhanced
        except Exception as e:
            logging.warning(f"Failed to load point cloud {pcd_file}: {e}")
            return None, None

    def _load_depth_map(self, seq_dir, frame_id, camera_name):
        """Load and validate depth map."""
        if not self.load_depth:
            return None,None
            
        depth_file = osp.join(seq_dir, f"{camera_name}_depth_{frame_id}.pkl")
        try:
            with open(depth_file, 'rb') as f:
                depth_map = pickle.load(f)
            
            # 🎯 严格的数据验证和清理（与多视角dataset保持一致）
            depth_map, nan_mask = self._validate_and_clean_depth(depth_map)
            return depth_map, nan_mask
            
        except Exception as e:
            logging.warning(f"Failed to load depth map {depth_file}: {e}. Using synthetic depth.")
            synthetic_depth = self._generate_synthetic_depth()
            return synthetic_depth, np.zeros_like(synthetic_depth, dtype=bool)

    def _enhance_colors(self, colors, brightness_boost=30, min_value=10):
        """
        Enhanced color processing for better visibility.
        
        Reduced brightness boost to avoid over-exposure.
        """
        # Check if colors are already normalized to [0,1]
        if colors.max() <= 1.0:
            colors = (colors * 255).astype(np.uint8)
        
        colors_enhanced = colors.astype(np.float32)
        
        # Adaptive enhancement based on scene brightness
        mean_brightness = np.mean(colors_enhanced)
        if mean_brightness < 100:  # Only enhance dark scenes
            # Gentle enhancement: gamma correction + slight boost
            colors_enhanced = np.power(colors_enhanced / 255.0, 0.8) * 255.0  # Gamma correction
            colors_enhanced = colors_enhanced + brightness_boost
        
        colors_enhanced = np.clip(colors_enhanced, min_value, 255)
        return colors_enhanced.astype(np.uint8)

    def _select_single_view(self):
        """
        🎯 选择单个视角：随机选择ext1或ext2，或使用固定视角
        返回选中的具体视角标识（ext1或ext2）
        """
        # if not self.random_view_selection and self.fixed_view:
        #     # 使用固定视角
        #     selected_view = self.fixed_view
        #     logging.debug(f"Using fixed view: {selected_view}")
        # else:
        #     # 随机选择视角
        selected_view = random.choice(['ext1', 'ext2'])
        logging.debug(f"Randomly selected view: {selected_view}")
        
        return selected_view

    def get_data(
        self,
        seq_index: int = None,
        img_per_seq: int = 1,  # 🎯 修改为1：只使用1个ext相机
        seq_name: str = None,
        ids: list = None,
        aspect_ratio: float = 1.0,
    ) -> dict:
        """
        Retrieve data for a specific sequence or frame.

        Args:
            seq_index (int): Global sample index or sequence index.
            img_per_seq (int): Number of images per sequence (should be 1 for single view).
            seq_name (str): Name of the sequence.
            ids (list): Specific frame IDs to retrieve.
            aspect_ratio (float): Aspect ratio for image processing.

        Returns:
            dict: A batch of data including images, depths, points, and camera parameters.
        """
        if self.inside_random:
            # 随机选择一个全局样本索引
            global_sample_idx = random.randint(0, self.len_train - 1)
            seq_idx, frame_idx = self._global_to_seq_frame_index(global_sample_idx)
        elif seq_index is not None:
            if seq_index >= self.len_train:
                # 如果超出范围，使用模运算
                seq_index = seq_index % self.len_train
            seq_idx, frame_idx = self._global_to_seq_frame_index(seq_index)
        else:
            # 默认使用第一个序列的第一帧
            seq_idx, frame_idx = 0, 0
            
        sequence = self.sequences[seq_idx]
        seq_dir = sequence['seq_dir']
        
        if seq_name is None:
            seq_name = sequence['seq_name']
            
        available_frames = sequence['frame_ids']
        
        if ids is None:
            # 使用计算出的帧索引，但仍然支持多帧
            if img_per_seq == 1:
                # 只要一帧，使用计算出的特定帧
                ids = [frame_idx]
            else:
                # 需要多帧，以计算出的帧为中心选择
                if len(available_frames) < img_per_seq:
                    # 帧数不够，使用重复
                    ids = np.random.choice(len(available_frames), img_per_seq, replace=True)
                else:
                    # 以frame_idx为中心选择相邻帧
                    start_idx = max(0, frame_idx - img_per_seq // 2)
                    end_idx = min(len(available_frames), start_idx + img_per_seq)
                    if end_idx - start_idx < img_per_seq:
                        start_idx = max(0, end_idx - img_per_seq)
                    ids = list(range(start_idx, start_idx + img_per_seq))
        
        frame_ids = [available_frames[i] for i in ids]
        
        # We'll process the first frame as the main frame
        main_frame_id = frame_ids[0]
        
        # Load camera parameters and point cloud
        cam_params = self._load_camera_parameters(seq_dir, main_frame_id)
        if cam_params is None:
            logging.warning(f"Failed to load camera parameters for frame {main_frame_id}, skipping")
            return self.get_data(seq_index=(seq_index+1) if seq_index else 1, img_per_seq=img_per_seq, aspect_ratio=aspect_ratio)
        
        if self.load_depth:
            points_world_orig, colors_enhanced = self._load_point_cloud(seq_dir, main_frame_id)
        else:
            points_world_orig = None
            colors_enhanced = None
        
        # 🎯 关键修改：只处理一个ext相机
        selected_view = self._select_single_view()
        cameras = [selected_view]  # 只使用选中的相机
        
        # 🎯 转换到选中视角的坐标系（选中的视角作为单位矩阵）
        points_new_world, new_cam_params = self._transform_to_selected_view_coordinate_system(
            points_world_orig, cam_params, selected_view
        )
        
        target_image_shape = self.get_target_shape(aspect_ratio)
        
        images = []
        depths = []
        depth_nan_masks = []  # 新增：存储深度图的NaN mask
        extrinsics = []
        intrinsics = []
        cam_points = []
        world_points = []
        point_masks = []
        image_paths = []
        original_sizes = []
        
        for camera_name in cameras:
            # Load RGB image
            image_path = osp.join(seq_dir, f"{camera_name}_color_{main_frame_id}.png")
            image = read_image_cv2(image_path)
            
            # Load depth map with NaN mask
            if self.load_depth:
                depth_map, nan_mask = self._load_depth_map(seq_dir, main_frame_id, camera_name)
            else:
                depth_map = None
                nan_mask = None
            
            # Get camera parameters
            extri_opencv = new_cam_params['extrinsics'][camera_name] # 都是world2camera
            intri_opencv = np.array(new_cam_params['intrinsics'][camera_name])
            
            original_size = np.array(image.shape[:2])
            
            # 使用BaseDataset的标准process_one_image方法来处理图像和内参
            # 这样可以正确处理aspect_ratio、landscape_check、actual_resize_scale等
            (
                processed_image,
                processed_depth,
                processed_extri,
                processed_intri,
                world_coords_points,
                cam_coords_points,
                point_mask,
                track,
            ) = self.process_one_image(
                image=image,
                depth_map=depth_map,
                extri_opencv=extri_opencv,
                intri_opencv=intri_opencv,
                original_size=original_size,
                target_image_shape=target_image_shape,
                track=None,
                filepath=image_path,
                safe_bound=4,
            )
            if processed_depth is not None:
            # 更新NaN mask以适应处理后的图像
                processed_nan_mask = np.isnan(processed_depth) | np.isinf(processed_depth) | (processed_depth <= 0.01)
                depths.append(processed_depth)
                depth_nan_masks.append(processed_nan_mask)
                world_points.append(world_coords_points)
                point_masks.append(point_mask)
            else:
                processed_nan_mask = None
                depths=None
                depth_nan_masks=None
                world_points = None
                point_masks = None
            images.append(processed_image)
            extrinsics.append(processed_extri)
            intrinsics.append(processed_intri)
            cam_points.append(cam_coords_points)
            image_paths.append(image_path)
            original_sizes.append(original_size)
        
        # 🎯 关键修改：单视角训练只输出一个"ext"视角
        # 不复制视角，而是重新组织数据结构
        batch = {
            "seq_name": f"vggt_droid_single_view_{seq_name}",
            "ids": ids,
            "frame_num": 1,  # 单视角只有1个相机
            "images": images,  # 只包含选中的视角
            "depths": depths,
            "depth_nan_masks": depth_nan_masks,
            "extrinsics": extrinsics,
            "intrinsics": intrinsics,
            "cam_points": cam_points,
            "world_points": world_points,
            "point_masks": point_masks,
            "original_sizes": original_sizes,
            
            # Additional VGGT-DROID specific data
            "point_cloud": points_new_world,  # Transformed point cloud
            "point_colors": colors_enhanced,   # Enhanced colors
            "frame_id": main_frame_id,
            
            # 🎯 新增：单视角训练信息
            "single_view_training": True,
            "ext_view": selected_view,  # 对外统一的"ext"视角标识
        }
        if batch['depths'] is None:
            batch.pop('depths')
            batch.pop('depth_nan_masks')
            batch.pop('world_points')
            batch.pop('point_masks')
            batch.pop("point_cloud")
            batch.pop("point_colors")
            batch.pop("cam_points")
        # Add wrist camera parameters if enabled
        # 对intrinsics的ext1、ext2、wrist三个进行缩放，使[0,2]分量变为518/2
        intri = np.array(new_cam_params['intrinsics']['wrist'], dtype=np.float32)
        # 计算当前主对角线像素（通常为fx, fy），目标为518/2=259
        scale = 259.0 / intri[0, 2]
        # 整体缩放内参矩阵
        intri_scaled = intri.copy()
        intri_scaled[0, :] *= scale
        intri_scaled[1, :] *= scale
        new_cam_params['intrinsics']['wrist'] = intri_scaled
        if self.enable_wrist_prediction:
            wrist_extri = new_cam_params['extrinsics']['wrist']
            wrist_intri = np.array(new_cam_params['intrinsics']['wrist'])
            
            batch.update({
                "wrist_extrinsics": wrist_extri,  # Target for wrist pose prediction
                "wrist_intrinsics": wrist_intri,  # Wrist intrinsics (not predicted)
            })
        
        # === 新增：加载wrist_color图像（不作为模型输入） ===
        wrist_color_path = osp.join(seq_dir, f"wrist_color_{main_frame_id}.png")
        if osp.exists(wrist_color_path):
            try:
                # 加载wrist RGB图像
                wrist_image = read_image_cv2(wrist_color_path)
                
                # 增强颜色（与ext1/ext2保持一致）
                wrist_image_enhanced = self._enhance_colors(wrist_image)
                # 和ext1的shape对齐
                ext1_shape = images[0].shape[:2]  # (H, W)
                wrist_image_resized = cv2.resize(
                    wrist_image_enhanced, 
                    (ext1_shape[1], ext1_shape[0]), 
                    interpolation=cv2.INTER_LINEAR
                )
                # 保存到batch中，但不作为模型输入
                batch.update({
                    "wrist_image": wrist_image_resized,  # 原始尺寸的wrist图像
                    "wrist_image_path": wrist_color_path,
                    "wrist_image_shape": wrist_image_resized.shape[:2],  # (H, W)
                })
                
                
            except Exception as e:
                logging.warning(f"⚠️ 加载wrist_color图像失败: {e}")
                batch.update({
                    "wrist_image": None,
                    "wrist_image_path": None,
                    "wrist_image_shape": None,
                })
        else:
            logging.warning(f"⚠️ wrist_color图像不存在: {wrist_color_path}")
            batch.update({
                "wrist_image": None,
                "wrist_image_path": None,
                "wrist_image_shape": None,
            })
        
        return batch 