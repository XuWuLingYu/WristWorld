#!/usr/bin/env python3
"""
VGGT video inference script (Droid, dual-view).

Inputs:
- Two RGB videos (ext1, ext2)
- One GT wrist video
- Optional: single-view mode via --single-view and --view

Outputs:
- Copies of input videos
- A folder of per-frame point cloud GLB files (a red sphere can be added at the wrist origin if desired)
- A projection video (point cloud projected to the inferred wrist pose, no red sphere)

Modes:
- Dual-view (default): use ext1 and ext2
- Single-view (optional): use --single-view with --view to select ext1 or ext2

Image sizes:
- VGGT input uses two views with image_size (W, H) = (518, 294). Functions that take (H, W) should receive (294, 518).
"""

import os
import sys
import argparse
import numpy as np
import cv2
from PIL import Image
import torch
import torch.nn.functional as F
import trimesh
import matplotlib.pyplot as plt
from pathlib import Path
import json
from typing import Tuple, List, Optional
import shutil
from tqdm import tqdm

# Add VGGT paths if needed
sys.path.append('../stage1_reconstruction/')
sys.path.append('../stage1_reconstruction/vggt')
sys.path.append('../stage1_reconstruction/training')

try:
    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    from vggt.utils.geometry import unproject_depth_map_to_point_map
    from visual_util import predictions_to_glb
    # Import helpers from validation_visualizer for GLB/projection utilities
    from validation_visualizer import ValidationVisualizer
except ImportError as e:
    print(f"Failed to import VGGT modules: {e}")
    print("Please ensure the script runs in the correct environment and PYTHONPATH is set.")
    sys.exit(1)


class VGGTVideoInference:
    def __init__(self, checkpoint_path: str, device: str = "cuda"):
        """
        Initialize VGGT video inference.

        Args:
            checkpoint_path: model checkpoint path
            device: inference device
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.checkpoint_path = checkpoint_path
        self.model = self._load_model()
        
        # Create ValidationVisualizer instance to reuse its utilities
        self.val_visualizer = ValidationVisualizer(output_base_dir="vggt_video_output")
        
        # Output directories
        self.output_dir = Path("vggt_video_output")
        self.output_dir.mkdir(exist_ok=True)
        self.pointclouds_dir = self.output_dir / "pointclouds"
        self.pointclouds_dir.mkdir(exist_ok=True)
        self.videos_dir = self.output_dir / "videos"
        self.videos_dir.mkdir(exist_ok=True)
        
        print(f"VGGT video inference initialized. Device: {self.device}")
    
    def _load_model(self) -> VGGT:
        """
        Load VGGT model (aligned with training config).

        Returns:
            Loaded VGGT model
        """
        print(f"Loading model checkpoint: {self.checkpoint_path}")
        
        # Create model instance (match training config)
        model = VGGT(
            img_size=518,
            patch_size=14,
            embed_dim=1024,
            enable_camera=True,      # training config: enable_camera=True
            enable_depth=True,       # training config: enable_depth=True
            enable_point=True,       # enable point head for world_points
            enable_track=False,      # keep track head disabled
            enable_wrist=True,       # enable wrist functionality for projection
            pretrained="facebook/VGGT-1B",
            use_lora=False,          # do not use LoRA
            lora_rank=16,
            lora_alpha=32
        )
        
        # Disable gradients for track_head to match training config
        for name, param in model.named_parameters():
            if "track_head" in name:
                param.requires_grad = False
                print(f"Disabled gradients for: {name}")
        
        # Load checkpoint
        try:
            checkpoint = torch.load(self.checkpoint_path, map_location="cpu")
            
            if "model" in checkpoint:
                model_state_dict = checkpoint["model"]
            else:
                model_state_dict = checkpoint
                
            missing_keys, unexpected_keys = model.load_state_dict(model_state_dict, strict=False)
            
            if missing_keys:
                print(f"Missing keys: {missing_keys}")
            if unexpected_keys:
                print(f"Unexpected keys: {unexpected_keys}")
                
            print("Model state_dict loaded.")
            
        except Exception as e:
            print(f"Failed to load checkpoint: {e}")
            print("Trying to load pretrained model...")
            try:
                model = VGGT.from_pretrained("facebook/VGGT-1B")
                print("Pretrained model loaded.")
            except Exception as e2:
                print(f"Failed to load pretrained model: {e2}")
                sys.exit(1)
        
        model.eval()
        model.to(self.device)
        return model
    
    def extract_video_frames(self, video_path: str) -> List[np.ndarray]:
        """
        Extract video frames.

        Args:
            video_path: path to the video file

        Returns:
            List of RGB frames (numpy arrays)
        """
        print(f"Extracting frames: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {video_path}")
        
        frames = []
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        with tqdm(total=frame_count, desc="extract") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
                pbar.update(1)
        
        cap.release()
        print(f"Extracted frames: {len(frames)}")
        return frames
    
    def get_video_info(self, video_path: str) -> dict:
        """
        Get basic video info.

        Args:
            video_path: path to the video file

        Returns:
            Dict with fps/frame_count/width/height
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {video_path}")
        
        info = {
            'fps': 30,
            'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        }
        
        cap.release()
        return info
    
    def preprocess_frame_pair(self, frame1: np.ndarray, frame2: np.ndarray = None, single_view: bool = False) -> torch.Tensor:
        """
        Preprocess a frame pair. Uses a temp dir to avoid conflicts.

        Args:
            frame1: first frame
            frame2: second frame (None in single-view mode)
            single_view: whether to run single-view
        """
        import tempfile, shutil
        temp_dir = Path(tempfile.mkdtemp(prefix=f"vggt_temp_{os.getpid()}_", dir=str(Path.cwd())))
        try:
            frame1_path = temp_dir / "frame1.jpg"
            Image.fromarray(frame1).save(frame1_path)
            
            if single_view:
                # Single-view: only one camera. Model expects [B, S, 3, H, W] with S=1
                try:
                    images = load_and_preprocess_images([str(frame1_path)])
                except ImportError:
                    images = self._simple_preprocess_images([str(frame1_path)])
            else:
                # Dual-view: two cameras
                frame2_path = temp_dir / "frame2.jpg"
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

    def preprocess_frames_multi(self, frames: List[np.ndarray]) -> torch.Tensor:
        """
        Preprocess a multi-view frame list (S >= 1).

        Returns:
            Tensor with shape [S, 3, H, W]
        """
        import tempfile, shutil
        from PIL import Image
        temp_dir = Path(tempfile.mkdtemp(prefix=f"vggt_temp_multi_{os.getpid()}_", dir=str(Path.cwd())))
        saved_paths: List[str] = []
        try:
            for i, frame in enumerate(frames):
                img_path = temp_dir / f"frame_{i:02d}.jpg"
                Image.fromarray(frame).save(img_path)
                saved_paths.append(str(img_path))
            try:
                images = load_and_preprocess_images(saved_paths)
            except ImportError:
                images = self._simple_preprocess_images(saved_paths)
            return images.to(self.device)
        finally:
            try:
                shutil.rmtree(temp_dir, ignore_errors=True)
            except Exception:
                pass
    
    def _simple_preprocess_images(self, image_paths: List[str]) -> torch.Tensor:
        """
        Simplified image preprocessing. VGGT uses (W, H) = (518, 294) for Droid dual-view.
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
            
            img = img.resize((518, 294), Image.Resampling.BICUBIC)
            img_tensor = to_tensor(img)
            images.append(img_tensor)
        
        return torch.stack(images)
    
    def run_inference_on_frames(self, frame1: np.ndarray, frame2: np.ndarray = None, single_view: bool = False) -> dict:
        """
        Run inference on a frame pair.

        Args:
            frame1: first frame
            frame2: second frame (None in single-view mode)
            single_view: whether single-view

        Returns:
            Predictions dict
        """
        # 预处理
        images = self.preprocess_frame_pair(frame1, frame2, single_view)
        images = images.unsqueeze(0)
        # 推理
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=dtype):
                predictions = self.model(images, only_wrist=True)
        
        # Convert pose encoding to extrinsic/intrinsic matrices
        if "pose_enc" in predictions:
            extrinsic, intrinsic = pose_encoding_to_extri_intri(
                predictions["pose_enc"],
                image_size_hw=(294, 518),  # Droid dual-view: (H=294, W=518)
                build_intrinsics=True
            )
            predictions["extrinsic"] = extrinsic
            predictions["intrinsic"] = intrinsic
        
        # Convert wrist pose encoding if present
        if "wrist_pose_enc" in predictions:
            wrist_extrinsic, wrist_intrinsic = pose_encoding_to_extri_intri(
                predictions["wrist_pose_enc"],
                image_size_hw=(294, 518),  # Droid dual-view: (H=294, W=518)
                build_intrinsics=True
            )
            predictions["wrist_extrinsic"] = wrist_extrinsic
            predictions["wrist_intrinsic"] = wrist_intrinsic
        
        # 添加原始images
        predictions["images"] = images.cpu().numpy()
        
        # Convert tensors to numpy
        for key in predictions.keys():
            if isinstance(predictions[key], torch.Tensor):
                predictions[key] = predictions[key].cpu().numpy()
                # if predictions[key].ndim > 0 and predictions[key].shape[0] == 1:
                #     predictions[key] = predictions[key].squeeze(0)
        
        return predictions

    def run_inference_on_frames_multi(self, frames: List[np.ndarray]) -> dict:
        """
        Run inference on multi-view frames (S >= 1).

        Args:
            frames: list of frames [ext1, ext2, ...]

        Returns:
            Predictions including wrist_extrinsic/intrinsic
        """
        if frames is None or len(frames) == 0:
            raise ValueError("frames 不能为空")

        images = self.preprocess_frames_multi(frames)  # [S, 3, H, W]
        images = images.unsqueeze(0)  # [1, S, 3, H, W]

        dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=dtype):
                predictions = self.model(images)

        # Convert pose encoding to extrinsic/intrinsic
        if "pose_enc" in predictions:
            extrinsic, intrinsic = pose_encoding_to_extri_intri(
                predictions["pose_enc"], image_size_hw=(294, 518), build_intrinsics=True
            )
            predictions["extrinsic"] = extrinsic
            predictions["intrinsic"] = intrinsic

        if "wrist_pose_enc" in predictions:
            wrist_extrinsic, wrist_intrinsic = pose_encoding_to_extri_intri(
                predictions["wrist_pose_enc"], image_size_hw=(294, 518), build_intrinsics=True
            )
            predictions["wrist_extrinsic"] = wrist_extrinsic
            predictions["wrist_intrinsic"] = wrist_intrinsic

        predictions["images"] = images.cpu().numpy()

        for key in list(predictions.keys()):
            if isinstance(predictions[key], torch.Tensor):
                predictions[key] = predictions[key].cpu().numpy()

        return predictions
    
    def generate_point_cloud_with_wrist_sphere(self, predictions: dict, single_view: bool = False, view_type: str = "ext1") -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate point cloud with a red sphere at wrist origin.

        Args:
            predictions: inference results
            single_view: whether single-view mode
            view_type: which view in single-view ("ext1" or "ext2")

        Returns:
            points and colors arrays
        """
        
        if "world_points" not in predictions:
            raise ValueError("'world_points' missing in predictions")
        
        # Use predicted point cloud
        points_3d = predictions["world_points"]  # use predicted point cloud
        if torch.is_tensor(points_3d):
            points_3d = points_3d.cpu().numpy()
        
        # Get sample images
        images_sample = predictions["images"][0]  # remove batch dimension
        
        if single_view:
            # Single-view mode: images_sample shape is (1, 3, 294, 518)
            # Use the first (and only) view
            rgb_image = images_sample[0]  # (3, 294, 518)
            
            if torch.is_tensor(rgb_image):
                rgb_image = rgb_image.cpu().numpy()
            
            # Transpose to (H, W, C)
            rgb_image = np.transpose(rgb_image, (1, 2, 0))  # (294, 518, 3)
            
            # Flatten to colors array
            colors = rgb_image.reshape(-1, 3)  # (294*518, 3)
        else:
            # Dual-view mode: images_sample shape is (2, 3, 294, 518)
            # Build colors manually from raw RGB images
            # Get ext1 and ext2 RGB images
            ext1_rgb = images_sample[0]  # (3, 294, 518)
            ext2_rgb = images_sample[1]  # (3, 294, 518)
            
            if torch.is_tensor(ext1_rgb):
                ext1_rgb = ext1_rgb.cpu().numpy()
            if torch.is_tensor(ext2_rgb):
                ext2_rgb = ext2_rgb.cpu().numpy()
            
            # Transpose to (H, W, C)
            ext1_rgb = np.transpose(ext1_rgb, (1, 2, 0))  # (294, 518, 3)
            ext2_rgb = np.transpose(ext2_rgb, (1, 2, 0))  # (294, 518, 3)
            
            # Concatenate colors from both cameras
            colors = np.concatenate([ext1_rgb.reshape(-1, 3), ext2_rgb.reshape(-1, 3)], axis=0)  # (2*294*518, 3)
        
        # Ensure point and color counts match
        points_3d = points_3d.reshape(-1, 3)
        assert len(points_3d) == len(colors), f"Point/color count mismatch: {len(points_3d)} vs {len(colors)}"
        
        # Call visualizer to add wrist-origin sphere
        # Create a copy with wrist_pose_enc if needed
        predictions_with_pose_enc = predictions.copy()
        if "wrist_extrinsic" in predictions and "wrist_intrinsic" in predictions:
            # Convert extrinsic/intrinsic back to pose_enc format for the visualizer
            wrist_extrinsic = predictions["wrist_extrinsic"]
            wrist_intrinsic = predictions["wrist_intrinsic"]
            
            # Handle batch dimension
            if wrist_extrinsic.ndim == 3:
                wrist_extrinsic = wrist_extrinsic[0]
            if wrist_intrinsic.ndim == 3:
                wrist_intrinsic = wrist_intrinsic[0]
            
            # Convert to pose_enc format
            from vggt.utils.pose_enc import extri_intri_to_pose_encoding
            # Ensure inputs are torch.Tensor
            if isinstance(wrist_extrinsic, np.ndarray):
                wrist_extrinsic = torch.from_numpy(wrist_extrinsic)
            if isinstance(wrist_intrinsic, np.ndarray):
                wrist_intrinsic = torch.from_numpy(wrist_intrinsic)
            
            wrist_pose_enc = extri_intri_to_pose_encoding(
                wrist_extrinsic.unsqueeze(0).unsqueeze(0),  # 添加batch和sequence维度
                wrist_intrinsic.unsqueeze(0).unsqueeze(0),
                image_size_hw=(294, 518)
            )
            predictions_with_pose_enc["wrist_pose_enc"] = wrist_pose_enc
        
        sphere_result = self.val_visualizer._add_wrist_origin_sphere(predictions_with_pose_enc, points_3d, colors)
        return sphere_result["points"], sphere_result["colors"]
    
    def generate_point_cloud_no_sphere(self, predictions: dict, single_view: bool = False, view_type: str = "ext1") -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate point cloud without the red sphere (for projection).

        Args:
            predictions: inference results
            single_view: whether single-view mode
            view_type: which single view ("ext1" or "ext2")

        Returns:
            points and colors arrays
        """
        if "world_points" not in predictions:
            raise ValueError("预测结果中未包含world_points")
        
        # Use predicted point cloud
        points_3d = predictions["world_points"]  # 使用预测的点云
        if torch.is_tensor(points_3d):
            points_3d = points_3d.cpu().numpy()
        
        # 获取单个样本的图像数据
        images_sample = predictions["images"][0]  # 移除batch维度
        
        S = images_sample.shape[0]
        if single_view or S == 1:
            rgb_image = images_sample[0]
            if torch.is_tensor(rgb_image):
                rgb_image = rgb_image.cpu().numpy()
            rgb_image = np.transpose(rgb_image, (1, 2, 0))
            colors = rgb_image.reshape(-1, 3)
        else:
            colors_list = []
            for s in range(S):
                rgb = images_sample[s]
                if torch.is_tensor(rgb):
                    rgb = rgb.cpu().numpy()
                rgb = np.transpose(rgb, (1, 2, 0))
                colors_list.append(rgb.reshape(-1, 3))
            colors = np.concatenate(colors_list, axis=0)
        
        # 确保点云和颜色数量匹配
        points_3d = points_3d.reshape(-1, 3)
        assert len(points_3d) == len(colors), f"点云和颜色数量不匹配: {len(points_3d)} vs {len(colors)}"
        
        return points_3d, colors
    
    def save_point_cloud_glb(self, points_3d: np.ndarray, colors: np.ndarray, filename: str):
        """
        Save point cloud as GLB file.

        Args:
            points_3d: (N, 3) points
            colors: (N, 3) colors (uint8)
            filename: output filename
        """
        # 🔥 直接调用validation_visualizer中的保存函数
        output_path = self.pointclouds_dir / filename
        self.val_visualizer._save_point_cloud_as_glb(points_3d, colors, str(output_path))
    
    def project_points_to_wrist_view(self, points_3d: np.ndarray, colors: np.ndarray,
                                   wrist_intrinsic: np.ndarray, wrist_extrinsic: np.ndarray,
                                   img_size: Tuple[int, int] = (294, 518)) -> np.ndarray:
        """
        Project point cloud to the wrist view.

        Args:
            points_3d: (N, 3) world coordinates
            colors: (N, 3) colors
            wrist_intrinsic: (3, 3)
            wrist_extrinsic: (3, 4) camera2world
            img_size: output (H, W), default (294, 518)

        Returns:
            RGB image (H, W, 3)
        """
        # 🔥 直接调用validation_visualizer中的投影函数
        # 兼容多种形状： (3,4)/(3,3) 或 (S,3,4)/(S,3,3) 或 (1,1,3,4)/(1,1,3,3)
        def _pick_matrix(mat: np.ndarray, expect_shape: Tuple[int, int]) -> np.ndarray:
            if mat.ndim == 2:
                if mat.shape != expect_shape:
                    raise ValueError(f"矩阵形状异常: {mat.shape} 期望 {expect_shape}")
                return mat
            if mat.ndim == 3:
                # 选择第一个视角
                return mat[0]
            if mat.ndim == 4:
                return mat[0, 0]
            raise ValueError(f"Unsupported matrix ndim: {mat.ndim}")

        extri = _pick_matrix(wrist_extrinsic, (3, 4))
        intri = _pick_matrix(wrist_intrinsic, (3, 3))

        return self.val_visualizer.visualize_point_cloud_projection(
            points_3d=points_3d,
            point_colors=colors,
            camera_extrinsics=extri,
            camera_intrinsics=intri,
            image_shape=img_size,
            need_inverse=False
        )
    
    def copy_input_videos(self, video_paths: List[str], names: List[str]):
        """
        Copy input videos to the output directory.

        Args:
            video_paths: list of paths
            names: list of names to save as
        """
        print("Copying input videos...")
        for video_path, name in zip(video_paths, names):
            dst_path = self.videos_dir / f"{name}.mp4"
            shutil.copy2(video_path, dst_path)
            print(f"Copied: {video_path} -> {dst_path}")
    
    def create_projection_video(self, projection_frames: List[np.ndarray], 
                              output_path: str, fps: float = 30.0):
        """
        Create a projection video from frames.

        Args:
            projection_frames: list of frames
            output_path: output path
            fps: frames per second
        """
        print(f"Creating projection video: {output_path}")
        
        if not projection_frames:
            print("No projection frames, skip writing video")
            return
        
        height, width = projection_frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        for frame in tqdm(projection_frames, desc="write frames"):
            # 转换RGB到BGR
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
        
        out.release()
        print(f"Projection video saved: {output_path}")
    
    def run_video_inference(self, ext1_video_path: str, ext2_video_path: str, 
                          wrist_video_path: str, single_view: bool = False, view_type: str = "ext1") -> dict:
        """
        Run the full video inference pipeline.

        Args:
            ext1_video_path: path to first input video
            ext2_video_path: path to second input video
            wrist_video_path: path to GT wrist video
            single_view: whether to run single-view
            view_type: which view in single-view ("ext1" or "ext2")

        Returns:
            Results summary dict
        """
        print("Starting video inference pipeline...")
        
        if single_view:
            print(f"Single-view mode: {view_type}")
        else:
            print("Dual-view mode: ext1 + ext2")
        
        # 1. 复制输入视频
        self.copy_input_videos(
            [ext1_video_path, ext2_video_path, wrist_video_path],
            ["ext1_input", "ext2_input", "wrist_gt"]
        )
        
        # 2. Extract video frames
        ext1_frames = self.extract_video_frames(ext1_video_path)
        ext2_frames = self.extract_video_frames(ext2_video_path)
        wrist_frames = self.extract_video_frames(wrist_video_path)
        
        # Check whether frame counts match
        frame_counts = [len(ext1_frames), len(ext2_frames), len(wrist_frames)]
        min_frames = min(frame_counts)
        if not all(count == min_frames for count in frame_counts):
            print(f"Frame count mismatch: {frame_counts}, using min frames: {min_frames}")
            ext1_frames = ext1_frames[:min_frames]
            ext2_frames = ext2_frames[:min_frames]
            wrist_frames = wrist_frames[:min_frames]
        
        # 获取视频信息
        video_info = self.get_video_info(ext1_video_path)
        
        # 3. Per-frame inference
        print(f"Running per-frame inference. Frames: {min_frames}")
        
        import time
        inference_start_time = time.time()
        visualization_start_time = None
        
        projection_frames = []
        
        for frame_idx in tqdm(range(min_frames), desc="推理进度"):
            # 推理阶段
            if single_view:
                # Single-view mode: only the specified view is used
                if view_type == "ext1":
                    predictions = self.run_inference_on_frames(
                        ext1_frames[frame_idx], 
                        None,  # second frame is None
                        single_view=True
                    )
                else:  # view_type == "ext2"
                    predictions = self.run_inference_on_frames(
                        ext2_frames[frame_idx], 
                        None,  # second frame is None
                        single_view=True
                    )
            else:
                # Dual-view mode: use two views
                predictions = self.run_inference_on_frames(
                    ext1_frames[frame_idx], 
                    ext2_frames[frame_idx],
                    single_view=False
                )
            
            # Visualization timing start
            if visualization_start_time is None:
                visualization_start_time = time.time()
            
            # Generate point cloud without wrist-origin sphere for projection
            points_no_sphere, colors_no_sphere = self.generate_point_cloud_no_sphere(
                predictions, single_view, view_type
            )
            
            # Project to wrist view using converted wrist params
            wrist_extrinsic = predictions["wrist_extrinsic"]
            wrist_intrinsic = predictions["wrist_intrinsic"]
            
            # Handle batched tensors if present
            if wrist_extrinsic.ndim == 3:
                wrist_extrinsic = wrist_extrinsic[0]
            if wrist_intrinsic.ndim == 3:
                wrist_intrinsic = wrist_intrinsic[0]
            
            projection_frame = self.project_points_to_wrist_view(
                points_no_sphere, colors_no_sphere,
                wrist_intrinsic, wrist_extrinsic,
                img_size=(294, 518)
            )
            
            # Stack GT and projection vertically. Fetch wrist GT frame.
            wrist_gt_frame = wrist_frames[frame_idx]
            
            # Resize GT to match projection
            wrist_gt_resized = cv2.resize(wrist_gt_frame, (518, 294))
            
            # Vertical stack: GT on top, projection at bottom
            combined_frame = np.vstack([wrist_gt_resized, projection_frame])
            
            projection_frames.append(combined_frame)
            
        
        # 4. 创建投影视频
        projection_video_path = str(self.videos_dir / "wrist_projection.mp4")
        self.create_projection_video(projection_frames, projection_video_path, 30.0)
        
        # 计算时间统计
        inference_end_time = time.time()
        inference_total_time = inference_end_time - inference_start_time
        visualization_total_time = inference_end_time - visualization_start_time
        
        print("Timing:")
        print(f"  Inference total: {inference_total_time:.2f}s")
        print(f"  Visualization total: {visualization_total_time:.2f}s")
        print(f"  Avg inference per frame: {inference_total_time/min_frames:.3f}s")
        print(f"  Avg visualization per frame: {visualization_total_time/min_frames:.3f}s")
        
        # 5. 生成结果统计
        results = {
            "input_videos": {
                "ext1": ext1_video_path,
                "ext2": ext2_video_path,
                "wrist_gt": wrist_video_path
            },
            "inference_mode": {
                "single_view": single_view,
                "view_type": view_type if single_view else "dual_view"
            },
            "output_files": {
                "videos_dir": str(self.videos_dir),
                "pointclouds_dir": str(self.pointclouds_dir),
                "projection_video": projection_video_path
            },
            "statistics": {
                "total_frames": min_frames,
                "fps": 30.0,  # 固定fps
                "video_resolution": f"{video_info['width']}x{video_info['height']}",
                "projection_resolution": "518x588",  # vertical concat: GT(294) + projection(294) = 588
                "glb_files_count": len(list(self.pointclouds_dir.glob("*.glb"))),
                "inference_time": inference_total_time,
                "visualization_time": visualization_total_time,
                "avg_inference_time_per_frame": inference_total_time/min_frames,
                "avg_visualization_time_per_frame": visualization_total_time/min_frames
            }
        }
        
        # Save summary
        summary_path = self.output_dir / "video_inference_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print("Video inference finished.")
        print(f"Output dir: {self.output_dir}")
        print(f"Input video copies: {self.videos_dir}")
        print(f"Point clouds: {self.pointclouds_dir} ({results['statistics']['glb_files_count']} GLB files)")
        print(f"Projection video: {projection_video_path} (GT + Projection vertical stack)")
        print(f"Summary: {summary_path}")
        
        return results


def main():
    parser = argparse.ArgumentParser(description="VGGT video inference (Droid, dual-view)")
    parser.add_argument("--ext1", required=True, help="Path to first input video")
    parser.add_argument("--ext2", required=True, help="Path to second input video")
    parser.add_argument("--wrist", required=True, help="Path to GT wrist video")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--device", default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--output", default="vggt_video_output", help="Output directory")
    parser.add_argument("--single-view", action="store_true", help="Enable single-view mode")
    parser.add_argument("--view", choices=["ext1", "ext2"], default="ext1", help="Single view type (ext1/ext2)")
    
    args = parser.parse_args()
    
    # Validate inputs
    for video_path, name in [(args.ext1, "ext1"), (args.ext2, "ext2"), (args.wrist, "wrist")]:
        if not os.path.exists(video_path):
            print(f"Video file does not exist: {video_path}")
            return
    
    if not os.path.exists(args.checkpoint):
        print(f"Checkpoint does not exist: {args.checkpoint}")
        return
    
    # Single-view parameter checks
    if args.single_view:
        if args.view not in ["ext1", "ext2"]:
            print(f"Invalid single view type: {args.view}")
            return
        print(f"Single-view mode: {args.view}")
    else:
        print("Dual-view mode: ext1 + ext2")
    
    # Prepare output directory
    if args.output != "vggt_video_output":
        if os.path.exists("vggt_video_output"):
            if os.path.islink("vggt_video_output"):
                os.unlink("vggt_video_output")
            else:
                shutil.rmtree("vggt_video_output")
        os.makedirs(args.output, exist_ok=True)
        os.symlink(args.output, "vggt_video_output")
    
    try:
        # Create inference object
        video_inference = VGGTVideoInference(args.checkpoint, args.device)
        
        # Run video inference
        results = video_inference.run_video_inference(
            args.ext1, args.ext2, args.wrist, args.single_view, args.view
        )
        
        print("\n" + "="*80)
        print("Inference summary:")
        print(f"  Inputs: {args.ext1}, {args.ext2}, {args.wrist}")
        print(f"  Checkpoint: {args.checkpoint}")
        print(f"  Output dir: {args.output}")
        if args.single_view:
            print(f"  Mode: single-view ({args.view})")
        else:
            print(f"  Mode: dual-view (ext1 + ext2)")
        print(f"  Frames: {results['statistics']['total_frames']}")
        print(f"  GLB files: {results['statistics']['glb_files_count']}")
        print(f"  FPS: {results['statistics']['fps']}")
        print(f"  Inference total: {results['statistics']['inference_time']:.2f}s")
        print(f"  Visualization total: {results['statistics']['visualization_time']:.2f}s")
        print(f"  Avg inference/frame: {results['statistics']['avg_inference_time_per_frame']:.3f}s")
        print(f"  Avg visualization/frame: {results['statistics']['avg_visualization_time_per_frame']:.3f}s")
        print("="*80)
        
    except Exception as e:
        print(f"Error during video inference: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 