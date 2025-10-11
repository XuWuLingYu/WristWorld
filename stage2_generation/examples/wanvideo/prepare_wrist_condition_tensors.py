#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import torch
from torchvision.transforms import v2
import imageio
from PIL import Image
import multiprocessing as mp
import inspect

from diffsynth import WanVideoPipeline, ModelManager


def list_samples(dataset_root: Path) -> List[Tuple[Path, Path, Path, Path, str]]:
    ext1_dir = dataset_root / "ext1"
    ext2_dir = dataset_root / "ext2"
    cond_dir = dataset_root / "condition"
    wrist_dir = dataset_root / "wrist_rgb"
    assert ext1_dir.is_dir() and ext2_dir.is_dir() and cond_dir.is_dir() and wrist_dir.is_dir(), \
        "数据集应包含 ext1/ ext2/ condition/ wrist_rgb/ 四个子目录"

    samples: List[Tuple[Path, Path, Path, Path, str]] = []
    for pw in sorted(wrist_dir.glob("*.mp4")):
        name = pw.name
        p1 = ext1_dir / name
        p2 = ext2_dir / name
        pc = cond_dir / name
        if p1.exists() and p2.exists() and pc.exists():
            base = name[:-4]
            samples.append((p1, p2, pc, pw, base))
    return samples


def read_video_frames(path: Path) -> Optional[List[Image.Image]]:
    try:
        reader = imageio.get_reader(str(path))
        frames: List[Image.Image] = []
        num_frames = min(reader.count_frames(), 81)
        for i in range(num_frames):
            arr = reader.get_data(i)
            img = Image.fromarray(arr).convert("RGB")
            img = img.resize((832, 480), resample=Image.BICUBIC)
            frames.append(img)
        reader.close()
        if len(frames) == 0:
            raise ValueError(f"空视频: {path}")
        return frames
    except Exception as e:
        print(f"⚠️  警告：无法读取视频文件 {path.name}: {e}")
        return None
def read_video_frames_all(path: Path) -> Optional[List[Image.Image]]:
    """读取视频帧，截断到81帧，每帧resize到640x480，添加错误处理"""
    try:
        reader = imageio.get_reader(str(path))
        frames: List[Image.Image] = []
        num_frames = reader.count_frames()
        for i in range(num_frames):
            arr = reader.get_data(i)
            img = Image.fromarray(arr).convert("RGB")
            img = img.resize((832, 480), resample=Image.BICUBIC)
            frames.append(img)
        reader.close()
        if len(frames) == 0:
            raise ValueError(f"空视频: {path}")
        return frames
    except Exception as e:
        print(f"⚠️  警告：无法读取视频文件 {path.name}: {e}")
        return None


def frames_to_tensor(frames: List[Image.Image], height: int, width: int) -> torch.Tensor:
    tfm = v2.Compose([
        v2.CenterCrop(size=(height, width)),
        v2.Resize(size=(height, width), antialias=True),
        v2.ToTensor(),
        v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    tensors = [tfm(img) for img in frames]
    video = torch.stack(tensors, dim=0)
    video = video.permute(1, 0, 2, 3).contiguous()
    return video


def encode_ext_frame_feats(pipe: WanVideoPipeline, frames: List[Image.Image]) -> torch.Tensor:
    feats: List[torch.Tensor] = []
    for img in frames:
        img_t = pipe.preprocess_image(img).to(pipe.device)
        with torch.no_grad():
            ctx = pipe.image_encoder.encode_image([img_t])
        feats.append(ctx[:, 0, :].to(dtype=pipe.torch_dtype, device=pipe.device))
    feats_t = torch.cat(feats, dim=0)
    return feats_t

