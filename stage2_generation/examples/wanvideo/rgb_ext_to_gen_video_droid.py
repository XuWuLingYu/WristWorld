#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dual-view Droid → condition → WAN video generation pipeline (supports >81 frames)

Per subdirectory:
1) Read <dir>/ext1.mp4, <dir>/ext2.mp4, <dir/wrist.mp4 (≥81 frames, any resolution)
2) Run VGGT on dual views (ext1, ext2). Project predicted 3D points to wrist view to create <dir>/condition.mp4 (same FPS/frame count/resolution as inputs)
3) Run WAN (1.3B) + pretrained LoRA, using condition as control_latents and (ext1, ext2) as external frame features, to produce <dir>/gen.mp4 (aligned)

Multi-frame strategy:
- VGGT inference: process all frames directly, no chunking needed
- WAN inference: if N is divisible by 81, run N//81 chunks of 81 frames; otherwise run N//81 + 1 chunks. For the last chunk, take the last 81 frames and keep only the required tail frames

Image sizes:
- VGGT: internal processing uses (W, H) = (518, 294) per view
- WAN: internal processing uses (W, H) = (832, 480) for both condition and ext features

Validation:
- Missing ext1/ext2/wrist videos, or invalid frame count/FPS/resolution → raise errors
- Any failure raises exceptions; no placeholders or silent failures

Dependencies:
- OpenCV, imageio, Pillow, torch, numpy, tqdm
- VGGTVideoInference (from vggt_video_inference_droid)
- diffsynth.WanVideoPipeline, ModelManager
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import imageio
from PIL import Image
from tqdm import tqdm


# BASE_DIR = Path(__file__).resolve().parent
# if str(BASE_DIR) not in sys.path:
#     sys.path.append(str(BASE_DIR))

# VGGT dual-view inference (Droid version)
from vggt_video_inference_droid import VGGTVideoInference  # type: ignore
from diffsynth import WanVideoPipeline, ModelManager
import inspect
import multiprocessing as mp
# 确保可导入同目录下的训练脚本
# if str(BASE_DIR) not in sys.path:
sys.path.append('examples/wanvideo')
from train_wan_t2v_vggt_condition_1_3B import LightningModelForTrain  # type: ignore
from prepare_wrist_condition_tensors import read_video_frames
from prepare_wrist_condition_tensors import read_video_frames_all



def _import_cv2() -> "object":
    try:
        import cv2  # type: ignore
    except Exception as e:
        raise RuntimeError("OpenCV (cv2) is required") from e
    return cv2


def read_video_meta(path: Path) -> Tuple[int, float, int, int]:
    cv2 = _import_cv2()
    cap = cv2.VideoCapture(str(path), cv2.CAP_ANY)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {path}")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if frame_count <= 0:
        raise ValueError(f"Invalid frame count: {frame_count} @ {path}")
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError(f"Invalid FPS: {fps} @ {path}")
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid resolution: {width}x{height} @ {path}")
    return frame_count, fps, width, height


def read_all_frames_rgb(path: Path) -> List[np.ndarray]:
    cv2 = _import_cv2()
    cap = cv2.VideoCapture(str(path), cv2.CAP_ANY)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {path}")
    frames: List[np.ndarray] = []
    try:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            frames.append(rgb)
    finally:
        cap.release()
    if len(frames) == 0:
        raise RuntimeError(f"Empty video: {path}")
    return frames


def write_mp4_from_rgb(frames_rgb: List[np.ndarray], out_path: Path, fps: float) -> None:
    if len(frames_rgb) == 0:
        raise ValueError("Cannot write video: empty frame sequence")
    h, w = frames_rgb[0].shape[:2]
    cv2 = _import_cv2()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open writer: {out_path}")
    try:
        for fr in frames_rgb:
            if fr.shape[:2] != (h, w):
                fr = cv2.resize(fr, (w, h), interpolation=cv2.INTER_AREA)
            writer.write(cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()


def build_condition_dual_view(
    vggt: VGGTVideoInference,
    ext1_frames: List[np.ndarray],
    ext2_frames: List[np.ndarray]
) -> List[np.ndarray]:
    """Generate wrist-projection condition frames using VGGT dual-view.
    - VGGT internal input size: (W=518, H=294)
    - Output frames are resized back to the original resolution
    - VGGT processes all frames directly (no chunking)
    """
    cond_frames: List[np.ndarray] = []
    total_frames = len(ext1_frames)
    print(f"VGGT total frames: {total_frames}")
    for ext1_f, ext2_f in tqdm(zip(ext1_frames, ext2_frames), desc="VGGT dual-view -> condition", total=total_frames):
        h0, w0 = ext1_f.shape[:2]
        # Resize to VGGT input size (W=518, H=294)
        ext1_rs = np.array(Image.fromarray(ext1_f).resize((518, 294), Image.BILINEAR))
        ext2_rs = np.array(Image.fromarray(ext2_f).resize((518, 294), Image.BILINEAR))
        preds = vggt.run_inference_on_frames(ext1_rs, ext2_rs, single_view=False)
        pts, cols = vggt.generate_point_cloud_no_sphere(preds, single_view=False)
        wrist_extri = preds["wrist_extrinsic"]
        wrist_intri = preds["wrist_intrinsic"]
        if isinstance(wrist_extri, np.ndarray) and wrist_extri.ndim == 3:
            wrist_extri = wrist_extri[0]
        if isinstance(wrist_intri, np.ndarray) and wrist_intri.ndim == 3:
            wrist_intri = wrist_intri[0]
        vis = vggt.project_points_to_wrist_view(pts, cols, wrist_intri, wrist_extri, img_size=(294, 518))
        vis_rs = np.array(Image.fromarray(vis).resize((w0, h0), Image.BICUBIC))
        cond_frames.append(vis_rs)
    print(f"VGGT generated {len(cond_frames)} condition frames")
    return cond_frames

from torchvision.transforms import v2
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


def infer_one_directory(
    dir_path: Path,
    vggt: VGGTVideoInference,
    model: LightningModelForTrain,
    num_inference_steps: int = 25,
) -> None:
    if os.path.exists(dir_path / "gen.mp4"):
        return
    # Validate dual-view inputs
    ext1_path = dir_path / "ext1.mp4"
    ext2_path = dir_path / "ext2.mp4"
    wrist_path = dir_path / "wrist.mp4"

    for path in [ext1_path, ext2_path, wrist_path]:
        if not path.exists():
            raise FileNotFoundError(f"Missing required file: {path}")

    # Read video metadata (use ext1 as reference)
    n_frames, fps, w, h = read_video_meta(ext1_path)
    # if n_frames < 81:
    #     raise ValueError(f"帧数不足，期望至少 81 帧，实际 {n_frames}: {front_path}")
    
    # # 验证其他视角的帧数
    # for path, name in [(left_path, "left"), (right_path, "right")]:
    #     n_frames_other, _, _, _ = read_video_meta(path)
    #     if n_frames_other != n_frames:
    #         raise ValueError(f"{name} 视角帧数与front不一致: {n_frames_other} vs {n_frames}: {path}")

    # Read all frames (RGB)
    ext1_frames = read_all_frames_rgb(ext1_path)
    ext2_frames = read_all_frames_rgb(ext2_path)
    n_frames = len(ext1_frames)

    # 1) VGGT dual-view -> condition (process all frames)
    cond_path = dir_path / "condition.mp4"
    cond_frames = build_condition_dual_view(vggt, ext1_frames, ext2_frames)
    write_mp4_from_rgb(cond_frames, cond_path, fps)

    # Re-read frames as PIL for WAN (full frame rate)
    ext1_frames_pil = read_video_frames_all(ext1_path)
    ext2_frames_pil = read_video_frames_all(ext2_path)

    # 2) WAN inference (supports >81 frames)
    print(f"Start WAN inference, total frames: {n_frames}")
    
    # Encode control_latents (condition) and ext features (dual-view)
    # WAN internal resolution is fixed to (W=832, H=480)
    enc_h = 480
    enc_w = 832
    pipe = model.pipe
    pipe.device = model.device
    cond_frames = read_video_frames_all(cond_path)
    print(len(cond_frames), len(ext1_frames_pil))
    # Read wrist frames (RGB)
    wrist_frames = read_all_frames_rgb(wrist_path)
    wrist_frames_pil = [Image.fromarray(f) for f in wrist_frames]
    # WAN chunking for >81 frames
    print("Frames > 81: start chunked WAN inference")
    
    # 计算分段
    if n_frames % 81 == 0:
        num_segments = n_frames // 81
    else:
        num_segments = n_frames // 81 + 1
    
    all_generated_frames = []
    
    for seg_idx in range(num_segments):
        start_idx = seg_idx * 81
        end_idx = min(start_idx + 81, n_frames)
        actual_frames = end_idx - start_idx
        
        print(f"WAN chunk {seg_idx + 1}/{num_segments}: frames {start_idx}-{end_idx-1}")
        
        
        
        # Handle last partial chunk
        if actual_frames < 81 and seg_idx == num_segments - 1:
            if n_frames >= 81:
                # Use the last 81 frames
                start_idx_adj = max(0, n_frames - 81)
                cond_segment = cond_frames[start_idx_adj:start_idx_adj + 81]
                ext1_segment = ext1_frames_pil[start_idx_adj:start_idx_adj + 81]
                ext2_segment = ext2_frames_pil[start_idx_adj:start_idx_adj + 81]
                actual_frames = 81
                print(f"Last chunk adjusted: take 81 frames starting at {start_idx_adj}")
            else:
                # Pad to 81 by repeating the last frame
                def pad_to_81(frames):
                    if len(frames) == 0:
                        raise RuntimeError("Empty frames, cannot pad to 81")
                    repeat_times = 81 - len(frames)
                    return frames + [frames[-1]] * repeat_times
                cond_segment = pad_to_81(cond_frames)
                ext1_segment = pad_to_81(ext1_frames_pil)
                ext2_segment = pad_to_81(ext2_frames_pil)
                actual_frames = len(ext1_frames_pil)
                print("Last chunk adjusted: frames <81 padded to 81")
        else:
            # Slice current chunk
            ext1_segment = ext1_frames_pil[start_idx:end_idx]
            ext2_segment = ext2_frames_pil[start_idx:end_idx]
            cond_segment = cond_frames[start_idx:end_idx]

        # Encode current chunk (condition:
        # resize to (832,480) before VAE encoding)
        cond_tensor = frames_to_tensor(cond_segment, enc_h, enc_w).to(device=pipe.device, dtype=pipe.torch_dtype)
        with torch.no_grad():
            control_latents = pipe.encode_video(cond_tensor.unsqueeze(0), tiled=True, tile_size=(34,34), tile_stride=(18,16))
            if isinstance(control_latents, (list, tuple)):
                control_latents = control_latents[0]
        # Text prompt encoding
        prompt = "robotic manipulation scene"
        prompt_emb = model.pipe.encode_prompt(prompt)
        assert isinstance(prompt_emb, dict) and "context" in prompt_emb
        
        # Encode dual-view features (ext1/ext2). Resize frames to (832,480) for consistency.
        def to_832x480(frames: List[Image.Image]) -> List[Image.Image]:
            return [img.resize((enc_w, enc_h), Image.BILINEAR) for img in frames]
        ext1_feats = encode_ext_frame_feats(pipe, to_832x480(ext1_segment))   # ext1 -> index 0
        ext2_feats = encode_ext_frame_feats(pipe, to_832x480(ext2_segment))   # ext2 -> index 1
        
        # Stack into shape (2, T, 1280)
        ext_frame_feats = torch.stack([ext1_feats, ext2_feats], dim=0)
        assert ext_frame_feats.shape[0] == 2 and ext_frame_feats.shape[1] == 81, f"Unexpected ext_frame_feats shape: {tuple(ext_frame_feats.shape)}"
        # Build batch
        latent_T = (len(cond_segment) - 1) // 4 + 1
        dummy_latents_shape = torch.zeros((1, 16, latent_T, enc_h // 8, enc_w // 8), dtype=pipe.torch_dtype)
        batch = {
            "latents": dummy_latents_shape,
            "prompt_emb": {"context": prompt_emb["context"]},
            "image_emb": {"control_latents": control_latents},
            "ext_frame_feats": ext_frame_feats,  # (2, T, 1280)
            "meta": {
                "paths": {
                    "condition": str(cond_path),
                    "wrist_rgb": str(wrist_path),
                }
            },
        }
        # Generate current chunk
        out_root = str(dir_path)
        base = f"{dir_path.name}_seg{seg_idx}"
        save_p, save_vis_p, save_gt_p = model.inference_generate_and_save(batch, out_root, save_basename=base, rank=0)
        
        # Read generated frames and keep the necessary tail when needed
        if save_p and os.path.isfile(save_p):
            generated_frames = read_video_frames(Path(save_p))
            
            # Last partial chunk: keep only the remaining frames
            if seg_idx == num_segments - 1 and n_frames % 81 != 0:
                remaining_frames = n_frames - (n_frames // 81) * 81
                generated_frames = generated_frames[-remaining_frames:]
                print(f"Last chunk: keep last {remaining_frames} frames")
            
            all_generated_frames.extend(generated_frames)
            print(f"Chunk {seg_idx + 1} generated {len(generated_frames)} frames")
        
        # Merge and save final video progressively
        if all_generated_frames:
            print(f"Merging all chunks, total {len(all_generated_frames)} frames")
            final_video_path = dir_path / "gen.mp4"
            write_mp4_from_rgb([np.array(f) for f in all_generated_frames], final_video_path, fps)
            save_p = str(final_video_path)
    
    # Ensure gen.mp4 exists (copy from intermediate path if needed)
    try:
        if save_p is not None and os.path.isfile(save_p):
            target = dir_path / "gen.mp4"
            if str(target) != str(save_p):
                import shutil
                shutil.copyfile(save_p, target)
    except Exception:
        pass


def worker_loop(
    gpu_id: int,
    task_queue: "mp.Queue[Path]",
    result_queue: "mp.Queue[Tuple[str, str, str]]",
    vggt_ckpt: str,
    text_encoder_path: str,
    vae_path: str,
    image_encoder_path: str,
    pretrained_lora_path: str,
    num_inference_steps: int,
) -> None:
    device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
    # Load VGGT and WAN once per process
    vggt = VGGTVideoInference(checkpoint_path=vggt_ckpt, device=device)
    # Build WAN + LoRA via LightningModelForTrain with training-aligned params
    model = LightningModelForTrain(
        dit_path="/mnt/fck/code/DiffSynth-Studio/output/wan_1.3B_full_agidroidmind600k/checkpoints/wan-epoch=99-train_loss=0.1211.ckpt",
        learning_rate=1e-5,
        train_architecture="full",
        lora_rank=4,
        lora_alpha=4,
        lora_target_modules="q,k,v,o,ffn.0,ffn.2",
        init_lora_weights="kaiming",
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        pretrained_lora_path=pretrained_lora_path,
        vae_path=vae_path,
        tiled=False,
        tile_size_height=34,
        tile_size_width=34,
        tile_stride_height=18,
        tile_stride_width=16,
        num_inference_steps=num_inference_steps,
        use_first_frame_guide=False,
    )
    # Device
    model.to(device)
    model.eval()
    while True:
        dir_path = task_queue.get()
        if dir_path is None:
            break
        try:
            infer_one_directory(dir_path, vggt=vggt, model=model, num_inference_steps=num_inference_steps)
            print(f"GPU{gpu_id} done: {dir_path}")
            result_queue.put(("ok", dir_path.name, ""))
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"GPU{gpu_id} failed: {dir_path} -> {e}")
            result_queue.put(("err", dir_path.name, str(e)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Droid dual-view → condition → WAN video generation")
    parser.add_argument("--input_root", required=True, help="Root directory with subfolders each containing ext1.mp4, ext2.mp4, wrist.mp4")
    parser.add_argument("--vggt_checkpoint", required=True, help="Path to VGGT checkpoint (dual-view)")
    parser.add_argument("--text_encoder_path", required=True)
    parser.add_argument("--vae_path", required=True)
    parser.add_argument("--image_encoder_path", required=True)
    parser.add_argument("--pretrained_lora_path", required=True, help="WAN pretrained LoRA weights (.pt/.safetensors)")
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--gpus", type=str, required=True, help="Comma-separated GPU IDs, e.g., '0,1,2'")
    parser.add_argument("--start_idx", type=int, default=-1, help="Start index for subdir slicing")
    parser.add_argument("--end_idx", type=int, default=-1, help="End index for subdir slicing")
    args = parser.parse_args()

    input_root = Path(args.input_root)
    if not input_root.exists() or not input_root.is_dir():
        raise FileNotFoundError(f"Input root does not exist or is not a directory: {input_root}")

    subdirs = [p for p in sorted(input_root.iterdir()) if p.is_dir()]
    if args.end_idx != -1:
        subdirs = subdirs[:args.end_idx]
    if args.start_idx != -1:
        subdirs = subdirs[args.start_idx:]
    
    
    if len(subdirs) == 0:
        raise FileNotFoundError(f"No subdirectories found under {input_root}")

    # Multi-GPU parallelism: one worker per GPU (adjust as needed)
    gpu_list = [int(x) for x in str(args.gpus).split(',') if x.strip() != ""]
    assert len(gpu_list) >= 1, "--gpus 至少指定一个 GPU"

    ctx = mp.get_context("spawn")
    task_queue: "mp.Queue[Path]" = ctx.Queue()
    result_queue: "mp.Queue[Tuple[str, str, str]]" = ctx.Queue()

    # Fill task queue
    for d in subdirs:
        task_queue.put(d)
    # Termination sentinels (one per worker)
    total_workers = 1 * len(gpu_list)
    for _ in range(total_workers):
        task_queue.put(None)

    procs: List[mp.Process] = []
    for gpu_id in gpu_list:
        for _ in range(1):
            p = ctx.Process(
                target=worker_loop,
                args=(
                    gpu_id,
                    task_queue,
                    result_queue,
                    args.vggt_checkpoint,
                    args.text_encoder_path,
                    args.vae_path,
                    args.image_encoder_path,
                    args.pretrained_lora_path,
                    int(args.num_inference_steps),
                ),
            )
            p.start()
            procs.append(p)

    # Collect results
    errors: List[Tuple[str, str]] = []
    for _ in range(len(subdirs)):
        try:
            status, name, info = result_queue.get(timeout=360000)
            if status == "err":
                errors.append((name, info))
        except Exception as e:
            print(f"Main process timed out or failed while waiting for results: {e}")

    for p in procs:
        p.join()

    if errors:
        msg = "\n".join([f"{n}: {err}" for n, err in errors])
        raise RuntimeError(f"Some directories failed:\n{msg}")

    print("All done!")


if __name__ == "__main__":
    main()
