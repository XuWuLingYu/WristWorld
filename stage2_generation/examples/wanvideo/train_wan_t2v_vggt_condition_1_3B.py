import torch, os, imageio, argparse
import shutil
from torchvision.transforms import v2
from einops import rearrange
import lightning as pl
import pandas as pd
from diffsynth import WanVideoPipeline, ModelManager, load_state_dict
from peft import LoraConfig, inject_adapter_in_model
import torchvision
from PIL import Image
import numpy as np
import json
import datetime
from pathlib import Path
from tqdm import tqdm


class LightningModelForTrain(pl.LightningModule):
    def __init__(
        self,
        dit_path,
        learning_rate=1e-5,
        lora_rank=4, lora_alpha=4, train_architecture="lora", lora_target_modules="q,k,v,o,ffn.0,ffn.2", init_lora_weights="kaiming",
        use_gradient_checkpointing=True, use_gradient_checkpointing_offload=False,
        pretrained_lora_path=None,
        max_frames_for_ext=200,  # for temporal embedding capacity
        ext_feat_dim=1280,
        text_dim=4096,
        ext_proj_hidden=2048,
        dropout=0.1,
        # validation/inference configs
        vae_path=None,
        tiled=False,
        tile_size_height=34,
        tile_size_width=34,
        tile_stride_height=18,
        tile_stride_width=16,
        num_inference_steps=50,
        use_first_frame_guide=False,
    ):
        super().__init__()
        model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu")

        # 加载基础模型
        model_manager.load_models(
            ["/mnt/fck/huggingface/models--Wan-AI--Wan2.1-I2V-14B-480P/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"],
            torch_dtype=torch.float32,
        )

        model_manager.load_models(
            [
                "/mnt/fck/huggingface/models--Wan-AI--Wan2.1-T2V-1.3B/snapshots/37ec512624d61f7aa208f7ea8140a131f93afc9a/diffusion_pytorch_model.safetensors",
                "/mnt/fck/huggingface/models--Wan-AI--Wan2.1-T2V-1.3B/snapshots/37ec512624d61f7aa208f7ea8140a131f93afc9a/models_t5_umt5-xxl-enc-bf16.pth",
                "/mnt/fck/huggingface/models--Wan-AI--Wan2.1-T2V-1.3B/snapshots/37ec512624d61f7aa208f7ea8140a131f93afc9a/Wan2.1_VAE.pth",
            ],
            torch_dtype=torch.bfloat16,
        )
        checkpoint_path = dit_path
        # 微调checkpoint
        if checkpoint_path:
            checkpoint_file = Path(checkpoint_path) / "checkpoint" / "mp_rank_00_model_states.pt"
            if not checkpoint_file.exists():
                raise FileNotFoundError(f"model file not exist: {checkpoint_file}")
            try:
                state_dict = torch.load(str(checkpoint_file), map_location="cpu")
                dit_model = model_manager.fetch_model("wan_video_dit")
                if dit_model is not None:
                    dit_model.load_state_dict(state_dict, strict=False)
            except Exception as e:
                import traceback
                traceback.print_exc()
                raise e

        # pipe = WanVideoPipeline.from_model_manager(, torch_dtype=torch.bfloat16, device=device)
        # pipe.enable_vram_management(num_persistent_param_in_dit=60 * 10**9)
        # if os.path.isfile(dit_path):
        #     model_manager.load_models([dit_path])
        # else:
        #     dit_path = dit_path.split(",")
        #     model_manager.load_models([dit_path])
        # # load VAE for decoding during validation
        # if vae_path is not None:
        #     model_manager.load_models([vae_path])
        
        self.pipe = WanVideoPipeline.from_model_manager(model_manager,torch_dtype=torch.bfloat16)
        self.pipe.enable_vram_management(num_persistent_param_in_dit=60 * 10**9)
        self.pipe.scheduler.set_timesteps(1000, training=True)
        self.training_num_timesteps = 1000
        
        # Initialize learnable embeddings and projection for pseudo text tokens
        self.max_frames_for_ext = max_frames_for_ext
        self.ext_feat_dim = ext_feat_dim
        self.text_dim = text_dim
        self.temporal_embed = torch.nn.Embedding(self.max_frames_for_ext, self.ext_feat_dim)
        self.view_embed = torch.nn.Embedding(2, self.ext_feat_dim)
        self.ext_proj = torch.nn.Sequential(
            torch.nn.Linear(self.ext_feat_dim, ext_proj_hidden),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(ext_proj_hidden, self.text_dim),
        )
        
        # Freeze base, then expand to 48 channels BEFORE LoRA
        self.freeze_parameters()
        self.ensure_in_channels_32_before_lora()
        if train_architecture == "lora":
            self.add_lora_to_model(
                self.pipe.denoising_model(),
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                lora_target_modules=lora_target_modules,
                init_lora_weights=init_lora_weights,
                pretrained_lora_path=pretrained_lora_path,
            )
        else:
            ...
            # 因为一些错误，导致之前的模型没有保存ext_proj等新参数，因此单独训练他们
            self.pipe.denoising_model().requires_grad_(True)
        # Lora pretrained lora weights
        if pretrained_lora_path is not None:

            if ',' in pretrained_lora_path:
                pretrained_lora_path = pretrained_lora_path.split(",")
                state_dict = load_state_dict(pretrained_lora_path[1])
                assert 'sign' in state_dict
                dit_state_dict = load_state_dict(pretrained_lora_path[0])
                loaded_keys=["denoising_model","ext_proj","temporal_embed","view_embed"]
                final_state_dict = {key:{} for key in loaded_keys}
                for name,param in state_dict.items():
                    if '.' in name and name.split(".")[0] in loaded_keys:
                        final_state_dict[name.split(".")[0]][name[len(name.split(".")[0])+1:]] = param
                final_state_dict['denoising_model'].update(dit_state_dict)
                missing_keys, unexpected_keys = self.pipe.denoising_model().load_state_dict(final_state_dict['denoising_model'], strict=True)
                all_keys = [i for i, _ in self.pipe.denoising_model().named_parameters()]
                num_updated_keys = len(all_keys) - len(missing_keys)
                num_unexpected_keys = len(unexpected_keys)
                self.ext_proj.load_state_dict(final_state_dict['ext_proj'])
                self.temporal_embed.load_state_dict(final_state_dict['temporal_embed'])
                self.view_embed.load_state_dict(final_state_dict['view_embed'])
                print(f"{num_updated_keys} parameters are loaded from {pretrained_lora_path}. {num_unexpected_keys} parameters are unexpected.")
            else:
                state_dict = load_state_dict(pretrained_lora_path)
                if 'sign' in state_dict:
                    loaded_keys=["denoising_model","ext_proj","temporal_embed","view_embed"]
                    final_state_dict = {key:{} for key in loaded_keys}
                    print(state_dict.keys())
                    for name,param in state_dict.items():
                        if '.' in name and name.split(".")[0] in loaded_keys:
                            final_state_dict[name.split(".")[0]][name[len(name.split(".")[0])+1:]] = param
                    missing_keys, unexpected_keys = self.pipe.denoising_model().load_state_dict(final_state_dict['denoising_model'], strict=True)
                    all_keys = [i for i, _ in self.pipe.denoising_model().named_parameters()]
                    num_updated_keys = len(all_keys) - len(missing_keys)
                    num_unexpected_keys = len(unexpected_keys)
                    self.ext_proj.load_state_dict(final_state_dict['ext_proj'])
                    self.temporal_embed.load_state_dict(final_state_dict['temporal_embed'])
                    self.view_embed.load_state_dict(final_state_dict['view_embed'])
                    print(f"{num_updated_keys} parameters are loaded from {pretrained_lora_path}. {num_unexpected_keys} parameters are unexpected.")
                else:
                    missing_keys, unexpected_keys = self.pipe.denoising_model().load_state_dict(state_dict, strict=False)
                    all_keys = [i for i, _ in self.pipe.denoising_model().named_parameters()]
                    num_updated_keys = len(all_keys) - len(missing_keys)
                    num_unexpected_keys = len(unexpected_keys)
                    print(f"{num_updated_keys} parameters are loaded from {pretrained_lora_path}. {num_unexpected_keys} parameters are unexpected.")
        # Re-enable gradient on the expanded input layer specifically, after LoRA injection
        self.enable_patch_embedding_grad()
        
        self.learning_rate = learning_rate
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        
        # validation/inference config
        self.val_num_inference_steps = num_inference_steps
        self.val_tiled = tiled
        self.val_tile_size = (tile_size_height, tile_size_width)
        self.val_tile_stride = (tile_stride_height, tile_stride_width)
        # validation sample counter (reset each validation epoch)
        self.val_sample_counter = 0
        self.use_first_frame_guide = use_first_frame_guide
    
    def freeze_parameters(self):
        # Freeze parameters
        self.pipe.requires_grad_(False)
        self.pipe.eval()
        self.pipe.denoising_model().train()
    
    def ensure_in_channels_32_before_lora(self,model=None):
        """Ensure model supports 32 input channels (x:16 + y:32) before LoRA injection."""
        if model is None:
            model = self.pipe.denoising_model()
        if hasattr(model, 'patch_embedding'):
            if hasattr(model.patch_embedding, 'module'):
                patch_emb = model.patch_embedding.module
            else:
                patch_emb = model.patch_embedding
            current_in_dim = patch_emb.weight.shape[1]
            if current_in_dim != 32:
                print(f"[INFO] Expanding patch_embedding input channels from {current_in_dim} to 32 before LoRA")
                original_weight = patch_emb.weight.data
                out_dims = list(original_weight.shape)
                out_channels = out_dims[0]
                in_channels = out_dims[1]
                remaining_shape = out_dims[2:]
                new_weight = torch.zeros((out_channels, 32, *remaining_shape), dtype=original_weight.dtype, device=original_weight.device)
                new_weight[:, :in_channels] = original_weight
                torch.nn.init.normal_(new_weight[:, in_channels:], mean=0.0, std=0.02)
                patch_emb.weight.data = new_weight
            else:
                print(f"[INFO] patch_embedding input channels already 32")
        else:
            print("[WARN] denoising model has no patch_embedding; please verify architecture")
    
    def enable_patch_embedding_grad(self):
        model = self.pipe.denoising_model()
        if hasattr(model, 'patch_embedding'):
            if hasattr(model.patch_embedding.module, 'weight'):
                model.patch_embedding.module.weight.requires_grad_(True)
                print("[INFO] Enabled gradient for patch_embedding.weight")
    
    def add_lora_to_model(self, model, lora_rank=4, lora_alpha=4, lora_target_modules="q,k,v,o,ffn.0,ffn.2", init_lora_weights="kaiming", pretrained_lora_path=None, state_dict_converter=None):
        # Add LoRA to UNet
        self.lora_alpha = lora_alpha
        if init_lora_weights == "kaiming":
            init_lora_weights = True
            
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            init_lora_weights=init_lora_weights,
            target_modules=lora_target_modules.split(","),
        )
        model = inject_adapter_in_model(lora_config, model)
        for param in model.parameters():
            # Upcast LoRA parameters into fp32
            if param.requires_grad:
                param.data = param.to(torch.float32)
                
        # Lora pretrained lora weights
        if pretrained_lora_path is not None:
            state_dict = load_state_dict(pretrained_lora_path)
            if state_dict_converter is not None:
                state_dict = state_dict_converter(state_dict)
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            all_keys = [i for i, _ in model.named_parameters()]
            num_updated_keys = len(all_keys) - len(missing_keys)
            num_unexpected_keys = len(unexpected_keys)
            print(f"{num_updated_keys} parameters are loaded from {pretrained_lora_path}. {num_unexpected_keys} parameters are unexpected.")

    def configure_optimizers(self):
        # Collect trainable params: LoRA params (in denoising model), expanded input layer, and new embedding/projection modules
        trainable = list(filter(lambda p: p.requires_grad, self.pipe.denoising_model().parameters()))
        trainable += list(self.temporal_embed.parameters())
        # self.temporal_embed.requires_grad_(True)
        trainable += list(self.view_embed.parameters())
        # self.view_embed.requires_grad_(True)
        trainable += list(self.ext_proj.parameters())
        optimizer = torch.optim.AdamW(trainable, lr=self.learning_rate)
        return optimizer

    def on_save_checkpoint(self, checkpoint):
        checkpoint.clear()
        # Collect trainable (LoRA and expanded) parameter names
        trainable_param_names = list(filter(lambda named_param: named_param[1].requires_grad, self.pipe.denoising_model().named_parameters()))
        trainable_param_names = set([named_param[0] for named_param in trainable_param_names])
        state_dict = self.pipe.denoising_model().state_dict()
        lora_state_dict = {}
        for name, param in state_dict.items():
            if name in trainable_param_names:
                lora_state_dict[name] = param
        tmp_state_dict = {"denoising_model":lora_state_dict,"ext_proj":self.ext_proj.state_dict(),"temporal_embed":self.temporal_embed.state_dict(),"view_embed":self.view_embed.state_dict()}
        final_state_dict = {}
        for name, param in tmp_state_dict.items():
            for subname,ts in param.items():
                final_state_dict[f"{name}.{subname}"] = ts   
        final_state_dict["sign"] = torch.zeros([1]) 
        from safetensors.torch import save_file
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = os.path.join(self.trainer.checkpoint_callback.dirpath, "lora_weights")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"lora_{timestamp}.safetensors")
        save_file(final_state_dict, save_path)
        print(f"Saved LoRA weights to {save_path}")
        
        # Update checkpoint with trainable params only
        checkpoint.update(lora_state_dict)


    def on_train_start(self):
        # Reset scheduler to training timesteps in case validation or other steps changed it
        self.pipe.scheduler.set_timesteps(self.training_num_timesteps, training=True)

    def on_train_epoch_start(self):
        # Ensure each epoch begins with correct training timesteps
        self.pipe.scheduler.set_timesteps(self.training_num_timesteps, training=True)

    def on_validation_epoch_start(self):
        # Reset validation sample counter at the start of each validation epoch
        self.val_sample_counter = 0

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        # Generate video without first-frame guidance; provide ext/control like training
        self.pipe.device = self.device
        # Shapes
        latents_shape = batch["latents"].squeeze(0).shape  # (B,16,F,H,W)
        B, Cx, F, H, W = latents_shape
        assert Cx == 16, f"Expected x channels=16, got {Cx}"
        
        # Prompt context + pseudo tokens from ext
        prompt_emb = batch["prompt_emb"]
        prompt_emb["context"] = prompt_emb["context"].to(self.device)
        ext_frame_feats = batch["ext_frame_feats"].to(self.device)
        ext = ext_frame_feats
        if ext.dim() == 3:
            ext = ext.unsqueeze(0)
        Bx, V, T, C = ext.shape
        assert V == 2 and C == self.ext_feat_dim
        temporal_ids = torch.arange(T, device=self.device)
        view_ids = torch.arange(V, device=self.device)
        temporal_pe = self.temporal_embed(temporal_ids).view(1, 1, T, C)
        view_pe = self.view_embed(view_ids).view(1, V, 1, C)
        enhanced_feats = ext + temporal_pe + view_pe
        proj_in = enhanced_feats.reshape(Bx * V * T, C)
        proj_out = self.ext_proj(proj_in).reshape(Bx, V * T, self.text_dim)
        text_context = prompt_emb["context"].squeeze(1)
        if proj_out.dtype != text_context.dtype:
            proj_out = proj_out.to(text_context.dtype)
        context = torch.cat([text_context[:,:(512-proj_out.shape[1])], proj_out], dim=1)
        prompt_emb["context"] = context

        # Classifier-Free Guidance settings for validation
        cfg_scale = 5.0
        negative_prompt_text = "low quality, distorted, ugly, bad anatomy"
        # Load text encoder to encode negative prompt
        self.pipe.load_models_to_device(["text_encoder"])  # mirror wan_video.__call__ behavior
        prompt_emb_nega = self.pipe.encode_prompt(negative_prompt_text, positive=False)
        
        # Image conditions: controlled by flag
        image_emb = batch["image_emb"]
        control_latents = image_emb["control_latents"].to(self.device)
        # if self.use_first_frame_guide:
        #     y_wrist16 = image_emb["y_wrist16"].to(self.device)
        #     clip_feature = image_emb["clip_feature"].to(self.device)
        # else:
        #     y_wrist16 = torch.zeros_like(image_emb["y_wrist16"]).to(self.device)
        #     clip_feature = torch.zeros_like(image_emb["clip_feature"]).to(self.device)
        # y = torch.cat([control_latents.squeeze(0), y_wrist16.squeeze(0)], dim=1)
        final_image_emb = {}
        
        # Denoising
        self.pipe.scheduler.set_timesteps(self.val_num_inference_steps, denoising_strength=1.0, shift=5.0)
        latents = torch.randn((B, 16, F, H, W), device=self.device, dtype=self.pipe.torch_dtype)
        latents = torch.cat([latents,control_latents.squeeze(0)],dim=1)
        extra_input = self.pipe.prepare_extra_input(latents)
        for step_id, timestep in tqdm(enumerate(self.pipe.scheduler.timesteps)):
            timestep = timestep.unsqueeze(0).to(dtype=self.pipe.torch_dtype, device=self.device)
            # Positive branch
            noise_pred_posi = self.pipe.denoising_model()( 
                latents, timestep=timestep, **prompt_emb, **extra_input, **final_image_emb,
                use_gradient_checkpointing=False,
                use_gradient_checkpointing_offload=False,
            )
            # Negative branch
            noise_pred_nega = self.pipe.denoising_model()( 
                latents, timestep=timestep, **prompt_emb_nega, **extra_input, **final_image_emb,
                use_gradient_checkpointing=False,
                use_gradient_checkpointing_offload=False,
            )
            # CFG combine
            noise_pred = noise_pred_nega + cfg_scale * (noise_pred_posi - noise_pred_nega)

            latents[:,:16] = self.pipe.scheduler.step(noise_pred, self.pipe.scheduler.timesteps[step_id], latents[:,:16])
        
        # Decode
        self.pipe.load_models_to_device(['vae'])
        frames_tensor = self.pipe.decode_video(latents[:,:16], tiled=self.val_tiled, tile_size=self.val_tile_size, tile_stride=self.val_tile_stride)
        self.pipe.load_models_to_device([])
        frames = self.pipe.tensor2video(frames_tensor[0])
        # Build concatenated video with original condition and wrist_rgb videos (top-to-bottom)
        if "meta" not in batch or "paths" not in batch["meta"]:
            raise ValueError("'meta.paths' not found in validation batch; cannot build concatenated video.")
        paths = batch["meta"]["paths"]
        required_video_keys = ["condition", "wrist_rgb"]
        for k in required_video_keys:
            if k not in paths:
                raise ValueError(f"Missing '{k}' in meta.paths; cannot build concatenated video.")

        condition_path = paths["condition"]
        wrist_path = paths["wrist_rgb"]

        # Helper to read up to T frames from a video
        def read_video_frames(video_path, max_frames):
            if isinstance(video_path,list):
                video_path = video_path[0]
            reader = imageio.get_reader(video_path)
            frames_list = []
            try:
                for idx, fr in enumerate(reader):
                    frames_list.append(fr)
                    if len(frames_list) >= max_frames:
                        break
            finally:
                reader.close()
            return frames_list

        T = len(frames)
        gen_frames_pil = frames  # list of PIL.Image
        # Read original videos
        cond_frames_np = read_video_frames(condition_path, T)
        wrist_frames_np = read_video_frames(wrist_path, T)
        if len(cond_frames_np) == 0 or len(wrist_frames_np) == 0:
            raise ValueError("Failed to read frames from condition or wrist_rgb videos during validation.")
        # Align by the minimum available frame count to avoid index errors
        T_aligned = min(T, len(cond_frames_np), len(wrist_frames_np))

        # Target size from generated frames
        target_w, target_h = gen_frames_pil[0].size

        concatenated_frames = []
        for i in range(T_aligned):
            gen_im = gen_frames_pil[i].resize((target_w, target_h), Image.BILINEAR)
            cond_im = Image.fromarray(cond_frames_np[i]).resize((target_w, target_h), Image.BILINEAR)
            wrist_im = Image.fromarray(wrist_frames_np[i]).resize((target_w, target_h), Image.BILINEAR)
            # Vertical stack: condition (top), wrist_rgb (middle), generated (bottom)
            stacked = np.concatenate([
                np.array(cond_im),
                np.array(wrist_im),
                np.array(gen_im)
            ], axis=0)
            concatenated_frames.append(stacked)

        # Save concatenated video to disk with a global counter
        save_dir = os.path.join(self.trainer.default_root_dir, "val_videos")
        os.makedirs(save_dir, exist_ok=True)
        # 获取当前进程的rank（分布式训练时有用）
        if hasattr(self.trainer, "strategy") and hasattr(self.trainer.strategy, "local_rank"):
            rank = self.trainer.strategy.local_rank
        elif hasattr(self.trainer, "global_rank"):
            rank = self.trainer.global_rank
        else:
            # 兼容单卡/非分布式情况
            rank = 0
        save_path = os.path.join(save_dir, f"epoch{self.current_epoch:03d}_sample{self.val_sample_counter:02d}_rank{rank:02d}.mp4")
        imageio.mimsave(save_path, concatenated_frames, fps=8)
        # Increment counter after successful save
        self.val_sample_counter += 1
        
        # Log to wandb (if available)
        try:
            from pytorch_lightning.loggers import WandbLogger
            loggers = self.trainer.loggers if hasattr(self.trainer, 'loggers') else ([self.logger] if self.logger is not None else [])
            for lg in loggers:
                if isinstance(lg, WandbLogger):
                    import wandb
                    lg.experiment.log({
                        f"val_video/epoch_{self.current_epoch:03d}_sample_{batch_idx:02d}": wandb.Video(save_path, fps=8, format="mp4")
                    }, step=self.global_step)
        except Exception as e:
            print(f"[WARN] wandb video log failed: {e}")
        
        # Also log a scalar to ensure validation loop is visible
        self.log("val_generated", float(batch_idx), prog_bar=False)
        
        # Restore training timesteps after validation step
        self.pipe.scheduler.set_timesteps(self.training_num_timesteps, training=True)

    @torch.no_grad()
    def inference_generate_and_save(self, batch, save_root_dir: str, save_basename: str, rank: int = 0):
        """复用validation_step的生成与拼接逻辑，保存纯生成/可视化/GT视频，按save_basename命名。"""
        self.pipe.device = self.device
        latents_shape = batch["latents"].squeeze(0).shape  # (B,16,F,H,W)
        if len(latents_shape) == 5:
            B, Cx, F, H, W = latents_shape
        else:
            Cx, F, H, W = latents_shape
            B = 1
        assert Cx == 16, f"Expected x channels=16, got {Cx}"

        prompt_emb = batch["prompt_emb"]
        prompt_emb["context"] = prompt_emb["context"].to(self.device)
        ext_frame_feats = batch["ext_frame_feats"].to(self.device)
        ext = ext_frame_feats
        if ext.dim() == 3:
            ext = ext.unsqueeze(0)
        Bx, V, T, C = ext.shape
        assert V == 2 and C == self.ext_feat_dim
        temporal_ids = torch.arange(T, device=self.device)
        view_ids = torch.arange(V, device=self.device)
        temporal_pe = self.temporal_embed(temporal_ids).view(1, 1, T, C)
        view_pe = self.view_embed(view_ids).view(1, V, 1, C)
        enhanced_feats = ext + temporal_pe + view_pe
        proj_in = enhanced_feats.reshape(Bx * V * T, C)
        proj_out = self.ext_proj(proj_in).reshape(Bx, V * T, self.text_dim)
        text_context = prompt_emb["context"].squeeze(1)
        if proj_out.dtype != text_context.dtype:
            proj_out = proj_out.to(text_context.dtype)
        context = torch.cat([text_context[:,:(512-proj_out.shape[1])], proj_out], dim=1)
        prompt_emb["context"] = context

        cfg_scale = 5.0
        negative_prompt_text = "low quality, distorted, ugly, bad anatomy"
        self.pipe.load_models_to_device(["text_encoder"])  # mirror wan_video.__call__ behavior
        prompt_emb_nega = self.pipe.encode_prompt(negative_prompt_text, positive=False)

        image_emb = batch["image_emb"]
        control_latents = image_emb["control_latents"].to(self.device)

        self.pipe.scheduler.set_timesteps(self.val_num_inference_steps, denoising_strength=1.0, shift=5.0)
        latents = torch.randn((B, 16, F, H, W), device=self.device, dtype=self.pipe.torch_dtype)
        latents = torch.cat([latents,control_latents],dim=1)
        extra_input = self.pipe.prepare_extra_input(latents)
        for step_id, timestep in enumerate(self.pipe.scheduler.timesteps):
            timestep = timestep.unsqueeze(0).to(dtype=self.pipe.torch_dtype, device=self.device)
            noise_pred_posi = self.pipe.denoising_model()( 
                latents, timestep=timestep, **prompt_emb, **extra_input,
                use_gradient_checkpointing=False,
                use_gradient_checkpointing_offload=False,
            )
            noise_pred_nega = self.pipe.denoising_model()( 
                latents, timestep=timestep, **prompt_emb_nega, **extra_input,
                use_gradient_checkpointing=False,
                use_gradient_checkpointing_offload=False,
            )
            noise_pred = noise_pred_nega + cfg_scale * (noise_pred_posi - noise_pred_nega)
            latents[:,:16] = self.pipe.scheduler.step(noise_pred, self.pipe.scheduler.timesteps[step_id], latents[:,:16])

        self.pipe.load_models_to_device(['vae'])
        frames_tensor = self.pipe.decode_video(latents[:,:16], tiled=self.val_tiled, tile_size=self.val_tile_size, tile_stride=self.val_tile_stride)
        self.pipe.load_models_to_device([])
        frames = self.pipe.tensor2video(frames_tensor[0])

        # 读取并拼接（与validation一致）
        def read_video_frames(video_path, max_frames):
            if isinstance(video_path,list):
                video_path = video_path[0]
            reader = imageio.get_reader(video_path)
            frames_list = []
            try:
                for idx, fr in enumerate(reader):
                    frames_list.append(fr)
                    if len(frames_list) >= max_frames:
                        break
            finally:
                reader.close()
            return frames_list

        if "meta" not in batch or "paths" not in batch["meta"]:
            raise ValueError("'meta.paths' not found in batch; cannot build concatenated video.")
        paths = batch["meta"]["paths"]
        required_video_keys = ["condition", "wrist_rgb"]
        for k in required_video_keys:
            if k not in paths:
                raise ValueError(f"Missing '{k}' in meta.paths; cannot build concatenated video.")
        T = len(frames)
        cond_frames_np = read_video_frames(paths["condition"], T)
        wrist_frames_np = read_video_frames(paths["wrist_rgb"], T)
        if len(cond_frames_np) == 0 or len(wrist_frames_np) == 0:
            raise ValueError("Failed to read frames from condition or wrist_rgb videos during inference.")
        T_aligned = min(T, len(cond_frames_np), len(wrist_frames_np))
        target_w, target_h = frames[0].size
        concatenated_frames = []
        pure_frames = []
        for i in range(T_aligned):
            gen_im = frames[i].resize((target_w, target_h), Image.BILINEAR)
            cond_im = Image.fromarray(cond_frames_np[i]).resize((target_w, target_h), Image.BILINEAR)
            wrist_im = Image.fromarray(wrist_frames_np[i]).resize((target_w, target_h), Image.BILINEAR)
            stacked = np.concatenate([np.array(cond_im), np.array(wrist_im), np.array(gen_im)], axis=0)
            concatenated_frames.append(stacked)
            pure_frames.append(np.array(gen_im))

        samples_dir = os.path.join(save_root_dir, "samples")
        samples_vis_dir = os.path.join(save_root_dir, "samples_vis")
        samples_gt_dir = os.path.join(save_root_dir, "samples_gt")
        sample_condition_dir = os.path.join(save_root_dir, "sample_condition")
        os.makedirs(samples_dir, exist_ok=True)
        os.makedirs(samples_vis_dir, exist_ok=True)
        os.makedirs(samples_gt_dir, exist_ok=True)
        os.makedirs(sample_condition_dir, exist_ok=True)
        save_path = os.path.join(samples_dir, f"{save_basename}.mp4")
        save_vis_path = os.path.join(samples_vis_dir, f"{save_basename}.mp4")
        imageio.mimsave(save_path, pure_frames, fps=8)
        imageio.mimsave(save_vis_path, concatenated_frames, fps=8)
        # 额外保存GT视频（直接拷贝条件视频，保持原始编码/帧率）
        if "meta" in batch and "paths" in batch["meta"] and "condition" in batch["meta"]["paths"]:
            cond_path = batch["meta"]["paths"]["condition"]
            if isinstance(cond_path, list):
                cond_path = cond_path[0]
            gt_out_path = os.path.join(samples_gt_dir, f"{save_basename}.mp4")
            try:
                shutil.copyfile(batch["meta"]["paths"]["wrist_rgb"], gt_out_path)
            except Exception:
                gt_out_path = None
            # 单独保存 condition 到 sample_condition 目录
            try:
                cond_out_path = os.path.join(sample_condition_dir, f"{save_basename}.mp4")
                shutil.copyfile(cond_path, cond_out_path)
            except Exception:
                pass
        else:
            gt_out_path = None
        return save_path, save_vis_path, gt_out_path


def load_inference_file_list(jsonl_path):
    """Load video file names from JSONL file for inference."""
    if not os.path.exists(jsonl_path):
        raise FileNotFoundError(f"JSONL file not found: {jsonl_path}")
    
    target_filenames = set()
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                if 'video_path' in data:
                    video_path = data['video_path']
                    filename = video_path.split('/')[-1]
                    target_filenames.add(filename)
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse JSON line: {line}, error: {e}")
                continue
    
    print(f"Loaded {len(target_filenames)} target filenames from {jsonl_path}")
    return target_filenames

