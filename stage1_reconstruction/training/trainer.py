# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

'''
# Plan: This file should provide a trainer class, which provides
# 1. The init of DDP                                            Done
# 2. The init of optimizers, tb, timers, and so on               Done
# 3. A basic training framework (especially for finetuning)
#       self._train_epoch_                                     Done
#       self._process_batch_                                  Done
#       self._step_                                           Done
# 4. The training loop: more utils to be added
'''
import contextlib
import time
import copy
import functools
import gc
import json
import logging
import math
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ["MKL_THREADING_LAYER"] = "GNU"
os.environ["HYDRA_FULL_ERROR"] = "1"
os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"

import random
import string
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

import torch
import torch.distributed as dist
import torch.nn as nn
import torchvision


import fvcore
from einops import rearrange
from hydra.utils import instantiate
from iopath.common.file_io import g_pathmgr
from PIL import Image

from datetime import timedelta

#
from training.train_utils.general import *
from training.train_utils.logging import setup_logging
from training.train_utils.distributed import get_machine_local_and_dist_rank
from training.train_utils.freeze import freeze_modules
from training.train_utils.optimizer import construct_optimizers
from training.train_utils.normalization import normalize_camera_extrinsics_and_points_batch
from training.train_utils.checkpoint import DDPCheckpointSaver
from training.validation_visualizer import ValidationVisualizer


def get_amp_type(amp_dtype_str):
    """
    Convert AMP dtype string to torch dtype.
    
    Args:
        amp_dtype_str: String representation of dtype ("bfloat16" or "float16")
        
    Returns:
        torch.dtype: Corresponding torch dtype
    """
    assert amp_dtype_str in ["bfloat16", "float16"], f"Invalid Amp type: {amp_dtype_str}"
    if amp_dtype_str == "bfloat16":
        return torch.bfloat16
    else:
        return torch.float16


def unwrap_ddp_or_fsdp_if_wrapped(model):
    """
    Unwrap DDP or FSDP wrapped model to get the original model.
    
    Args:
        model: Model that might be wrapped by DDP/FSDP
        
    Returns:
        Original unwrapped model
    """
    # Handle DDP wrapped models
    if hasattr(model, 'module'):
        return model.module
    # If not wrapped, return the model as-is
    return model


class Trainer:
    """
    Trainer supporting the DDP training strategies.
    """

    EPSILON = 1e-8

    def __init__(
        self,
        *,
        data: Dict[str, Any],
        model: Dict[str, Any],
        logging: Dict[str, Any],
        checkpoint: Dict[str, Any],
        max_epochs: int,
        mode: str = "train",
        device: str = "cuda",
        seed_value: int = 123,
        val_epoch_freq: int = 1,
        distributed: Dict[str, bool] = None,
        cuda: Dict[str, bool] = None,
        limit_train_batches: Optional[int] = None,
        limit_val_batches: Optional[int] = None,
        optim: Optional[Dict[str, Any]] = None,
        loss: Optional[Dict[str, Any]] = None,
        env_variables: Optional[Dict[str, Any]] = None,
        accum_steps: int = 1,
        early_validation: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        self._setup_env_variables(env_variables)
        self._setup_timers()

        self.data_conf = data
        self.model_conf = model
        self.loss_conf = loss
        self.logging_conf = logging
        self.checkpoint_conf = checkpoint

        # hyperparameters
        self.accum_steps = accum_steps
        self.max_epochs = max_epochs
        self.mode = mode
        self.val_epoch_freq = val_epoch_freq
        self.limit_train_batches = limit_train_batches
        self.limit_val_batches = limit_val_batches
        self.optim_conf = optim

        # Early validation configuration
        self.early_validation_conf = early_validation

        self.where = 0.0
        self.seed_value = seed_value
        
        self.track_confidence_threshold = self.loss_conf.projection.track_confidence_threshold

        self._setup_device(device)
        self._setup_torch_dist_and_backend(cuda, distributed)

        safe_makedirs(self.logging_conf.log_dir)
        setup_logging(
            __name__,
            output_dir=self.logging_conf.log_dir,
            rank=self.rank,
            log_level_primary=self.logging_conf.log_level_primary,
            log_level_secondary=self.logging_conf.log_level_secondary,
            all_ranks=self.logging_conf.all_ranks,
        )
        set_seeds(seed_value, self.max_epochs, self.distributed_rank)

        # Only assert distributed initialization if distributed training is enabled
        if distributed is not None and distributed is not False:
            assert (
                is_dist_avail_and_initialized()
            ), "Torch distributed needs to be initialized before calling the trainer."



        self._setup_components()  # Except Optimizer everything is setup here.
        self._setup_dataloaders()

        self.model.to(self.device)
        if self.scaler:
            copy_data_to_device(self.scaler, self.device)

        self.time_elapsed_meter = DurationMeter("Time Elapsed", self.device, ":.4f")

        # 初始化validation可视化器
        self.val_visualizer = ValidationVisualizer(
            output_base_dir=self.logging_conf.log_dir,
            rank=self.distributed_rank,
            experiment_name=kwargs['exp_name']
        )

        if self.mode != "val":
            self.optims = construct_optimizers(
                self.model,
                self.optim_conf,
            )


        ################################
        # If you want to force to resume from a specific checkpoint, you can do so by setting the resume_checkpoint_path in the config
        if self.checkpoint_conf.resume_checkpoint_path is not None:
            self._load_resuming_checkpoint(self.checkpoint_conf.resume_checkpoint_path)
        # else:   
        #     ckpt_path = get_resume_checkpoint(self.checkpoint_conf.save_dir)
        #     if ckpt_path is not None:
        #         self._load_resuming_checkpoint(ckpt_path)
                
        self._setup_ddp_distributed_training(distributed, device)
        
        # Only call barrier if distributed training is enabled
        if distributed is not None and distributed is not False:
            dist.barrier()

    def _setup_timers(self):
        """
        Initializes counters for elapsed time and eta.
        """
        self.start_time = time.time()
        self.ckpt_time_elapsed = 0


    def _get_meters(self, phase_filters=None):
        if self.meters is None:
            return {}
        meters = {}
        for phase, phase_meters in self.meters.items():
            if phase_filters is not None and phase not in phase_filters:
                continue
            for key, key_meters in phase_meters.items():
                for name, meter in key_meters.items():
                    meters[f"{phase}_{key}/{name}"] = meter
        return meters


    def _setup_env_variables(self, env_variables_conf) -> None:
        if env_variables_conf is not None:
            for variable_name, value in env_variables_conf.items():
                os.environ[variable_name] = value
        print(f"Environment:\n{json.dumps(dict(os.environ), sort_keys=True, indent=2)}")

    def _setup_torch_dist_and_backend(self, cuda_conf, distributed_conf) -> None:
        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = cuda_conf.cudnn_deterministic
            torch.backends.cudnn.benchmark = cuda_conf.cudnn_benchmark
            torch.backends.cuda.matmul.allow_tf32 = cuda_conf.allow_tf32
            torch.backends.cudnn.allow_tf32 = cuda_conf.allow_tf32

        # Only initialize distributed training if distributed_conf is not None and not False
        if distributed_conf is not None and distributed_conf is not False:
            dist.init_process_group(backend=distributed_conf.backend,
            timeout=timedelta(minutes=distributed_conf.timeout_mins))
            self.rank = dist.get_rank()
        else:
            # Single GPU mode
            self.rank = 0



    def _load_resuming_checkpoint(self, ckpt_path: str):
        try:
            logging.info(f"Resuming training from {ckpt_path} (rank {self.rank})")
            print("loading checkpoint")
            with g_pathmgr.open(ckpt_path, "rb") as f:
                checkpoint = torch.load(f, map_location="cpu")
            print("loading checkpoint done")
            model_state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
            missing_keys, unexpected_keys = self.model.load_state_dict(model_state_dict, strict=self.checkpoint_conf.strict)
            
            if self.rank == 0:
                # 注释掉冗余的missing/unexpected keys输出
                # if missing_keys:
                #     logging.warning(f"Missing keys when loading model state dict: {missing_keys}")
                # else:
                #     logging.info(f"No missing keys when loading model state dict")
                    
                # if unexpected_keys:
                #     logging.warning(f"Unexpected keys when loading model state dict: {unexpected_keys}")
                # else:
                #     logging.info(f"No unexpected keys when loading model state dict")
                
                # 简化输出，只在有问题时显示
                if missing_keys or unexpected_keys:
                    if len(missing_keys) >0:
                        print(missing_keys[:10])
                    logging.info(f"🔧 模型权重加载: 跳过 {len(missing_keys)} 个missing keys, {len(unexpected_keys)} 个unexpected keys")
                else:
                    logging.info("✅ 模型权重完全匹配加载")
                
            del checkpoint
        except:
            import traceback
            traceback.print_exc()




    def _setup_device(self, device):
        # Try to get distributed ranks, fall back to single GPU if not available
        try:
            self.local_rank, self.distributed_rank = get_machine_local_and_dist_rank()
        except (AssertionError, TypeError, ValueError):
            # Single GPU mode - use environment variable or default to 0
            self.local_rank = int(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0])
            self.distributed_rank = 0
            
        if device == "cuda":
            self.device = torch.device("cuda", self.local_rank)
            torch.cuda.set_device(self.local_rank)
        elif device == "cpu":
            self.device = torch.device("cpu")
        else:
            raise ValueError(f"Unsupported device: {device}")


    def _setup_components(self):
        logging.info("Setting up components: Model, loss, optim, meters etc.")
        self.epoch = 0
        self.steps = {'train': 0, 'val': 0}
        
        # 初始化meters属性，避免AttributeError
        self.meters = None
        
        # 初始化est_epoch_time属性，避免AttributeError
        self.est_epoch_time = {'train': 0.0, 'val': 0.0}

        self.tb_writer = instantiate(self.logging_conf.tensorboard_writer, _recursive_=False)
        self.model = instantiate(self.model_conf, _recursive_=False)
        if getattr(self.optim_conf, "frozen_module_names", None):
            logging.info(
                f"[Start] Freezing modules: {self.optim_conf.frozen_module_names} on rank {self.distributed_rank}"
            )
            self.model = freeze_modules(
                self.model,
                patterns=self.optim_conf.frozen_module_names,
            )
            logging.info(
                f"[Done] Freezing modules: {self.optim_conf.frozen_module_names} on rank {self.distributed_rank}"
            )

        model_summary_path = os.path.join(self.logging_conf.log_dir, "model.txt")
        model_summary(self.model, log_file=model_summary_path)
        logging.info(f"Model summary saved to {model_summary_path}")

        # TODO: Remind myself to finish this
        # Clean the dirty loss and build a single object
        self.loss = instantiate(self.loss_conf, _recursive_=False)


        # Use standard Gradient Scaler for DDP
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.optim_conf.amp.enabled)
        self.gradient_clipper = instantiate(self.optim_conf.gradient_clip)

        # 禁用track_head参数的梯度（只训练camera_head和wrist_head）
        for name, param in self.model.named_parameters():
            if "track_head" in name:
                param.requires_grad = False
                # logging.info(f"🔒 禁用track_head参数梯度: {name}")

        logging.info("Successfully initialized all training components: model, loss function, optimizer, and etc.")


    def _setup_dataloaders(self):
        self.train_dataset = None
        self.val_dataset = None

        # 🔥 检查是否启用视频输入模式
        video_input_enabled = False
        if hasattr(self, 'data_conf') and hasattr(self.data_conf, 'video_input'):
            video_input_enabled = getattr(self.data_conf.video_input, 'enabled', False)
        
        if video_input_enabled:
            # 🔥 视频输入模式：使用视频数据集
            logging.info("🎬 启用视频输入模式")
            
            # 获取视频路径
            ext1_path = self.data_conf.video_input.ext1_video_path
            ext2_path = self.data_conf.video_input.ext2_video_path
            wrist_path = self.data_conf.video_input.wrist_video_path
            
            # 验证视频路径
            if not ext1_path or not ext2_path or not wrist_path:
                raise ValueError("视频输入模式下必须提供所有三个视频路径")
            
            logging.info(f"📹 视频路径:")
            logging.info(f"  ext1: {ext1_path}")
            logging.info(f"  ext2: {ext2_path}")
            logging.info(f"  wrist: {wrist_path}")
            
            # 修改数据集配置以传递视频路径
            if self.mode in ["train", "val"]:
                val_config = copy.deepcopy(self.data_conf.val)
                val_config.dataset.ext1_video_path = ext1_path
                val_config.dataset.ext2_video_path = ext2_path
                val_config.dataset.wrist_video_path = wrist_path
                self.val_dataset = instantiate(val_config, _recursive_=False)
                if self.val_dataset is not None:
                    self.val_dataset.seed = self.seed_value

            if self.mode in ["train"]:
                train_config = copy.deepcopy(self.data_conf.train)
                train_config.dataset.ext1_video_path = ext1_path
                train_config.dataset.ext2_video_path = ext2_path
                train_config.dataset.wrist_video_path = wrist_path
                self.train_dataset = instantiate(train_config, _recursive_=False)
                self.train_dataset.seed = self.seed_value
                
        else:
            # 🔥 正常模式：使用原有数据集
            if self.mode in ["train", "val"]:
                self.val_dataset = instantiate(
                    self.data_conf.get('val', None), _recursive_=False
                )
                if self.val_dataset is not None:
                    self.val_dataset.seed = self.seed_value

            if self.mode in ["train"]:
                self.train_dataset = instantiate(self.data_conf.train, _recursive_=False)
                self.train_dataset.seed = self.seed_value


    def _setup_ddp_distributed_training(self, distributed_conf, device):
        assert isinstance(self.model, torch.nn.Module)

        # Only wrap with DDP if distributed training is enabled
        if distributed_conf is not None and distributed_conf is not False:
            ddp_options = dict(
                find_unused_parameters=distributed_conf.find_unused_parameters,
                gradient_as_bucket_view=distributed_conf.gradient_as_bucket_view,
                bucket_cap_mb=distributed_conf.bucket_cap_mb,
                broadcast_buffers=distributed_conf.broadcast_buffers,
            )

            self.model = nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[self.local_rank] if device == "cuda" else [],
                **ddp_options,
            )
        else:
            # Single GPU mode - no DDP wrapping needed
            logging.info("Single GPU mode: skipping DDP wrapping")


    def _move_to_device(self):
        print(
            f"Moving components to device {self.device} and local rank {self.local_rank}."
        )
        self.model.to(self.device)

        if self.loss:
            copy_data_to_device(self.loss, self.device)
        if self.scaler:
            copy_data_to_device(self.scaler, self.device)
        for meter in self._get_meters().values():
            meter.set_sync_device(self.device)

        print(
            f"Done moving components to device {self.device} and local rank {self.local_rank}."
        )

    def save_checkpoint(self, epoch, checkpoint_names=None):        
        checkpoint_folder = self.checkpoint_conf.save_dir
        safe_makedirs(checkpoint_folder)
        if checkpoint_names is None:
            checkpoint_names = ["checkpoint"]
            if (
                self.checkpoint_conf.save_freq > 0
                and int(epoch) % self.checkpoint_conf.save_freq == 0
                and (int(epoch) > 0 or self.checkpoint_conf.save_freq == 1)
            ):
                checkpoint_names.append(f"checkpoint_{int(epoch)}")

        checkpoint_content = {
            "prev_epoch": epoch,
            "steps": self.steps,
            "time_elapsed": self.time_elapsed_meter.val,
            "optimizer": [optim.optimizer.state_dict() for optim in self.optims],
        }
        
        if len(self.optims) == 1:
            checkpoint_content["optimizer"] = checkpoint_content["optimizer"][0]
        if self.optim_conf.amp.enabled:
            checkpoint_content["scaler"] = self.scaler.state_dict()

        # Save the checkpoint for DDP only
        saver = DDPCheckpointSaver(
            checkpoint_folder,
            checkpoint_names=checkpoint_names,
            rank=self.distributed_rank,
            epoch=epoch,
        )

        # Get the unwrapped model (handle both DDP and single GPU)
        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            model = self.model.module
        else:
            model = self.model

        saver.save_checkpoint(
            model=model,
            ema_models = None,
            skip_saving_parameters=[],
            **checkpoint_content,
        )



    def _get_train_dataset_checkpoint_state(self):
        if self.train_dataset is not None:
            return self.train_dataset.get_checkpoint_state()
        return None


    def _get_scalar_log_keys(self, phase):
        if self.logging_conf.scalar_keys_to_log is not None:
            return self.logging_conf.scalar_keys_to_log[phase].keys_to_log
        else:
            return []



    def _init_model_initializer(self):
        return instantiate(self.checkpoint_conf.model_weight_initializer)

    def _call_model_initializer(self, model_weight_initializer):
        if model_weight_initializer is not None:
            logging.info(
                f"Loading pretrained checkpoint from {self.checkpoint_conf.model_weight_initializer}"
            )
            self.model = model_weight_initializer(model=self.model)

    def is_intermediate_val_epoch(self, epoch):
        return epoch % self.val_epoch_freq == 0 and epoch < self.max_epochs - 1

    def run(self):
        mode = self.mode
        assert mode in [
            "train",
            "val",
        ]
        if mode == "train":
            self.run_train()
            self.run_val()
        elif mode == "val":
            self.run_val()
        else:
            raise ValueError(f"Invalid mode: {mode}")

    def run_train(self):
        dataloader = self.train_dataset.get_loader(epoch=int(self.epoch))
        while self.epoch < self.max_epochs:
            set_seeds(self.seed_value + self.epoch * 100, self.max_epochs, self.distributed_rank)
            
            # dataloader = self.train_dataset.get_loader(epoch=int(self.epoch))
            
            self.train_epoch(dataloader)
            
            # Save checkpoint before validating
            self.save_checkpoint(self.epoch)
            
            # Run validation after each epoch if needed
            if self.epoch % self.val_epoch_freq == 0:
                logging.info(f"🔍 Running validation after epoch {self.epoch}")
                # self.run_val()

            # del dataloader
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            self.epoch += 1
        self.epoch -= 1

    @torch.no_grad()
    def _dump_model_stats_for_tests(self):
        # Done on all ranks because of FSDP and also for debugging DDP
        logging.info("Dumping stats of the trained model")
        stats = {
            "epoch": self.epoch,
            "rank": self.distributed_rank,
            "model": sum(p.sum() for p in self.model.parameters()).item(),
        }
        with g_pathmgr.open(
            os.path.join(
                self.logging_conf.log_dir,
                f"unit_tests_model_stats_{self.distributed_rank}.json",
            ),
            "a",
        ) as f:
            f.write(json.dumps(stats) + "\n")

    def run_val(self):
        if not self.val_dataset:
            return

        # 仅主进程进入validation，其他进程在barrier等待
        distributed_enabled = is_dist_avail_and_initialized()
        if distributed_enabled:
            dist.barrier()
        if distributed_enabled and self.distributed_rank != 0:
            # 非主进程：等待主进程完成validation
            dist.barrier()
            return

        dataloader = self.val_dataset.get_loader(epoch=int(self.epoch))
        
        # 检查是否有is_fresh_epoch方法，如果没有则使用默认值
        if hasattr(self.val_dataset, 'is_fresh_epoch'):
            is_fresh_epoch = self.val_dataset.is_fresh_epoch(epoch=int(self.epoch))
        else:
            is_fresh_epoch = True  # 默认值
            
        outs = self.val_epoch(dataloader, is_fresh_epoch)
        del dataloader
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        self.tb_writer.log_dict(outs, self.epoch)  # Logged only on rank 0

        # 通知其他进程validation完成
        if distributed_enabled:
            dist.barrier()

        if self.distributed_rank == 0:
            with g_pathmgr.open(
                os.path.join(self.logging_conf.log_dir, "val_stats.json"),
                "a",
            ) as f:
                f.write(json.dumps(outs) + "\n")

    def val_epoch(self, val_loader, is_fresh_epoch):
        import torch
        batch_time = AverageMeter("Batch Time", self.device, ":.4f")
        data_time = AverageMeter("Data Time", self.device, ":.4f")
        mem = AverageMeter("Mem (GB)", self.device, ":.4f")
        data_times = []

        iters_per_epoch = len(val_loader)

        curr_phases = ['val']
        curr_models = [self.model]

        assert len(curr_phases)==1

        phase = curr_phases[0]
        loss_names = ["objective"] + self._get_scalar_log_keys(phase)
        loss_names = [f"Loss/{phase}_{name}" for name in loss_names]
        
        loss_meters = {
            name: AverageMeter(name, self.device, ":.4f") for name in loss_names
        }

        for model in curr_models:
            with self._to_full_val_model(model):
                model.eval()
                if hasattr(
                    unwrap_ddp_or_fsdp_if_wrapped(model), "on_validation_epoch_start"
                ):
                    unwrap_ddp_or_fsdp_if_wrapped(model).on_validation_epoch_start()

        # we track the data indices to ensure we are getting proper number of samples
        # with expected shuffling
        local_data_ids = []

        progress = ProgressMeter(
            num_batches=iters_per_epoch,
            meters=[batch_time, data_time, mem,
                   self.time_elapsed_meter,
                   *loss_meters.values(),],
            real_meters=self._get_meters(curr_phases),
            prefix="Val Epoch: [{}]".format(self.epoch),
        )
        import time
        end = time.time()

        # 强制限制validation的最大batch数为20
        limit_val_batches = min(iters_per_epoch, 40)

        # 🎯 新增：每个epoch随机选择3个batch进行可视化
        import random
        random.seed(self.seed_value + self.epoch * 1000)  # 确保每个epoch都有不同的随机选择
        max_batches = min(25, limit_val_batches)
        if max_batches >= 20:
            # 随机选择3个不同的batch索引
            visualization_batches = set(random.sample(range(max_batches), 20))
            logging.info(f"🎲 Epoch {self.epoch}: 随机选择batch {sorted(visualization_batches)} 进行可视化")
        else:
            # 如果总batch数小于3，则选择所有batch
            visualization_batches = set(range(max_batches))
            logging.info(f"🎲 Epoch {self.epoch}: 选择所有batch {sorted(visualization_batches)} 进行可视化")
        
        # 🔧 修复：将可视化计数器移到函数开始，确保在最后的总结中可以访问
        self.current_epoch_visualization_count = 0  # 跟踪已可视化的batch数量
        self.current_epoch_visualization_batches = visualization_batches

        for data_iter, batch in enumerate(val_loader):
            if data_iter >= limit_val_batches:
                break

            # measure data loading time
            data_time.update(time.time() - end)
            data_times.append(data_time.val)

            with torch.cuda.amp.autocast(enabled=False):
                batch = self._process_batch(batch)
            batch = copy_data_to_device(batch, self.device)

            # compute output
            with torch.no_grad():
                with torch.cuda.amp.autocast(
                    enabled=self.optim_conf.amp.enabled,
                    dtype=get_amp_type(self.optim_conf.amp.amp_dtype),
                ):
                    for phase, model in zip(curr_phases, curr_models):
                        with self._to_full_val_model(model):
                            predictions = self._step(
                                batch,
                                model,
                                phase,
                                loss_meters,
                            )
                            # 🎯 修改：对随机选择的batch进行可视化
                            if (data_iter in self.current_epoch_visualization_batches and 
                                hasattr(self, 'val_visualizer') and 
                                predictions is not None):
                                try:
                                    self.current_epoch_visualization_count += 1
                                    logging.info(f"🎨 开始可视化第{self.current_epoch_visualization_count}/3个选中batch (batch_idx={data_iter})")
                                    
                                    # 执行validation可视化
                                    self.val_visualizer.visualize_validation_results(
                                        predictions, batch, self.epoch, data_iter
                                    )
                                    logging.info(f"✅ 完成可视化batch {data_iter}")
                                except Exception as e:
                                    logging.warning(f"❌ Validation可视化失败 (batch {data_iter}): {e}")
                                    import traceback
                                    traceback.print_exc()
                                    exit(0)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            self.time_elapsed_meter.update(
                time.time() - self.start_time + self.ckpt_time_elapsed
            )

            if torch.cuda.is_available():
                mem.update(torch.cuda.max_memory_allocated() // 1e9)

            if data_iter % self.logging_conf.log_freq == 0:
                progress.display(data_iter)

        # 记录预估epoch时间
        self.est_epoch_time['val'] = batch_time.avg * iters_per_epoch
        
        # 简单记录数据加载时间 - 只使用已存在的方法
        if data_times and len(data_times) > 0:
            avg_data_time = sum(data_times) / len(data_times)
            logging.info(f"Average data loading time for val: {avg_data_time:.4f}s")

        for model in curr_models:
            with self._to_full_val_model(model):
                if hasattr(
                    unwrap_ddp_or_fsdp_if_wrapped(model), "on_validation_epoch_end"
                ):
                    unwrap_ddp_or_fsdp_if_wrapped(model).on_validation_epoch_end()

        # 简化输出字典构建 - 只使用已存在的方法
        out_dict = {}
        
        # 添加trainer状态
        for phase in curr_phases:
            out_dict.update(self._get_trainer_state(phase))

        # 添加loss值
        for k, v in loss_meters.items():
            out_dict[k] = v.avg
        
        # 增强validation输出，显示重要信息
        print("\n" + "🎯"*30 + " VALIDATION RESULTS " + "🎯"*30)
        print(f"📊 Epoch {self.epoch} Validation Summary:")
        
        # 显示Loss信息
        loss_items = []
        for key, value in out_dict.items():
            if "Loss" in key:
                loss_name = key.replace("Loss/val_", "").replace("loss_", "")
                loss_items.append(f"{loss_name}: {value:.6f}")
        
        if loss_items:
            print("📈 Loss Values:")
            for i, loss_str in enumerate(loss_items):
                if i % 3 == 0 and i > 0:  # 每3个换行
                    print()
                print(f"  {loss_str:<25}", end="")
            print()  # 最后换行
        
        # 🎯 修复：验证可视化总结
        if hasattr(self, 'val_visualizer') and hasattr(self, 'current_epoch_visualization_count'):
            print(f"🎨 可视化总结: 本epoch共可视化了 {self.current_epoch_visualization_count} 个随机选择的batch")
            if hasattr(self, 'current_epoch_visualization_batches'):
                print(f"   选中的batch索引: {sorted(self.current_epoch_visualization_batches)}")
        
        print("🎯" + "="*78 + "🎯")
        
        # 显示保存路径信息
        save_dir = getattr(self.checkpoint_conf, 'save_dir', 'logs/default/ckpts')
        print(f"💾 Checkpoint Save Dir: {save_dir}")
        if hasattr(self, 'val_visualizer') and self.val_visualizer.should_visualize:
            vis_dir = str(self.val_visualizer.val_vis_dir)
            print(f"📸 Validation Visuals: {vis_dir}")
        
        print("🎯" * 80 + "\n")
        
        logging.info(f"Meters: {out_dict}")
        return out_dict




    @contextlib.contextmanager
    def _to_full_val_model(self, model):
        # Simplified since we only support standard DDP models
        # No special handling needed for DDP models during validation
        yield

    def _get_trainer_state(self, phase):
        return {
            "Trainer/where": self.where,
            "Trainer/epoch": self.epoch,
            f"Trainer/steps_{phase}": self.steps[phase],
        }

    def _perform_early_validation(self):
        """
        Perform early validation for debugging purposes.
        Uses the same logic as regular validation but with limited batches.
        """
        if not self.val_dataset:
            logging.warning("No validation dataset available for early validation")
            return
        
        # 临时保存原始的limit_val_batches设置
        original_limit = self.limit_val_batches
        early_val_limit = self.early_validation_conf.get("limit_batches", 40)
        
        try:
            # 临时设置early validation的batch限制
            self.limit_val_batches = early_val_limit
            
            logging.info(f"🚀 Running early validation with {early_val_limit} batches")
            
            # 使用完全相同的validation逻辑
            dataloader = self.val_dataset.get_loader(epoch=int(self.epoch))
            dataloader.shuffle = True
            
            # 检查是否有is_fresh_epoch方法，如果没有则使用默认值
            if hasattr(self.val_dataset, 'is_fresh_epoch'):
                is_fresh_epoch = self.val_dataset.is_fresh_epoch(epoch=int(self.epoch))
            else:
                is_fresh_epoch = True  # 默认值，适用于early validation
            
            # 调用相同的val_epoch方法
            out_dict = self.val_epoch(dataloader, is_fresh_epoch)
            
            # 特殊的early validation输出格式
            print("\n" + "🚀"*20 + " EARLY VALIDATION " + "🚀"*20)
            print(f"📊 Training Step {self.steps['train']} Early Validation:")
            
            # 显示Loss信息
            loss_items = []
            for key, value in out_dict.items():
                if "Loss" in key:
                    loss_name = key.replace("Loss/val_", "").replace("loss_", "")
                    loss_items.append(f"{loss_name}: {value:.6f}")
            
            if loss_items:
                print("📈 Early Val Loss:")
                for i, loss_str in enumerate(loss_items):
                    if i % 3 == 0 and i > 0:  # 每3个换行
                        print()
                    print(f"  {loss_str:<25}", end="")
                print()  # 最后换行
            
            print("🚀" * 60 + "\n")
            
            # Log to tensorboard with special prefix
            for key, value in out_dict.items():
                if "Loss" in key:
                    early_key = key.replace("Loss/val_", "EarlyVal/")
                    self.tb_writer.log(early_key, value, self.steps['train'])
            
            # Clean up
            del dataloader
            gc.collect()
            torch.cuda.empty_cache()
            
        finally:
            # 恢复原始的limit_val_batches设置
            self.limit_val_batches = original_limit

    def train_epoch(self, train_loader):        
        batch_time = AverageMeter("Batch Time", self.device, ":.4f")
        data_time = AverageMeter("Data Time", self.device, ":.4f")
        mem = AverageMeter("Mem (GB)", self.device, ":.4f")
        data_times = []
        phase = 'train'
        
        loss_names = self._get_scalar_log_keys(phase)
        loss_names = [f"Loss/{phase}_{name}" for name in loss_names]
        loss_meters = {
            name: AverageMeter(name, self.device, ":.4f") for name in loss_names
        }
        
        for config in self.gradient_clipper.configs: 
            param_names = ",".join(config['module_names'])
            loss_meters[f"Grad/{param_names}"] = AverageMeter(f"Grad/{param_names}", self.device, ":.4f")


        progress = ProgressMeter(
            num_batches=len(train_loader),
            meters=[
                batch_time,
                data_time,
                mem,
                self.time_elapsed_meter,
                *loss_meters.values(),
            ],
            real_meters={},
            prefix="Train Epoch: [{}]".format(self.epoch),
        )

        self.model.train()
        end = time.time()

        iters_per_epoch = len(train_loader)
        limit_train_batches = (
            iters_per_epoch
            if self.limit_train_batches is None
            else self.limit_train_batches
        )
        
        if self.gradient_clipper is not None:
            # setup gradient clipping at the beginning of training
            self.gradient_clipper.setup_clipping(self.model)

        for data_iter, batch in enumerate(train_loader):
            if data_iter > limit_train_batches:
                break
            
            # measure data loading time
            data_time.update(time.time() - end)
            data_times.append(data_time.val)
            
            with torch.cuda.amp.autocast(enabled=False):
                batch = self._process_batch(batch)

            batch = copy_data_to_device(batch, self.device, non_blocking=True)

            accum_steps = self.accum_steps

            if accum_steps==1:
                chunked_batches = [batch]
            else:
                chunked_batches = chunk_batch_for_accum_steps(batch, accum_steps)

            self._run_steps_on_batch_chunks(
                chunked_batches, phase, loss_meters
            )

            # compute gradient and do SGD step
            assert data_iter <= limit_train_batches  # allow for off by one errors
            exact_epoch = self.epoch + float(data_iter) / limit_train_batches
            self.where = float(exact_epoch) / self.max_epochs
            
            assert self.where <= 1 + self.EPSILON
            if self.where < 1.0:
                for optim in self.optims:
                    optim.step_schedulers(self.where)
            else:
                logging.warning(
                    f"Skipping scheduler update since the training is at the end, i.e, {self.where} of [0,1]."
                )
                    
            # Log schedulers
            if self.steps[phase] % self.logging_conf.log_freq == 0:
                for i, optim in enumerate(self.optims):
                    for j, param_group in enumerate(optim.optimizer.param_groups):
                        for option in optim.schedulers[j]:
                            optim_prefix = (
                                f"{i}_"
                                if len(self.optims) > 1
                                else (
                                    "" + f"{j}_"
                                    if len(optim.optimizer.param_groups) > 1
                                    else ""
                                )
                            )
                            self.tb_writer.log(
                                os.path.join("Optim", f"{optim_prefix}", option),
                                param_group[option],
                                self.steps[phase],
                            )
                self.tb_writer.log(
                    os.path.join("Optim", "where"),
                    self.where,
                    self.steps[phase],
                )

            # Clipping gradients and detecting diverging gradients
            if self.gradient_clipper is not None:
                for optim in self.optims:
                    self.scaler.unscale_(optim.optimizer)

                grad_norm_dict = self.gradient_clipper(model=self.model)

                for key, grad_norm in grad_norm_dict.items():
                    loss_meters[f"Grad/{key}"].update(grad_norm)

            # Optimizer step
            for optim in self.optims:   
                self.scaler.step(optim.optimizer)
            self.scaler.update()

            # Measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            self.time_elapsed_meter.update(
                time.time() - self.start_time + self.ckpt_time_elapsed
            )
            mem.update(torch.cuda.max_memory_allocated() // 1e9)

            if data_iter % self.logging_conf.log_freq == 0:
                progress.display(data_iter)

        return True



    def _run_steps_on_batch_chunks(
        self,
        chunked_batches: List[Any],
        phase: str,
        loss_meters: Dict[str, AverageMeter],
    ):
        """
        Run the forward / backward as many times as there are chunks in the batch,
        accumulating the gradients on each backward
        """        
        
        for optim in self.optims:   
            optim.zero_grad(set_to_none=True)

        accum_steps = len(chunked_batches)

        amp_type = get_amp_type(self.optim_conf.amp.amp_dtype)
        
        for i, chunked_batch in enumerate(chunked_batches):
            ddp_context = (
                self.model.no_sync()
                if i < accum_steps - 1
                else contextlib.nullcontext()
            )

            with ddp_context:
                with torch.cuda.amp.autocast(
                    enabled=self.optim_conf.amp.enabled,
                    dtype=amp_type,
                ):
                    loss_dict = self._step(
                        chunked_batch, self.model, phase, loss_meters
                    )


                loss = loss_dict["objective"]
                loss_key = f"Loss/{phase}_loss_objective"
                batch_size = chunked_batch["images"].shape[0]

                if not math.isfinite(loss.item()):
                    error_msg = f"Loss is {loss.item()}, attempting to stop training"
                    logging.error(error_msg)
                    return

                loss /= accum_steps
                self.scaler.scale(loss).backward()
                loss_meters[loss_key].update(loss.item(), batch_size)



    def _reset_meters(self, phases: str) -> None:
        for meter in self._get_meters(phases).values():
            meter.reset()



    def _apply_batch_repetition(self, batch: Mapping) -> Mapping:
        tensor_keys = [
            "images", "depths", "extrinsics", "intrinsics", 
            "cam_points", "world_points", "point_masks", 
        ]        
        string_keys = ["seq_name"]
        
        for key in tensor_keys:
            if key in batch:
                original_tensor = batch[key]
                batch[key] = torch.concatenate([original_tensor, 
                                                torch.flip(original_tensor, dims=[1])], 
                                                dim=0)
        
        for key in string_keys:
            if key in batch:
                batch[key] = batch[key] + batch[key]
        
        return batch


    def _process_batch(self, batch: Mapping):      
        if self.data_conf.train.common_config.repeat_batch:
            batch = self._apply_batch_repetition(batch)
        if "world_points" in batch.keys():
            # Normalize camera extrinsics and points. The function returns new tensors.
            normalized_extrinsics, normalized_cam_points, normalized_world_points, normalized_depths = \
                normalize_camera_extrinsics_and_points_batch(
                    extrinsics=batch["extrinsics"],
                    cam_points=batch["cam_points"],
                    world_points=batch["world_points"],
                    depths=batch["depths"],
                    point_masks=batch["point_masks"],
                )
            batch["cam_points"] = normalized_cam_points
            batch["world_points"] = normalized_world_points
            batch["depths"] = normalized_depths
        else:
            normalized_extrinsics, normalized_cam_points, normalized_world_points, normalized_depths = \
                normalize_camera_extrinsics_and_points_batch(
                    extrinsics=batch["extrinsics"],
                    cam_points=None,
                    world_points=None,
                    depths=None,
                    point_masks=None,
                )
        # Replace the original values in the batch with the normalized ones.
        batch["extrinsics"] = normalized_extrinsics

        return batch
    
    
    
    def _step(
        self,
        batch,
        model: nn.Module,
        phase: str,
        loss_meters: dict[str, AverageMeter],
    ):
        # Step 1: Track evaluation (for both training and validation phases)
        track_pairs = None
        if "wrist_image" in batch and batch["wrist_image"] is not None:
            track_pairs = self._step1_track_evaluation_generic(batch, model)
        
        # Step 2: Normal forward pass (without track head)
        y_hat = model(images=batch["images"])
        
        # Step 3: Add track pairs to batch for loss computation
        if track_pairs is not None and len(track_pairs.get("wrist_uv", [])) > 0:
            batch["track_pairs"] = track_pairs
            if phase == "val":
                logging.info(f"✅ Validation: Found {len(track_pairs.get('wrist_uv', []))} track pairs")
        else:
            if phase == "val" and "wrist_image" in batch and batch["wrist_image"] is not None:
                logging.info(f"⚠️ Validation: No valid track pairs found (track_pairs={track_pairs})")
        
        # Compute the loss
        loss_dict = self.loss(y_hat, batch)
        
        # concatenate y_hat, loss_dict and batch for visualizations
        y_hat_batch = {**y_hat, **loss_dict, **batch}

        self._update_and_log_scalars(y_hat_batch, phase, self.steps[phase], loss_meters)

        self._log_tb_visuals(y_hat_batch, phase, self.steps[phase])

        self.steps[phase] += 1

        # Early validation check
        if (phase == "train" and 
            self.early_validation_conf is not None and 
            self.early_validation_conf.get("enabled", False) and
            self.steps[phase] == self.early_validation_conf.get("step", 5)):
            
            logging.info(f"🚀 Performing early validation at step {self.steps[phase]} for debugging")
            self._perform_early_validation()

        # 修复：validation阶段也应该返回包含loss的完整信息
        if phase == "val":
            # 返回包含predictions和loss的完整字典，用于可视化
            return {**y_hat, **loss_dict}
        else:
            return loss_dict
    
    def _step1_track_evaluation_generic(self, batch: Dict, model: nn.Module) -> Optional[Dict]:
        """
        通用多视角：使用 wrist + ext1..extN 进行track评估。
        """
        images = batch["images"]  # [B, S, 3, H, W]
        wrist_image = batch["wrist_image"]  # [B, H, W, 3]
        # Convert to [B, 3, H, W] float in [0,1]
        wrist_image = wrist_image.permute(0, 3, 1, 2)
        if wrist_image.dtype == torch.uint8:
            wrist_image = wrist_image.float() / 255.0
        wrist_image = torch.nn.functional.interpolate(wrist_image, size=images.shape[-2:], mode='bilinear', align_corners=False)

        # Stack wrist + all ext views
        stacked_images = torch.cat([wrist_image.unsqueeze(1), images], dim=1).contiguous()  # [B, S+1, 3, H, W]

        # Generate query points from wrist image (use first in batch for grid, then repeat across batch)
        query_points = self._generate_query_points(wrist_image[0])
        query_points = query_points.repeat(stacked_images.shape[0], 1, 1)

        # Temporarily disable wrist_head
        if hasattr(model, 'module'):
            original_wrist_head = model.module.wrist_head
            model.module.wrist_head = None
        else:
            original_wrist_head = model.wrist_head
            model.wrist_head = None

        with torch.no_grad():
            predictions = model(stacked_images, query_points=query_points)
            tracks = predictions["track"]   # [B, S+1, N, 2] (0=wrist, 1..S=exts)
            visibility = predictions["vis"] # [B, S+1, N]
            confidence = predictions["conf"] # [B, S+1, N]
            # print(confidence.max(),confidence.min(),confidence.mean())
            # print(visibility.max(),visibility.min(),visibility.mean())
            track_pairs = self._extract_multi_view_track_pairs(tracks, visibility, confidence)

        # Restore wrist_head
        if hasattr(model, 'module'):
            model.module.wrist_head = original_wrist_head
        else:
            model.wrist_head = original_wrist_head

        return track_pairs
    
        
    def _generate_query_points(self, wrist_image: torch.Tensor) -> torch.Tensor:
        """
        Generate query points from wrist image
        """
        H, W = wrist_image.shape[1:]

        # Dense grid sampling strategy - 确保生成256个点
        grid_size = 32
        h_coords = torch.linspace(grid_size, H - grid_size, grid_size)
        w_coords = torch.linspace(grid_size, W - grid_size, grid_size)
        
        h_grid, w_grid = torch.meshgrid(h_coords, w_coords, indexing='ij')
        query_points = torch.stack([w_grid.flatten(), h_grid.flatten()], dim=1)
        
        if query_points.dim() == 2:
            query_points = query_points.unsqueeze(0)  # [1, N, 2]
        
        # 验证形状
        assert query_points.shape == (1, grid_size**2, 2), f"Expected shape (1, {grid_size}**2, 2), got {query_points.shape}"
        
        return query_points.to(wrist_image.device)
    
    def _extract_track_pairs(self, tracks: torch.Tensor, visibility: torch.Tensor, confidence: torch.Tensor, batch: Dict = None) -> Dict:
        """
        Extract valid track pairs from track results
        Support wrist + ext1 OR wrist + ext2 pairs (not requiring all three views)
        
        🎯 新增：支持单视角训练模式
        - 如果batch中包含single_view_training信息，则根据selected_view生成对应的track pairs
        - 保持与双视角训练的兼容性
        
        FIXED: 修复了validation阶段track pairs为0的问题
        - 原因：之前只处理第一个batch (tracks[0, ...])，但validation batch size可能更大
        - 解决：改为遍历所有batch (for b in range(tracks.shape[0]))
        - 结果：validation阶段现在能正确找到track pairs并计算projection loss
        - 测试结果：validation batch size=24时，能找到45个track pairs，projection loss正常计算
        
        NEW: 添加batch_indices信息，支持多batch的projection loss计算
        """
        track_confidence_threshold = self.track_confidence_threshold  # 降低置信度阈值
        
        # 🎯 检查是否为单视角训练模式
        is_single_view = False
        selected_view = None
        if batch is not None and "single_view_training" in batch:
            is_single_view = batch["single_view_training"][0].item() if torch.is_tensor(batch["single_view_training"]) else batch["single_view_training"][0]
            if is_single_view and "selected_view" in batch:
                selected_view = batch["selected_view"][0] if isinstance(batch["selected_view"], list) else batch["selected_view"]
                logging.debug(f"Single view training mode detected, selected view: {selected_view}")
        
        if is_single_view:
            # 单视角：退化为仅与ext1配对
            return self._extract_single_view_track_pairs(tracks, visibility, confidence, batch)
        else:
            return self._extract_multi_view_track_pairs(tracks, visibility, confidence)
    
    def _extract_single_view_track_pairs(self, tracks: torch.Tensor, visibility: torch.Tensor, confidence: torch.Tensor, batch: Dict) -> Dict:
        """
        🎯 单视角训练模式：提取track pairs
        只处理wrist + ext的匹配（ext在第1位）
        """
        track_confidence_threshold = self.track_confidence_threshold
        
        # 🎯 在单视角模式下，ext视角总是在第1位
        ext_idx = 1  # ext视角总是在第1位
        
        # 获取选中的视角类型
        selected_view = batch.get("ext_view", "ext1")[0] if isinstance(batch.get("ext_view"), list) else batch.get("ext_view", "ext1")
        
        # 确定pair_type
        if selected_view == "ext1":
            pair_type = 0  # wrist + ext1
        elif selected_view == "ext2":
            pair_type = 1  # wrist + ext2
        else:
            logging.warning(f"Unknown selected view: {selected_view}, falling back to dual view mode")
            return self._extract_track_pairs(tracks, visibility, confidence, batch)
        
        # 检查wrist + ext的匹配
        wrist_ext_valid = (visibility[:, 0, :] > 0.5) & (visibility[:, 1, :] > 0.5) & \
                          (confidence[:, 0, :] > track_confidence_threshold) & (confidence[:, 1, :] > track_confidence_threshold)
        
        # 统计track pairs数量
        total_points = tracks.shape[2]
        wrist_ext_count = wrist_ext_valid.sum().item()
        
        logging.debug(f"Single view ({selected_view}) track pairs: {wrist_ext_count}/{total_points}")
        
        track_pairs = {
            "wrist_uv": [],
            "ext_uv": [],  # 🎯 统一的ext视角UV坐标
            "confidence": [],
            "pair_type": [],  # 0: wrist+ext1, 1: wrist+ext2
            "batch_indices": [],  # 记录每个track pair属于哪个batch
            "single_view_mode": True,  # 🎯 标记为单视角模式
            "ext_view": selected_view,  # 🎯 记录选中的视角
        }
        
        # 收集wrist + ext pairs (处理所有batch)
        for b in range(tracks.shape[0]):
            for i in range(tracks.shape[2]):
                if wrist_ext_valid[b, i]:
                    track_pairs["wrist_uv"].append(tracks[b, 0, i].cpu().float().numpy())  # wrist view (第0位)
                    track_pairs["ext_uv"].append(tracks[b, 1, i].cpu().float().numpy())   # ext view (第1位)
                    
                    track_pairs["confidence"].append((confidence[b, 0, i] + confidence[b, 1, i]) / 2)
                    track_pairs["pair_type"].append(pair_type)
                    track_pairs["batch_indices"].append(b)
        
        # Convert to numpy arrays
        for key in track_pairs:
            if key in ["single_view_mode", "ext_view"]:
                continue  # 跳过标记字段
            if track_pairs[key]:
                if isinstance(track_pairs[key], torch.Tensor):
                    track_pairs[key] = track_pairs[key].cpu().numpy()
                elif isinstance(track_pairs[key], list) and isinstance(track_pairs[key][0], torch.Tensor):
                    track_pairs[key] = [t.cpu().float().numpy() for t in track_pairs[key]]
                    track_pairs[key] = np.array(track_pairs[key])
                else:
                    track_pairs[key] = np.array(track_pairs[key])
            else:
                track_pairs[key] = np.array([])
        
        return track_pairs
    
    def _extract_multi_view_track_pairs(self, tracks: torch.Tensor, visibility: torch.Tensor, confidence: torch.Tensor) -> Dict:
        """
        通用多视角提取：从 wrist 与所有 ext_k (k>=1) 中提取匹配点对。
        返回统一数据结构：{"wrist_uv", "ext_uv", "confidence", "pair_type", "batch_indices"}
        其中 pair_type 为对应的ext索引（与batch["images"]中顺序一致，范围[0..S-1]）。
        """
        track_confidence_threshold = self.track_confidence_threshold
        B, S_plus_1, N, _ = tracks.shape
        result = {
            "wrist_uv": [],
            "ext_uv": [],
            "confidence": [],
            "pair_type": [],
            "batch_indices": [],
        }
        # For each ext view (index 1..S)
        for ext_idx in range(1, S_plus_1):
            valid = (visibility[:, 0, :] > 0.5) & (visibility[:, ext_idx, :] > 0.5) & \
                    (confidence[:, 0, :] > track_confidence_threshold) & (confidence[:, ext_idx, :] > track_confidence_threshold)
            # print(ext_idx,valid.sum(),visibility.shape,confidence.shape,visibility[:,ext_idx].mean(),confidence[:,ext_idx].mean())
            for b in range(B):
                for i in range(N):
                    if valid[b, i]:
                        result["wrist_uv"].append(tracks[b, 0, i].cpu().float().numpy())
                        result["ext_uv"].append(tracks[b, ext_idx, i].cpu().float().numpy())
                        result["confidence"].append(((confidence[b, 0, i] + confidence[b, ext_idx, i]) / 2).cpu())
                        # pair_type maps to ext index in batch["images"] (0-based)
                        result["pair_type"].append(ext_idx - 1)
                        result["batch_indices"].append(b)

        # Convert lists to numpy arrays (or empty arrays)
        for key in ["wrist_uv", "ext_uv"]:
            if len(result[key]) > 0:
                if isinstance(result[key][0], torch.Tensor):
                    result[key] = np.array([t.cpu().float().numpy() for t in result[key]])
                else:
                    result[key] = np.array(result[key])
            else:
                result[key] = np.array([])
        for key in ["confidence"]:
            if len(result[key]) > 0:
                result[key] = np.array([float(c) for c in result[key]])
            else:
                result[key] = np.array([])
        for key in ["pair_type", "batch_indices"]:
            if len(result[key]) > 0:
                result[key] = np.array(result[key])
            else:
                result[key] = np.array([])

        return result

    def _visualize_track_pairs(self, three_view_images: torch.Tensor, tracks: torch.Tensor, track_pairs: Dict):
        """
        可视化track pairs：三个视角放一起，track的点连起来（从wrist出发）
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as patches
        except ImportError:
            logging.warning("matplotlib not available, skipping track visualization")
            return
        
        # 只处理第一个batch
        images = three_view_images[0]  # [3, 3, H, W] - 3 views, 3 channels
        tracks_batch = tracks[0]  # [3, N, 2] - 3 views, N points, 2 coords
        
        # 转换为numpy并反归一化
        images_np = images.cpu().numpy()
        tracks_np = tracks_batch.cpu().numpy()
        
        # 创建图像网格
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        view_names = ['wrist', 'ext1', 'ext2']  # 调整顺序：wrist在第0位
        
        # 绘制每个视角的图像和track点
        for i, (ax, view_name) in enumerate(zip(axes, view_names)):
            # 显示图像
            img = images_np[i].transpose(1, 2, 0)  # [H, W, 3]
            img = np.clip(img, 0, 1)
            ax.imshow(img)
            ax.set_title(f'{view_name} view')
            ax.axis('off')
            
            # 绘制track点
            points = tracks_np[i]  # [N, 2]
            ax.scatter(points[:, 0], points[:, 1], c='red', s=10, alpha=0.7)
        
        # 绘制连接线（从wrist出发）
        if len(track_pairs.get("wrist_uv", [])) > 0:
            wrist_uv = track_pairs["wrist_uv"]
            pair_types = track_pairs["pair_type"]
            
            # 🎯 新的数据结构：使用统一的ext_uv字段
            if "ext_uv" in track_pairs:
                ext_uv = track_pairs["ext_uv"]
                
                # 为每个track pair绘制连接线
                for i, pair_type in enumerate(pair_types):
                    if pair_type == 0:  # wrist + ext1
                        # 在wrist和ext1之间画线
                        axes[0].plot([wrist_uv[i][0]], [wrist_uv[i][1]], 'go', markersize=8)  # wrist起点 (第0位)
                        axes[1].plot([ext_uv[i][0]], [ext_uv[i][1]], 'go', markersize=8)   # ext1终点 (第1位)
                        
                        # 添加连接线（用不同颜色表示）
                        color = 'yellow'
                        axes[0].plot([wrist_uv[i][0]], [wrist_uv[i][1]], 'o', color=color, markersize=6)
                        axes[1].plot([ext_uv[i][0]], [ext_uv[i][1]], 'o', color=color, markersize=6)
                    
                    elif pair_type == 1:  # wrist + ext2
                        # 在wrist和ext2之间画线
                        axes[0].plot([wrist_uv[i][0]], [wrist_uv[i][1]], 'bo', markersize=8)  # wrist起点 (第0位)
                        axes[2].plot([ext_uv[i][0]], [ext_uv[i][1]], 'bo', markersize=8)   # ext2终点 (第2位)
                        
                        # 添加连接线（用不同颜色表示）
                        color = 'cyan'
                        axes[0].plot([wrist_uv[i][0]], [wrist_uv[i][1]], 'o', color=color, markersize=6)
                        axes[2].plot([ext_uv[i][0]], [ext_uv[i][1]], 'o', color=color, markersize=6)
            else:
                # 双视角数据结构
                ext1_uv = track_pairs["ext1_uv"]
                ext2_uv = track_pairs["ext2_uv"]
                
                # 为每个track pair绘制连接线
                for i, pair_type in enumerate(pair_types):
                    if pair_type == 0:  # wrist + ext1
                        if ext1_uv[i][0] >= 0:  # 有效点
                            # 在wrist和ext1之间画线
                            axes[0].plot([wrist_uv[i][0]], [wrist_uv[i][1]], 'go', markersize=8)  # wrist起点 (第0位)
                            axes[1].plot([ext1_uv[i][0]], [ext1_uv[i][1]], 'go', markersize=8)   # ext1终点 (第1位)
                            
                            # 添加连接线（用不同颜色表示）
                            color = 'yellow'
                            axes[0].plot([wrist_uv[i][0]], [wrist_uv[i][1]], 'o', color=color, markersize=6)
                            axes[1].plot([ext1_uv[i][0]], [ext1_uv[i][1]], 'o', color=color, markersize=6)
                    
                    elif pair_type == 1:  # wrist + ext2
                        if ext2_uv[i][0] >= 0:  # 有效点
                            # 在wrist和ext2之间画线
                            axes[0].plot([wrist_uv[i][0]], [wrist_uv[i][1]], 'bo', markersize=8)  # wrist起点 (第0位)
                            axes[2].plot([ext2_uv[i][0]], [ext2_uv[i][1]], 'bo', markersize=8)   # ext2终点 (第2位)
                            
                            # 添加连接线（用不同颜色表示）
                            color = 'cyan'
                            axes[0].plot([wrist_uv[i][0]], [wrist_uv[i][1]], 'o', color=color, markersize=6)
                            axes[2].plot([ext2_uv[i][0]], [ext2_uv[i][1]], 'o', color=color, markersize=6)
        
        plt.tight_layout()
        plt.savefig('track_visualization.png')
        
        plt.close(fig)
        logging.info(f"🎨 Track visualization saved with {len(track_pairs.get('wrist_uv', []))} pairs")


    def _update_and_log_scalars(
        self,
        batch: Mapping,
        phase: str,
        step: int,
        loss_meters: dict[str, AverageMeter],
    ) -> None:
        keys_to_log = self._get_scalar_log_keys(phase)
        batch_size = batch['extrinsics'].shape[0]
        for key in keys_to_log:
            if key in batch:
                value = batch[key].item() if torch.is_tensor(batch[key]) else batch[key]
                loss_meters[f"Loss/{phase}_{key}"].update(value, batch_size)
                if step % self.logging_conf.log_freq == 0:
                    self.tb_writer.log(f"Values/{phase}/{key}", value, step)




    def _log_tb_visuals(self, batch: Mapping, phase: str, step: int) -> None:
        if not (
            self.logging_conf.log_visuals
            and (phase in self.logging_conf.log_visual_frequency)
            and self.logging_conf.log_visual_frequency[phase] > 0
            and (step % self.logging_conf.log_visual_frequency[phase] == 0)
            and (self.logging_conf.visuals_keys_to_log is not None)
        ):
            return

        if phase in self.logging_conf.visuals_keys_to_log:
            keys_to_log = self.logging_conf.visuals_keys_to_log[phase][
                "keys_to_log"
            ]
            assert (
                len(keys_to_log) > 0
            ), "Need to include some visual keys to log"
            modality = self.logging_conf.visuals_keys_to_log[phase][
                "modality"
            ]
            assert modality in [
                "image",
                "video",
            ], "Currently only support video or image logging"

            name = f"Visuals/{phase}"

            # 处理不同通道数的tensor，避免stack错误
            processed_visuals = []
            
            for key in keys_to_log:
                if key in batch and hasattr(batch[key], '__getitem__') and len(batch[key]) > 0:
                    try:
                        tensor = batch[key][0]
                        
                        if isinstance(tensor, torch.Tensor) and tensor.dim() >= 3:
                            # 将tensor转换为可视化格式
                            if tensor.dim() == 4 and tensor.shape[-1] in [1, 3]:  # (B, H, W, C) format
                                if tensor.shape[-1] == 1:
                                    # 单通道转为3通道（灰度图）
                                    tensor = tensor.repeat(1, 1, 1, 3)
                                tensor = tensor.permute(0, 3, 1, 2)  # 转为 (B, C, H, W)
                            elif tensor.dim() == 3:  
                                # 判断是 (C, H, W) 还是 (B, H, W) 格式
                                if tensor.shape[0] <= 16:  # 假设通道数不会超过16，batch size可能更大
                                    # (C, H, W) 格式处理
                                    if tensor.shape[0] == 1:
                                        # 单通道转为3通道
                                        tensor = tensor.repeat(3, 1, 1)
                                    elif tensor.shape[0] > 3:
                                        # 多通道取前3个
                                        tensor = tensor[:3]
                                    tensor = tensor.unsqueeze(0)  # 添加batch维度
                                else:
                                    # (B, H, W) 格式处理
                                    # 这是 (B, H, W) 格式，需要添加通道维度
                                    tensor = tensor.unsqueeze(1)  # 变成 (B, 1, H, W)
                                    # 扩展为3通道
                                    tensor = tensor.repeat(1, 3, 1, 1)
                            
                            # 确保tensor是3通道 - 关键修复！
                            if tensor.dim() == 4 and tensor.shape[1] != 3:
                                if tensor.shape[1] == 1:
                                    tensor = tensor.repeat(1, 3, 1, 1)
                                elif tensor.shape[1] == 2:
                                    # 2通道扩展为3通道：复制第一个通道
                                    third_channel = tensor[:, :1, :, :]  # 取第一个通道
                                    tensor = torch.cat([tensor, third_channel], dim=1)
                                elif tensor.shape[1] > 3:
                                    tensor = tensor[:, :3]
                            
                            # 归一化到[0,1]范围
                            min_val = tensor.min()
                            max_val = tensor.max()
                            tensor = (tensor - min_val) / (max_val - min_val + 1e-8)
                            
                            grid = torchvision.utils.make_grid(
                                tensor,
                                nrow=self.logging_conf.visuals_per_batch_to_log,
                            )
                            processed_visuals.append(grid)
                    except Exception as e:
                        logging.warning(f"跳过可视化key '{key}': {e}")
            
            if processed_visuals:
                # 调整所有grid到相同尺寸，避免stack错误
                if len(processed_visuals) > 1:
                    try:
                        # 找到最大尺寸
                        max_h = max(grid.shape[-2] for grid in processed_visuals)
                        max_w = max(grid.shape[-1] for grid in processed_visuals)
                        
                        # 将所有grid填充到相同尺寸
                        padded_grids = []
                        for grid in processed_visuals:
                            h, w = grid.shape[-2:]
                            pad_h = max_h - h
                            pad_w = max_w - w
                            
                            if pad_h > 0 or pad_w > 0:
                                # 使用零填充
                                grid = torch.nn.functional.pad(grid, (0, pad_w, 0, pad_h))
                            padded_grids.append(grid)
                        
                        visuals_to_log = torchvision.utils.make_grid(
                            padded_grids,
                            nrow=1,
                        ).clamp(0, 1)
                    except Exception as e:
                        logging.warning(f"Grid堆叠失败，使用第一个grid: {e}")
                        visuals_to_log = processed_visuals[0].clamp(0, 1)
                else:
                    visuals_to_log = processed_visuals[0].clamp(0, 1)
            else:
                return

            visuals_to_log = visuals_to_log.cpu()
            if visuals_to_log.dtype == torch.bfloat16:
                visuals_to_log = visuals_to_log.to(torch.float16)
            visuals_to_log = visuals_to_log.numpy()

            # 使用默认的video_logging_fps，避免配置缺失错误
            video_fps = getattr(self.logging_conf, 'video_logging_fps', 10)  # 默认10fps
            self.tb_writer.log_visuals(
                name, visuals_to_log, step, video_fps
            )

    def _log_sync_data_times(self, phase: str, data_times: List[float]) -> None:
        """
        Log synchronous data loading times.
        
        Args:
            phase: Phase name ('train' or 'val')
            data_times: List of data loading times
        """
        if data_times and len(data_times) > 0:
            avg_data_time = sum(data_times) / len(data_times)
            self.tb_writer.log(f"DataTime/{phase}_avg", avg_data_time, self.epoch)
            logging.info(f"Average data loading time for {phase}: {avg_data_time:.4f}s")

    def _log_meters_and_save_best_ckpts(self, phases):
        """
        Log meters and save best checkpoints.
        
        Args:
            phases: List of phases to process
            
        Returns:
            Dictionary containing logged metrics
        """
        out_dict = {}
        
        # Since meters is None (no meter tracking), return empty dict
        if self.meters is None:
            return out_dict
            
        # If meters were implemented, this would log and save best checkpoints
        # For now, just return empty dict to avoid crashes
        return out_dict


def chunk_batch_for_accum_steps(batch, accum_steps: int):
    return [get_chunk_from_data(batch, i, accum_steps) for i in range(accum_steps)]


def is_sequence_of_primitives(data):
    return (
        isinstance(data, Sequence)
        and not isinstance(data, str)
        and len(data) > 0
        and isinstance(data[0], (str, int, float, bool))
    )


def get_chunk_from_data(data, chunk_id, num_chunks):
    """
    Recursively splits all the tensors inside the passed data object into num_chunks.
    """
    if isinstance(data, torch.Tensor) or is_sequence_of_primitives(data):
        # either a tensor or a list of primitive objects
        # assert len(data) % num_chunks == 0
        start = (len(data) // num_chunks) * chunk_id
        end = (len(data) // num_chunks) * (chunk_id + 1)
        return data[start:end]
    elif isinstance(data, Mapping):
        return {
            key: get_chunk_from_data(value, chunk_id, num_chunks)
            for key, value in data.items()
        }
    elif isinstance(data, str):
        # NOTE: this is a hack to support string keys in the batch
        return data
    elif isinstance(data, Sequence):
        return [get_chunk_from_data(value, chunk_id, num_chunks) for value in data]
    else:
        return data

