## WristWorld — Stage 1: Reconstruction

This folder contains launch commands to train or finetune VGGT-based reconstruction on several datasets/configurations. Commands are provided as-is and assume your environment follows the official VGGT training setup.

## Installation

Please follow the official VGGT training branch for installation, environment setup, and dependencies. We align with their instructions:

- Installation guide: [VGGT (training branch)](https://github.com/facebookresearch/vggt/tree/training)

If you are in regions with limited Hugging Face bandwidth, you may optionally set a mirror endpoint before running:

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

## System Packages (Ubuntu)

```bash
apt-get update
apt-get install -y libgl1
```

## Training Commands

All commands assume you are at the repository root and want `PYTHONPATH` to include the current working directory.

```bash
# droid training
PYTHONPATH=$(pwd) torchrun --nproc-per-node 8 training/launch.py --config_file vggt_droid_camera_only_finetune.yaml
# singleview
PYTHONPATH=$(pwd) torchrun --nproc-per-node 8 training/launch.py --config_file vggt_droid_camera_only_finetune_single_view.yaml

# calvin
PYTHONPATH=$(pwd) torchrun --nproc-per-node 8 training/launch.py --config_file vggt_calvin.yaml
# realbot
PYTHONPATH=$(pwd) torchrun --nproc-per-node 8 training/launch.py --config_file vggt_realbot.yaml

# Ablation w/o projection loss
PYTHONPATH=$(pwd) torchrun --nproc-per-node 8 training/launch.py --config_file vggt_droid_camera_only_finetune_no_projection_loss.yaml

# singleview
PYTHONPATH=$(pwd) torchrun --nproc-per-node 8 training/launch.py --config_file vggt_realbot_singleview.yaml
```

Notes:
- Adjust `--nproc-per-node` to match your available GPUs.
- Ensure the referenced YAML files exist and contain correct dataset paths and hyperparameters.

## Quick Start (example)

Example: run DROID camera-only finetuning with 8 GPUs:

```bash
export HF_ENDPOINT=https://hf-mirror.com
apt-get update && apt-get install -y libgl1
PYTHONPATH=$(pwd) torchrun --nproc-per-node 8 training/launch.py --config_file vggt_droid_camera_only_finetune.yaml
```

For more details, refer to the official docs and examples in the VGGT training branch: [VGGT (training branch)](https://github.com/facebookresearch/vggt/tree/training).
export HF_ENDPOINT=https://hf-mirror.com