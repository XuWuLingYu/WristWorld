## Step 1. Installation
Follow the official DiffSynth Studio installation guide: [DiffSynth Studio](https://github.com/modelscope/DiffSynth-Studio)

## Step 2. Download Pretrained Weights
Download the pretrained checkpoints from [WristWorld](https://huggingface.co/XuWuLingYu/WristWorld) to `../checkpoints/` so the directory looks like:

```
checkpoints/
├─ BaseModel/
├─ VGGT/
├─ VideoModel/
└─ README.md
```

## Step 3. Run Inference
Run the following command:

```bash
python examples/wanvideo/rgb_ext_to_gen_video_droid.py \
  --input_root ../examples/ \
  --vggt_checkpoint ../checkpoints/VGGT/checkpoint.pt \
  --image_encoder_path /mnt/fck/huggingface/models--Wan-AI--Wan2.1-I2V-14B-480P/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth \
  --text_encoder_path /mnt/fck/huggingface/models--Wan-AI--Wan2.1-I2V-14B-480P/models_t5_umt5-xxl-enc-bf16.pth \
  --vae_path /mnt/fck/huggingface/models--Wan-AI--Wan2.1-I2V-14B-480P/Wan2.1_VAE.pth \
  --pretrained_lora_path ../checkpoints/VideoModel/video_dit_lora.safetensors \
  --gpus 0
```
