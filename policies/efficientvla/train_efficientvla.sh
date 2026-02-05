# !/bin/bash
set -e
export TOKENIZERS_PARALLELISM="false"

dataset_name="trantor2nd/rheovla_dataset"
current_model_path="/home/img/project/lerobot/policies/efficientvla/train/pretrain_mid_scale_horizon16_lora_rank32_add04_trantor2nd_rheovla_dataset/checkpoints/200000/pretrained_model"
grad_acc=16
steps=60000


lerobot-train \
      --dataset.repo_id=$dataset_name \
      --policy.pretrained_path=$current_model_path \
      --output_dir="policies/efficientvla/train/debug_finetune_mid_scale_horizon16_lora_rank64" \
      --job_name="training" \
      --policy.type=efficientvla \
      --policy.device=cuda \
      --policy.lora_rank=32 \
      --policy.lora_alpha=64 \
      --policy.scale="medium" \
      --wandb.enable=false \
      --policy.push_to_hub=false \
      --batch_size=4 \
      --steps=$steps \
      --save_freq=$steps \
      --policy.training_steps=$steps \
      --dataset.video_backend="torchcodec" \
      --gradient_accumulation_steps=$grad_acc \
      --policy.gradient_accumulation_steps=$grad_acc \
      # --policy.repo_id=$current_model_path \