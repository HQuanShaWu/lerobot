# !/bin/bash
# dataset_name="trantor2nd/rheovla_dataset"
dataset_name="/home/img/project/lerobot/dynvla_dataset_lerobot"
# current_model_path="/home/img/project/lerobot/policies/smolvla/train/100ksteps/checkpoints/100000/pretrained_model"
grad_acc=4
steps=25000
save_freq=25000


lerobot-train \
  --policy.type=groot \
  --dataset.repo_id=${dataset_name} \
  --batch_size=32 \
  --steps=${steps} \
  --save_freq=${save_freq} \
  --output_dir=policies/gr00t/train/100ksteps_batch1 \
  --job_name=training \
  --policy.device=cuda \
  --wandb.enable=false \
  --policy.push_to_hub=false \
  --policy.tune_diffusion_model=false