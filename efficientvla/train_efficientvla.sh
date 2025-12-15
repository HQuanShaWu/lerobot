export TOKENIZERS_PARALLELISM="false"

training_steps=100000

lerobot-train \
  --dataset.repo_id=trantor2nd/rheovla_dataset \
  --policy.type=efficientvla \
  --policy.device=cuda \
  --policy.repo_id=lerobot/diffusion_pusht \
  --policy.lora_rank=8 \
  --policy.scale="medium" \
  --output_dir=efficientvla/train/midscale_horizon16_20ksteps_lora_rank8_backbone \
  --job_name=train_efficientvla \
  --wandb.enable=false \
  --batch_size=3 \
  --steps=${training_steps} \
  --policy.training_steps=${training_steps} \
  # --use_policy_training_preset=false \
