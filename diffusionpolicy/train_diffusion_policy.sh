lerobot-train \
  --dataset.repo_id=trantor2nd/rheovla_dataset \
  --policy.type=diffusion \
  --output_dir=outputs/train/train_diffusion_policy_rehovla_dataset_200ksteps \
  --job_name=train_diffusion_policy_rehovla_dataset \
  --policy.device=cuda \
  --wandb.enable=true \
  --policy.repo_id=lerobot/diffusion_pusht \
  --batch_size=16 \
  --steps=200000 \
  --save_freq=80000 \
  # --use_policy_training_preset=false \
