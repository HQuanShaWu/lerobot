export TOKENIZERS_PARALLELISM="false"

training_steps=100000

lerobot-train \
  --dataset.repo_id=trantor2nd/rheovla_dataset \
  --policy.type=efficientvla \
  --output_dir=efficientvla/train/rehovla_dataset_acthorizon32_200ksteps \
  --job_name=train_diffusion_policy_rehovla_dataset \
  --policy.device=cuda \
  --wandb.enable=false \
  --policy.repo_id=lerobot/diffusion_pusht \
  --batch_size=4 \
  --steps=${training_steps} \
  --policy.training_steps=${training_steps} \
  --save_freq=50000 \
  # --use_policy_training_preset=false \
