# !/bin/bash
# dataset_name="trantor2nd/rheovla_dataset"
dataset_name="/home/img/project/lerobot/dynvla_dataset_lerobot"
# current_model_path="/home/img/project/lerobot/policies/diffusion_policy/train/100ksteps/checkpoints/100000/pretrained_model"
grad_acc=16
steps=100000
save_freq=50000


lerobot-train \
      --dataset.repo_id=$dataset_name \
      --policy.type=diffusion \
      --output_dir=diffusion_policy/train/100ksteps \
      --policy.device=cuda \
      --job_name="training" \
      --batch_size=16 \
      --wandb.enable=false \
      --policy.push_to_hub=false \
      --steps=$steps \
      --save_freq=$save_freq \
      --gradient_accumulation_steps=$grad_acc \
      --policy.crop_shape=[480,480] \
      --policy.noise_scheduler_type="DDIM" \
      --policy.num_train_timesteps=100 \
      --policy.num_inference_steps=10 
      # --policy.pretrained_path=$current_model_path 
