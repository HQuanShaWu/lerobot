# !/bin/bash
# dataset_name="trantor2nd/rheovla_dataset"
dataset_name="/home/img/project/lerobot/dynvla_dataset_lerobot"
# current_model_path="/home/img/project/lerobot/policies/smolvla/train/100ksteps/checkpoints/100000/pretrained_model"
grad_acc=4
steps=40000
save_freq=40000


python src/lerobot/scripts/lerobot_train.py\
    --dataset.repo_id=${dataset_name} \
    --policy.type=pi05 \
    --output_dir=policies/pi05/train \
    --job_name=pi05_training \
    --policy.push_to_hub=False \
    --policy.pretrained_path=lerobot/pi05_base \
    --policy.compile_model=true \
    --policy.gradient_checkpointing=true \
    --wandb.enable=false \
    --policy.dtype=bfloat16 \
    --steps=${steps} \
    --policy.device=cuda \
    --batch_size=16