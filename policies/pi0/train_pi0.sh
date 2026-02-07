# !/bin/bash
# dataset_name="trantor2nd/rheovla_dataset"
dataset_name="/home/img/project/lerobot/dynvla_dataset_lerobot"
# current_model_path="/home/img/project/lerobot/policies/smolvla/train/100ksteps/checkpoints/100000/pretrained_model"
grad_acc=4
steps=200000
save_freq=200000
export HF_TOKEN=


# xxx

python src/lerobot/scripts/lerobot_train.py \
    --dataset.repo_id=${dataset_name} \
    --policy.type=pi0 \
    --output_dir=/policies/pi0/train \
    --job_name=pi0_training \
    --policy.pretrained_path=lerobot/pi0_base \
    --policy.push_to_hub=False \
    --policy.compile_model=false \
    --policy.gradient_checkpointing=true \
    --policy.dtype=bfloat16 \
    --steps=${steps} \
    --policy.device=cuda \
    --batch_size=1 \
    --gradient_accumulation_steps=$grad_acc \