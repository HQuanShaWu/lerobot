delta_steps=100000
GRAD_ACCUM_STEPS=16
BATCH_SIZE=16
this_output_dir="efficientvla/train/division_internvl_1b_pretrain_mid_scale_horizon16_lora_rank32"
dataset_name="trantor2nd/rheovla_dataset"
# current_model_path="/home/img/project/lerobot/efficientvla/train/division_internvl_1b_pretrain_mid_scale_horizon16_lora_rank32_/checkpoints/000001/pretrained_model"
lerobot-train \
    --dataset.repo_id="$dataset_name" \
    --output_dir="$this_output_dir" \
    --job_name="train&internvl_1b" \
    --policy.type=efficientvla \
    --policy.device=cuda \
    --policy.lora_rank=32 \
    --policy.lora_alpha=64 \
    --policy.scale="medium" \
    --wandb.enable=false \
    --policy.push_to_hub=false \
    --batch_size=$BATCH_SIZE \
    --steps=${delta_steps} \
    --policy.training_steps=${delta_steps} \
    --save_freq=1 \
    --gradient_accumulation_steps=$GRAD_ACCUM_STEPS \
    --dataset.video_backend="torchcodec" \
    # --policy.pretrained_path="$current_model_path" \
    # --policy.repo_id="lerobot/pusht" 
    