#!/bin/bash
NUM_GPUS=1
OUTPUT_DIR="output/rheovla/codebook_finetune"
BATCH_SIZE=4
NUM_STEPS=100000
SAVE_FREQ=20000
LOG_FREQ=100
REPO_ID="nvidia/GR00T-N1.5-3B"
DATASET_ID="trantor2nd/rheovla_dataset"
JOB_NAME="codebook"

lerobot-train \
  --output_dir=$OUTPUT_DIR \
  --save_checkpoint=true \
  --batch_size=$BATCH_SIZE \
  --steps=$NUM_STEPS \
  --save_freq=$SAVE_FREQ \
  --log_freq=$LOG_FREQ \
  --policy.push_to_hub=false \
  --policy.type=rheovla \
  --policy.repo_id=$REPO_ID \
  --policy.tune_diffusion_model=false \
  --dataset.repo_id=$DATASET_ID \
  --wandb.enable=true \
  --wandb.disable_artifact=true \
  --job_name=$JOB_NAME 