# !/bin/bash
# dataset_name="trantor2nd/rheovla_dataset"
dataset_name="/home/img/project/lerobot/dynvla_dataset_lerobot"
# current_model_path="/home/img/project/lerobot/policies/smolvla/train/100ksteps/checkpoints/100000/pretrained_model"
grad_acc=4
steps=40000
save_freq=40000


lerobot-train \
  --policy.path=jolch/piper_smolvla_0110 \
  --dataset.repo_id=${dataset_name} \
  --batch_size=32 \
  --steps=${steps} \
  --output_dir=policies/smolvla/train/pretrain_jolch_piper_smolvla_0110_40ksteps_batch16 \
  --job_name=training \
  --policy.device=cuda \
  --wandb.enable=false \
  --policy.push_to_hub=false \
  --rename_map='{"observation.images.fisheye_rgb": "observation.images.camera1", "observation.images.realsense_rgb": "observation.images.camera2"}'
  #   --save_freq=${save_freq} \


# lerobot-train \
#       --dataset.repo_id=$dataset_name \
#       --policy.path= \
#       --policy.type=smolvla \
#       --output_dir=policies/smolvla/train/40ksteps_batch16_2.2B \
#       --policy.vlm_model_name="HuggingFaceTB/SmolVLM2-2.2B-Instruct" \
#       --policy.load_vlm_weights=true \
#       --policy.device=cuda \
#       --job_name="training" \
#       --batch_size=16 \
#       --wandb.enable=false \
#       --policy.push_to_hub=false \
#       --steps=$steps \
#       --save_freq=$save_freq \
#       --gradient_accumulation_steps=$grad_acc \
      # --policy.pretrained_path=$current_model_path 
