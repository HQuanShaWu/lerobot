# !/bin/bash
# 根据数据集长度自适应增加全局步数
set -e
export TOKENIZERS_PARALLELISM="false"
export PYTHONWARNINGS="ignore:The video decoding and encoding capabilities of torchvision are deprecated"

datasets=(
    "lerobot/ucsd_pick_and_place_dataset"  # ok
    "lerobot/stanford_kuka_multimodal_dataset"  # ok
    "lerobot/jaco_play"  # ok
    "trantor2nd/rheovla_dataset"  # ok
    # "lerobot/taco_play"  # use dataset.video_backend="pyav" and resize image to 224,224, ok
    # "lerobot/toto"       # dataset video time error
    # "lerobot/stanford_robocook"  # dataset video time error
    # "lerobot/utaustin_mutex"  # dataset video time error
    # "lerobot/stanford_hydra_dataset"  # dataset video time error
    # "lerobot/berkeley_autolab_ur5" # dataset video time error
    #  more...
)

NUM_EPOCHS=5
BATCH_SIZE=4
GRAD_ACCUM_STEPS=16

# 这是一个累积计数器，仅用于日志显示，不传给训练命令
total_accumulated_steps=180000 

current_model_path="/home/img/project/lerobot/policies/efficientvla/train/mid_scale_horizon16_180ksteps_lora_rank32/checkpoints/180000/pretrained_model"
base_output_dir="policies/efficientvla/train/pretrain_mid_scale_horizon16_lora_rank32"

i=1

get_frames_cmd='
import sys
from lerobot.datasets.lerobot_dataset import LeRobotDataset
try:
    ds = LeRobotDataset(sys.argv[1])
    print(len(ds))
except Exception as e:
    print(0)
'

for dataset_name in "${datasets[@]}"; do

    echo "--------------------------------------------------"
    echo "正在分析数据集: $dataset_name ..."
    num_frames=$(python -c "$get_frames_cmd" "$dataset_name")
    
    if [ "$num_frames" -eq 0 ]; then
        echo "❌ [ERROR] 无法获取数据集大小或数据集为空: $dataset_name"
        exit 1
    fi
    
    # 计算本次需要训练的增量步数 (Delta Steps)
    delta_steps=$(python -c "print(int(($num_frames * $NUM_EPOCHS) / ($BATCH_SIZE)))")
    
    # 设置一个最小步数，防止因为数据集太小导致步数过少
    if [ "$delta_steps" -lt 100 ]; then
        delta_steps=100
    fi

    if [ "$dataset_name" == "trantor2nd/rheovla_dataset" ]; then
        delta_steps=200000
    fi
    
    # 仅用于日志显示的累计步数
    total_accumulated_steps=$((total_accumulated_steps + delta_steps))

    seq_num=$(printf "%02d" $i)
    safe_name=${dataset_name//\//_}
    this_output_dir="${base_output_dir}_add${seq_num}_${safe_name}"

    echo "▶ 序列号: $seq_num"
    echo "▶ 数据集帧数: $num_frames"
    echo "▶ 本轮训练时长 (Delta Steps): $delta_steps"
    echo "▶ 预计训练后总累计步数 (Log only): $total_accumulated_steps"
    echo "--------------------------------------------------"

    # 修改说明：
    # 1. --steps 使用 delta_steps (训练时长)
    # 2. --save_freq 使用 delta_steps (跑完保存)
    # 3. --policy.training_steps 如果是用来定义scheduler的总长度，也应该匹配当前训练时长
    
    lerobot-train \
      --dataset.repo_id="$dataset_name" \
      --policy.pretrained_path="$current_model_path" \
      --output_dir="$this_output_dir" \
      --job_name="train_${seq_num}_${safe_name}" \
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
      --save_freq=${delta_steps} \
      --gradient_accumulation_steps=$GRAD_ACCUM_STEPS \
      --dataset.video_backend="torchcodec" \
      --policy.repo_id="$current_model_path" 

    # 路径检查逻辑更新：
    # 因为是加载 pretrained_model 权重而非 resume 状态，训练器内部步数可能从 0 开始。
    # 所以检查点目录名应该是 delta_steps 的数值。
    
    if [ -d "$this_output_dir/checkpoints/last/pretrained_model" ]; then
        current_model_path="$this_output_dir/checkpoints/last/pretrained_model"
    elif [ -d "$this_output_dir/checkpoints/${delta_steps}/pretrained_model" ]; then
        current_model_path="$this_output_dir/checkpoints/${delta_steps}/pretrained_model"
    else
        echo "❌ [ERROR] Checkpoint 未生成，停止脚本。"
        echo "检查目录: $this_output_dir/checkpoints/"
        # 此时可以打印一下目录结构以便调试
        ls -R "$this_output_dir/checkpoints/"
        exit 1
    fi

    echo "✅ 序列 $seq_num 完成。模型更新为: $current_model_path"
    echo "等待 5 秒开始下一个数据集..."
    sleep 5

    i=$((i+1))

done

echo "🎉 所有数据集训练完成！"