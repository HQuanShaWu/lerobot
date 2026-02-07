#!/bin/bash
# =============================================================================
# SmolVLA policy deployment launcher
# =============================================================================

trap cleanup INT TERM

# ----------------------------- Configuration ----------------------------- #
CONDA_ENV="py310"
PIPER_WS="/home/data/Project/piper_ros"
DEPLOY_WS="/home/data/Project/policies/smolvla"
ORACLE_WS="/home/data/Project/policies/smolvla"
CAN_PORT="can0"
GRIPPER_PORT="/dev/ttyUSB0"
CAN_ACTIVATE_SCRIPT="$PIPER_WS/can_activate.sh"
PIPER_LOG="$DEPLOY_WS/piper.log"

# [修改点] 指向新的 python 脚本
DEPLOY_SCRIPT="$ORACLE_WS/deploy_smolvla.py"

# [修改点] 修改默认 Checkpoint 路径和任务文本
CHECKPOINT="${CHECKPOINT:-/home/img/project/lerobot/policies/smolvla/train/100ksteps/checkpoints/100000/pretrained_model}"
# 这里的任务文本必须和训练时的一致，或者能够触发相应动作
TASK_TEXT="${TASK_TEXT:-put_the_gel_pen_on_the_stationery_organizer}"

DEVICE="${DEVICE:-cuda:0}"
EXEC_RATE="${EXEC_RATE:-15.0}" # SmolVLA 推理较快，可以尝试更高频率
FISHEYE_INDEX="${FISHEYE_INDEX:-6}"
REALSENSE_SERIAL="${REALSENSE_SERIAL:-230322275684}"
# SmolVLA 默认步数较少 (Flow Matching)，通常 10 步足够
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-10}"

# >>> Use Conda Python for ROS2 >>>
export PYTHON_EXECUTABLE=/home/hsb/miniforge3/envs/py310/bin/python
export PYTHONPATH=/home/hsb/miniforge3/envs/py310/lib/python3.10/site-packages:$PYTHONPATH
# <<< Use Conda Python for ROS2 <<<

PIPER_PID=""
DEPLOY_PID=""

sudo chmod 666 "$GRIPPER_PORT" >/dev/null 2>&1 || true

# ----------------------------- Cleanup logic ----------------------------- #
cleanup() {
    echo ""
    echo "[ACTION] 捕获退出信号，正在执行安全退出..."

    timeout 2 ros2 topic pub --once /enable_flag std_msgs/msg/Bool "data: false" >/dev/null 2>&1 || true
    timeout 3 ros2 service call /enable_srv piper_msgs/srv/Enable "enable_request: false" >/dev/null 2>&1 || true

    for pid in $DEPLOY_PID $PIPER_PID; do
        if [ -n "$pid" ] && ps -p "$pid" >/dev/null 2>&1; then
            kill -9 "$pid" 2>/dev/null || true
        fi
    done

    pkill -9 -f "piper_single_ctrl" 2>/dev/null || true
    sudo ifconfig "$CAN_PORT" down 2>/dev/null || true

    wait $DEPLOY_PID $PIPER_PID 2>/dev/null || true

    echo "[INFO] ✅ SmolVLA 部署已安全退出"
    exit 0
}

# ---------------------------- Environment setup ------------------------- #
echo "[INFO] 激活 conda 环境: $CONDA_ENV"
source ~/miniforge3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"

echo "[INFO] 加载 ROS2 Humble 与 Piper 环境"
source /opt/ros/humble/setup.bash
source "$PIPER_WS/install/setup.bash"

echo "[INFO] 清理旧 Piper 控制节点..."
pkill -9 -f "piper_single_ctrl" 2>/dev/null || true
sleep 1

# --------------------------- Activate CAN bus --------------------------- #
echo "[INFO] 激活 CAN 总线: $CAN_PORT"
if [ -f "$CAN_ACTIVATE_SCRIPT" ]; then
    bash "$CAN_ACTIVATE_SCRIPT" "$CAN_PORT" 1000000
else
    echo "[ERROR] 未找到 CAN 激活脚本: $CAN_ACTIVATE_SCRIPT"
    exit 1
fi
sleep 2

# --------------------------- Start Piper control ------------------------ #
echo "[INFO] 启动 Piper 控制节点..."
ros2 run piper piper_single_ctrl --ros-args \
    -p can_port:="$CAN_PORT" \
    -p auto_enable:=true \
    -p gripper_exist:=true \
    -p gripper_val_mutiple:=2 \
    --log-level WARN >"$PIPER_LOG" 2>&1 &
PIPER_PID=$!
sleep 2

if ! ps -p "$PIPER_PID" >/dev/null 2>&1; then
    echo "[ERROR] Piper 控制节点启动失败，请检查日志 $PIPER_LOG"
    cleanup
fi

# --------------------------- Launch SmolVLA policy ----------------------- #
if [ ! -f "$DEPLOY_SCRIPT" ]; then
    echo "[ERROR] 未找到部署脚本 $DEPLOY_SCRIPT"
    cleanup
fi

echo "[INFO] 启动 SmolVLA 实时控制..."
DEPLOY_CMD=(
    python3 -u "$DEPLOY_SCRIPT"
    --checkpoint "$CHECKPOINT"
    --device "$DEVICE"
    --task-text "$TASK_TEXT"
    --rate "$EXEC_RATE"
    --gripper-port "$GRIPPER_PORT"
    --fisheye-index "$FISHEYE_INDEX"
    --num-inference-steps "$NUM_INFERENCE_STEPS"
)

if [ -n "$REALSENSE_SERIAL" ]; then
    DEPLOY_CMD+=(--realsense-serial "$REALSENSE_SERIAL")
fi

"${DEPLOY_CMD[@]}" &
DEPLOY_PID=$!

echo "[INFO] ✅ 所有节点已启动。按 Ctrl+C 退出。"

wait $DEPLOY_PID
cleanup