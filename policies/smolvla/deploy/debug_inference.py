#!/usr/bin/env python3
import torch
import json
import numpy as np
from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.processor import PolicyProcessorPipeline
from lerobot.processor.converters import policy_action_to_transition

# ================= 配置区域 =================
DATASET_PATH = "/home/img/project/lerobot/dynvla_dataset_lerobot"
CHECKPOINT_PATH = "/home/img/project/lerobot/policies/smolvla/train/pretrain_1Nono1_piper_placecup_smolvla_v2_40ksteps_batch16/checkpoints/040000/pretrained_model"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TEST_SAMPLES = 50  # 修改点2: 测试帧数增加到 50
# ===========================================

def load_deployment_model(checkpoint_dir):
    ckpt_path = Path(checkpoint_dir)
    
    # 1. Load Config
    with open(ckpt_path / "config.json", "r") as f:
        cfg_dict = json.load(f)
    cfg_dict.pop("type", None) 
    
    from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
    
    def _to_feature(feat_dict: dict) -> PolicyFeature:
        return PolicyFeature(type=FeatureType[feat_dict["type"]], shape=tuple(feat_dict.get("shape") or ()))

    if "input_features" in cfg_dict:
        cfg_dict["input_features"] = {k: _to_feature(v) for k, v in cfg_dict["input_features"].items()}
    if "output_features" in cfg_dict:
        cfg_dict["output_features"] = {k: _to_feature(v) for k, v in cfg_dict["output_features"].items()}
    if "normalization_mapping" in cfg_dict:
        cfg_dict["normalization_mapping"] = {
            k: NormalizationMode[v] for k, v in cfg_dict["normalization_mapping"].items()
        }

    cfg = SmolVLAConfig(**cfg_dict)
    cfg.device = DEVICE
    
    # 2. Load Processors
    overrides = {"device_processor": {"device": str(DEVICE)}}
    preproc = PolicyProcessorPipeline.from_pretrained(
        pretrained_model_name_or_path=ckpt_path,
        config_filename="policy_preprocessor.json",
        overrides=overrides,
    )
    postproc = PolicyProcessorPipeline.from_pretrained(
        pretrained_model_name_or_path=ckpt_path,
        config_filename="policy_postprocessor.json",
        overrides={"device_processor": {"device": "cpu"}},
        to_transition=policy_action_to_transition,
    )

    # 3. Load Policy
    print(f"Loading SmolVLAPolicy from {ckpt_path}...")
    policy = SmolVLAPolicy.from_pretrained(
        pretrained_name_or_path=str(ckpt_path),
        config=cfg,
    )
    policy.eval()
    
    return policy, preproc, postproc

def main():
    print(f"Loading model from: {CHECKPOINT_PATH}")
    policy, preproc, postproc = load_deployment_model(CHECKPOINT_PATH)
    policy.reset()
    
    print(f"Loading dataset from: {DATASET_PATH}")
    dataset = LeRobotDataset(root=DATASET_PATH, repo_id="local/debug_dataset") 

    # 修改点2: 随机采样更多帧数
    total_len = len(dataset)
    indices = np.random.choice(total_len, min(TEST_SAMPLES, total_len), replace=False)
    indices = np.sort(indices) # 排序方便观察
    print(f"Selected {len(indices)} samples for testing...")

    # 用于统计 Metrics
    arm_mses = []      # 存放每一帧的 Arm MSE (前6维)
    gripper_mses = []  # 存放每一帧的 Gripper MSE (第7维)
    
    for i, idx in enumerate(indices):
        item = dataset[idx]
        
        # === 构造输入 ===
        obs = {}
        if "observation.state" in item:
            obs["observation.state"] = item["observation.state"]
        for key in item.keys():
            if "observation.images" in key:
                obs[key] = item[key]
        
        # 处理 Task 文本
        if "task" in item:
            obs["task"] = item["task"]
        elif "language_instruction" in item:
            obs["task"] = item["language_instruction"]
        else:
            obs["task"] = "pick up the object" 

        # === 推理 ===
        try:
            model_input = preproc(obs)
        except Exception as e:
            print(f"Skipping index {idx}: Preprocessing failed ({e})")
            continue

        with torch.no_grad():
            policy.reset() 
            action_out = policy.select_action(model_input)
            
        result = postproc(action_out)
        pred_action = result["action"]
        gt_action = item["action"]
        
        # === 数据转换 ===
        pred_np = pred_action.squeeze().cpu().numpy()
        gt_np = gt_action.cpu().numpy()
        
        # === 修改点1: 分离 Arm (前6维) 和 Gripper (第7维) ===
        # 假设 Action 维度至少是 7
        if len(pred_np) >= 7:
            # 1. Arm Metrics (前 6 个关节)
            pred_arm = pred_np[:6]
            gt_arm = gt_np[:6]
            # 计算这一帧的 Arm MSE
            curr_arm_mse = np.mean((pred_arm - gt_arm) ** 2)
            arm_mses.append(curr_arm_mse)

            # 2. Gripper Metrics (第 7 个维度及之后)
            pred_gripper = pred_np[6:]
            gt_gripper = gt_np[6:]
            # 计算这一帧的 Gripper MSE
            curr_gripper_mse = np.mean((pred_gripper - gt_gripper) ** 2)
            gripper_mses.append(curr_gripper_mse)
        else:
            # 维度不足 7，退回到计算整体
            curr_arm_mse = np.mean((pred_np - gt_np) ** 2)
            arm_mses.append(curr_arm_mse)
            curr_gripper_mse = 0.0

        # === 打印前 3 个样本的详细信息用于调试 ===
        if i < 3:
            print(f"\n{'='*20} Sample {idx} (Preview) {'='*20}")
            print(f"Task: {obs.get('task', 'N/A')}")
            print(f"{'Dim':<5} | {'Pred':<10} | {'GT':<10} | {'Diff':<10}")
            print("-" * 45)
            # 只打印前 7 个维度
            for d in range(min(7, len(pred_np))):
                part = "Arm" if d < 6 else "Grip"
                print(f"{d:<2} {part}| {pred_np[d]:<10.4f} | {gt_np[d]:<10.4f} | {abs(pred_np[d]-gt_np[d]):<10.4f}")
            print("-" * 45)
            print(f"Frame Arm MSE:   {curr_arm_mse:.6f}")
            print(f"Frame Grip MSE:  {curr_gripper_mse:.6f}")

    # === 最终统计报告 ===
    print("\n" + "#" * 40)
    print(f"TEST SUMMARY (N={len(arm_mses)} frames)")
    print("#" * 40)
    
    if len(arm_mses) > 0:
        avg_arm_mse = np.mean(arm_mses)
        print(f"✅ Avg Arm MSE (Joints 0-5) : {avg_arm_mse:.6f}")
        print(f"   Arm RMSE               : {np.sqrt(avg_arm_mse):.6f}")
    
    if len(gripper_mses) > 0:
        avg_grip_mse = np.mean(gripper_mses)
        print(f"⚠️ Avg Gripper MSE (Joint 6) : {avg_grip_mse:.6f}")
        print(f"   (Expect large value due to 0-100 range)")
        print(f"   Gripper RMSE           : {np.sqrt(avg_grip_mse):.6f}")

    print("#" * 40)
    
    # 简单的判定
    if avg_arm_mse < 0.1:
        print("\n结论: Arm 动作拟合良好 (MSE < 0.1)")
    else:
        print("\n结论: Arm 动作拟合存在偏差，请检查归一化或训练步数")

if __name__ == "__main__":
    main()