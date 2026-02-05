#!/usr/bin/env python3
import torch
import json
import numpy as np
from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.processor import PolicyProcessorPipeline
from lerobot.processor.converters import policy_action_to_transition

# ================= 配置区域 =================
# 1. 你的 Dataset 路径 (来自 train_diffusion_policy.sh)
DATASET_PATH = "/home/img/project/lerobot/dynvla_dataset_lerobot"
# 2. 你的 Checkpoint 路径
CHECKPOINT_PATH = "/home/img/project/lerobot/policies/diffusion_policy/train/100ksteps/checkpoints/100000/pretrained_model"
# 3. 设备
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ===========================================

def load_deployment_model(checkpoint_dir):
    ckpt_path = Path(checkpoint_dir)
    
    # 1. Load Config
    with open(ckpt_path / "config.json", "r") as f:
        cfg_dict = json.load(f)
    # 清理不必要的字段以兼容 DiffusionConfig
    cfg_dict.pop("type", None) 
    
    # 重新构建 Config 对象
    # 注意：这里我们简单处理，假设 lerobot 能够自动处理 FeatureType 枚举
    # 如果报错，说明需要像 deploy 脚本里那样手动转换 FeatureType
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

    cfg = DiffusionConfig(**cfg_dict)
    cfg.device = DEVICE
    cfg.num_inference_steps = 10  # 保持和部署一致，或者设为训练时的 100

    # 2. Load Processors (Pre & Post)
    # 模拟部署时的 overrides
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
    policy = DiffusionPolicy.from_pretrained(
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
    # 使用 LeRobotDataset 读取数据
    dataset = LeRobotDataset(root=DATASET_PATH, repo_id="dummy_id") 
    # 注意：repo_id 这里可能不重要，只要 root 对了通常能读本地数据
    # 如果报错，请尝试 dataset = LeRobotDataset(root=DATASET_PATH) 或指向正确的 repo_id

    # 随机取几个样本进行测试
    indices = [0, 100, 500] # 或者用 np.random.randint
    
    for idx in indices:
        print(f"\n{'='*20} Testing Sample Index: {idx} {'='*20}")
        item = dataset[idx]
        
        # === 1. 构造部署时的输入格式 ===
        # 部署脚本里是单独拿出来 keys 组装成 dict，这里模拟这个过程
        # Dataset 里通常已经是 torch.Tensor
        obs = {}
        
        # 提取 State
        if "observation.state" in item:
            obs["observation.state"] = item["observation.state"] # Shape: (D,)
            
        # 提取 Images
        for key in item.keys():
            if "observation.images" in key:
                # 部署脚本里由 _frame_to_tensor 处理，通常已经是 Float 0-1, CHW
                # Dataset 里取出来的通常也是 Float 0-1 (如果用了 image_transforms) 或者 Uint8
                # 需要确认 dataset 的输出是否已经是 float 0-1。LeRobotDataset 默认通常是 Float 0-1。
                img = item[key]
                obs[key] = img

        # 增加 Batch 维度？
        # 部署脚本的 preproc 包含 'AddBatchDimensionProcessorStep' 吗？
        # 看了你的 policy_preprocessor.json，包含 "registry_name": "to_batch_processor" 
        # (通常旧版本叫 AddBatchDimension，新版可能是 to_batch)
        # 实际上 PolicyProcessorPipeline 会处理这个。如果输入是单个样本 (C,H,W)，它会变成 (1,C,H,W)。
        
        # === 2. 运行部署管线 ===
        # Step A: Pre-process (归一化, 上传GPU, 加Batch)
        try:
            model_input = preproc(obs)
        except Exception as e:
            print(f"Preprocessing failed: {e}")
            # 有可能是 dataset 返回的数据带了 batch 维度，或者是 uint8 而 preproc 期待 float
            # 调试打印一下 shape
            print("Input shapes:", {k: v.shape for k,v in obs.items()})
            continue

        # Step B: Model Inference
        with torch.no_grad():
            # select_action 会处理 Horizon 和 Queue，这里直接调用
            # 注意：select_action 内部有 queue 逻辑，第一次调用可能只会填充 queue。
            # 为了测试纯推理，我们最好绕过 queue 直接由 predict_action 
            # 或者重置 queue 后运行一次。
            policy.reset() 
            action_out = policy.select_action(model_input)
            
        # Step C: Post-process (反归一化, 转CPU, 去Batch)
        # 部署脚本里：post = self.postproc(policy_action)["action"]
        result = postproc(action_out)
        pred_action = result["action"]
        
        # === 3. 获取 Ground Truth (真实标签) ===
        gt_action = item["action"] # Shape: (Action_Dim,)
        
        # === 4. 对比与诊断 ===
        # 把 Tensor 转成 numpy 方便打印
        pred_np = pred_action.squeeze().cpu().numpy()
        gt_np = gt_action.cpu().numpy()
        
        print(f"State Input (Raw): {obs['observation.state'].numpy()}")
        print("-" * 30)
        print(f"{'Dimension':<5} | {'Pred (Deploy)':<15} | {'GT (Dataset)':<15} | {'Diff':<10}")
        print("-" * 60)
        
        for i in range(len(pred_np)):
            diff = abs(pred_np[i] - gt_np[i])
            print(f"J{i:<4} | {pred_np[i]:<15.4f} | {gt_np[i]:<15.4f} | {diff:<10.4f}")
            
        print("-" * 60)
        
        # === 核心诊断逻辑 ===
        max_val = np.max(np.abs(pred_np[:6])) # 只看前6个关节，忽略夹爪
        if max_val > 4.0:
            print("⚠️ 警告: 预测值包含 > 4.0 的数值。这极可能是【角度 (Degrees)】！")
            print("   如果你的机器人底层需要弧度 (Radians, -3.14 ~ 3.14)，这就是乱飞的原因。")
        elif max_val <= 3.15:
            print("✅ 提示: 预测值在弧度范围内 (-3.14 ~ 3.14)。")
            
        mse = np.mean((pred_np - gt_np)**2)
        print(f"Mean Squared Error: {mse:.6f}")
        if mse > 0.1:
            print("❌ 严重偏差: 即使在训练集上，部署管线的输出也和 GT 差很远。")
            print("   可能原因: 归一化统计量加载错误、Observation 预处理不一致（如 RGB vs BGR）。")
        else:
            print("✅ 验证通过: 部署管线能够复现 Dataset 的分布。")

if __name__ == "__main__":
    main()