#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
[DEBUG VERSION] Oracle Policy Deployment
此版本增加了大量日志用于排查“机械臂乱飞”问题。
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool

try:
    from piper_msgs.srv import Enable as EnableSrv
except ImportError:
    EnableSrv = None

from pika.gripper import Gripper

# --------------------------------------------------------------------------- #
# Constants & Config                                                          #
# --------------------------------------------------------------------------- #

ORACLE_ROOT = Path("/home/data/Project/Oracle") # 请确保路径正确
sys.path.append(str(ORACLE_ROOT))

# 引入 LeRobot 相关库
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.processor import PolicyAction, PolicyProcessorPipeline
from lerobot.processor.converters import policy_action_to_transition

# 关节限制 (弧度)
JOINT_LIMITS: Sequence[tuple[float, float]] = [
    (-3.14, 3.14), (0.00, 2.00), (-2.00, 0.00),
    (-1.50, 1.80), (-1.30, 1.57), (-3.14, 3.14),
]
GRIPPER_MIN = 0.0
GRIPPER_MAX = 90.0 # 假设是 0-100 或 0-90 的数值

HOME_POS = [0.0, -0.035, 0.0, 0.0, 0.35, 0.0]

def clamp(value: float, bounds: Tuple[float, float]) -> float:
    lo, hi = bounds
    return float(max(lo, min(hi, value)))

@dataclass
class DeployArgs:
    checkpoint: Path
    device: str
    num_inference_steps: Optional[int]
    task_text: str
    rate: float
    gripper_port: str
    fisheye_index: Optional[int]
    realsense_serial: Optional[str]
    camera_width: int
    camera_height: int
    camera_fps: int

class OracleDeployNode(Node):
    def __init__(self, args: DeployArgs) -> None:
        super().__init__("oracle_realtime_controller")
        self.args = args
        # 强制修正频率：如果传入 0.5，强制改为 30Hz 进行测试，避免低频导致的控制滞后
        if self.args.rate < 1.0:
            self.get_logger().warn(f"⚠️ 检测到极低的控制频率 ({self.args.rate} Hz)。强制覆盖为 15.0 Hz 进行调试！")
            self.dt = 1.0 / 15.0
        else:
            self.dt = 1.0 / self.args.rate
            
        self.device = torch.device(args.device)
        torch.set_grad_enabled(False)

        # ROS Setup
        self.pub_joint = self.create_publisher(JointState, "/joint_ctrl_single", 10)
        self.pub_enable = self.create_publisher(Bool, "/enable_flag", 10)
        self.sub_joint = self.create_subscription(JointState, "/joint_states_single", self._joint_cb, 10)
        self.enable_client = self.create_client(EnableSrv, "/enable_srv") if EnableSrv else None

        self.expected_joint_names = [f"joint{i+1}" for i in range(6)]
        self.joint_index_map = None
        self.actual_positions = [0.0] * 6
        self.have_joint_sample = False
        self.latest_gripper = GRIPPER_MIN

        # Hardware Setup
        self.gripper = None
        self.fisheye_camera = None
        self.realsense_camera = None
        self._init_gripper()

        # Model Load
        self.cfg, self.preproc, self.postproc, self.model = self._load_policy_and_processors()
        self.model.eval()
        self.model.reset()

        self.timer = None
        self._shutdown_called = False
        self._last_command_time = time.monotonic()
        
        # 获取图像尺寸用于 Resize 检查
        self.image_shapes = {
            name: tuple(feature.shape)
            for name, feature in self.cfg.input_features.items()
            if name.startswith("observation.images")
        }

        self.get_logger().info("✅ Debug Node Ready. Moving to HOME...")
        self._move_to_home()
        time.sleep(2.0)
        self.get_logger().info("🚀 Starting Control Loop...")
        self.timer = self.create_timer(self.dt, self._tick)

    def _init_gripper(self) -> None:
        try:
            self.get_logger().info(f"Connecting Gripper on {self.args.gripper_port}...")
            gripper = Gripper(self.args.gripper_port)
            if not gripper.connect() or not gripper.enable():
                raise RuntimeError("Gripper connection failed")
            
            gripper.set_camera_param(self.args.camera_width, self.args.camera_height, self.args.camera_fps)
            if self.args.fisheye_index is not None:
                gripper.set_fisheye_camera_index(self.args.fisheye_index)
            if self.args.realsense_serial is not None:
                gripper.set_realsense_serial_number(self.args.realsense_serial)
                
            self.fisheye_camera = gripper.get_fisheye_camera()
            self.realsense_camera = gripper.get_realsense_camera()
            self.gripper = gripper
            self.get_logger().info(f"Hardware OK. Fisheye: {bool(self.fisheye_camera)}, RS: {bool(self.realsense_camera)}")
        except Exception as e:
            self.get_logger().error(f"Hardware Init Failed: {e}")
            sys.exit(1)

    def _load_policy_and_processors(self):
        ckpt_dir = Path(self.args.checkpoint)
        if (ckpt_dir / "pretrained_model").is_dir():
            ckpt_dir = ckpt_dir / "pretrained_model"
            
        self.get_logger().info(f"Loading Model from: {ckpt_dir}")
        with open(ckpt_dir / "config.json", "r") as f:
            cfg_dict = json.load(f)
        cfg_dict.pop("type", None)

        def _to_feature(feat_dict: dict) -> PolicyFeature:
            return PolicyFeature(type=FeatureType[feat_dict["type"]], shape=tuple(feat_dict.get("shape") or ()))

        if "input_features" in cfg_dict:
            cfg_dict["input_features"] = {k: _to_feature(v) for k, v in cfg_dict["input_features"].items()}
        if "output_features" in cfg_dict:
            cfg_dict["output_features"] = {k: _to_feature(v) for k, v in cfg_dict["output_features"].items()}
        if "normalization_mapping" in cfg_dict:
            cfg_dict["normalization_mapping"] = {k: NormalizationMode[v] for k, v in cfg_dict["normalization_mapping"].items()}
            
        cfg = DiffusionConfig(**cfg_dict)
        cfg.device = str(self.device)
        if self.args.num_inference_steps:
            cfg.num_inference_steps = int(self.args.num_inference_steps)

        overrides = {"device_processor": {"device": str(self.device)}}
        pre = PolicyProcessorPipeline.from_pretrained(ckpt_dir, config_filename="policy_preprocessor.json", overrides=overrides)
        post = PolicyProcessorPipeline.from_pretrained(ckpt_dir, config_filename="policy_postprocessor.json", overrides={"device_processor": {"device": "cpu"}}, to_transition=policy_action_to_transition)
        policy = DiffusionPolicy.from_pretrained(str(ckpt_dir), config=cfg)
        
        return cfg, pre, post, policy

    def _joint_cb(self, msg: JointState) -> None:
        if self.joint_index_map is None:
            name_to_index = {name: i for i, name in enumerate(msg.name)}
            self.joint_index_map = [name_to_index.get(name, i) for i, name in enumerate(self.expected_joint_names)]
            
        for out_idx, src_idx in enumerate(self.joint_index_map):
            if src_idx < len(msg.position):
                self.actual_positions[out_idx] = msg.position[src_idx]
        self.have_joint_sample = True

    def _frame_to_tensor(self, frame_bgr: np.ndarray, key: str) -> torch.Tensor:
        # Debug: 检查图像是否有信号
        if frame_bgr is None or frame_bgr.size == 0:
            return torch.zeros(self.image_shapes.get(key), dtype=torch.float32)
        
        # 简单的统计量检查
        mean_val = np.mean(frame_bgr)
        if mean_val < 1.0:
            self.get_logger().warn(f"⚠️ Image {key} seems extremely dark (Mean: {mean_val:.2f})")
            
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        
        target_shape = self.image_shapes.get(key)
        if target_shape:
            _, h, w = target_shape
            if tensor.shape[1:] != (h, w):
                tensor = F.interpolate(tensor.unsqueeze(0), size=(h, w), mode="bilinear", align_corners=False).squeeze(0)

        if not hasattr(self, "save_cnt"): self.save_cnt = 0
        self.save_cnt += 1
        if self.save_cnt % 30 == 0:
            # 保存到当前目录，文件名带上 key，看清楚是谁
            cv2.imwrite(f"debug_{key}.jpg", frame_bgr)
            print(f"📸 Saved debug image: debug_{key}.jpg")
        return tensor

    def _tick(self) -> None:
        if not self.have_joint_sample:
            self.get_logger().info("Waiting for joint states...")
            return

        start_time = time.time()
        
        # 1. Capture Data
        images = {}
        # 尝试捕获图像，失败则用全黑填充以防报错(仅调试用)
        if self.fisheye_camera:
            ok, frame = self.fisheye_camera.get_frame()
            if ok and frame is not None:
                images["observation.images.fisheye_rgb"] = self._frame_to_tensor(frame, "observation.images.fisheye_rgb")
        if self.realsense_camera:
            ok, frame = self.realsense_camera.get_color_frame()
            if ok and frame is not None:
                images["observation.images.realsense_rgb"] = self._frame_to_tensor(frame, "observation.images.realsense_rgb")
        
        # 构建 State
        grip_val = 0.0
        if self.gripper:
            try:
                grip_val = self.gripper.get_gripper_distance()
            except: pass
        
        # === 🕵️ DEBUG LOG 1: INPUT STATE ===
        # 这里最关键：打印实际收到的关节值
        # 如果你看到 [30.5, -10.2 ...] 说明是角度 -> 完蛋，需要转弧度
        # 如果你看到 [0.52, -0.18 ...] 说明是弧度 -> 正常
        raw_joints = list(self.actual_positions)
        state_vec = raw_joints + [grip_val]
        
        print(f"\n[{time.strftime('%H:%M:%S')}] === INPUT CHECK ===")
        print(f" > Raw Joints (Input): {[round(x, 4) for x in raw_joints]}")
        print(f" > Gripper Val: {grip_val}")
        
        if any(abs(x) > 3.2 for x in raw_joints):
            print("⚠️⚠️⚠️ WARNING: Joint value > 3.2 detected! Likely DEGREES. Dataset expects RADIANS.")
        
        # 组装 Observation
        state_tensor = torch.tensor(state_vec, dtype=torch.float32)
        obs = {"observation.state": state_tensor}
        obs.update(images)

        # 2. Inference
        try:
            model_in = self.preproc(obs)
            
            # === 🕵️ DEBUG LOG 2: NORMALIZED INPUT ===
            # 检查归一化后的 State 是否在 [-2, 2] 这种合理范围内
            # 如果这里出现 20, 30 这种大数，说明输入分布和训练分布不匹配
            norm_state = model_in["observation.state"]
            if isinstance(norm_state, torch.Tensor):
                print(f" > Norm State (In Model): {norm_state[0].cpu().numpy().round(2)}")
            
            with torch.no_grad():
                action_out = self.model.select_action(model_in)
                
            result = self.postproc(action_out)
            action = result["action"] # Tensor
            
            if action.dim() > 1: action = action[0]
            command = action.detach().cpu().numpy().tolist()
            
            # === 🕵️ DEBUG LOG 3: OUTPUT ACTION ===
            print(f" > Model Pred Action: {[round(x, 4) for x in command]}")
            
            # 3. Execution (With Safety Check)
            self._execute_command_safe(command)
            
            end_time = time.time()
            print(f" > Latency: {(end_time - start_time)*1000:.1f} ms")
            
        except Exception as e:
            self.get_logger().error(f"Inference Error: {e}")

    def _execute_command_safe(self, target):
        # 简单的安全过滤
        safe_target = []
        for i, val in enumerate(target[:6]):
            limit = JOINT_LIMITS[i]
            clipped = clamp(val, limit)
            safe_target.append(clipped)
            # 如果裁剪幅度很大，说明模型想去非法区域
            if abs(clipped - val) > 0.1:
                print(f"⚠️ Safety Clip J{i}: {val:.2f} -> {clipped:.2f}")

        # 发布
        msg = JointState()
        msg.name = [f"joint{i+1}" for i in range(6)]
        msg.position = safe_target
        self.pub_joint.publish(msg)
        
        # Gripper
        if len(target) > 6:
            g_tgt = clamp(target[6], (GRIPPER_MIN, GRIPPER_MAX))
            if self.gripper:
                self.gripper.set_gripper_distance(g_tgt)

    def _move_to_home(self):
        # 简化版回零
        msg = JointState()
        msg.name = [f"joint{i+1}" for i in range(6)]
        msg.position = HOME_POS
        for _ in range(10):
            self.pub_joint.publish(msg)
            time.sleep(0.1)

    def shutdown(self):
        self.get_logger().info("Shutdown called.")
        if self.gripper: self.gripper.disconnect()

# Main Entry
def main():
    # 模拟参数解析，为了方便直接运行
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--rate", type=float, default=15.0) # 默认提高频率
    parser.add_argument("--gripper-port", type=str, default="/dev/ttyUSB0")
    # ... 其他参数可按需补全，这里主要为了跑起来
    
    # 这里的 hack 是为了让你复用现有的 shell 脚本传参
    # 如果 shell 脚本传了 --task-text 等参数，parser 需要定义它们
    # 为了简单，建议在 shell 脚本里只保留核心参数，或者在这里补全所有 arguments
    
    # 补全 args 以防报错
    parser.add_argument("--task-text", type=str, default="")
    parser.add_argument("--num-inference-steps", type=int, default=10)
    parser.add_argument("--fisheye-index", type=int, default=None)
    parser.add_argument("--realsense-serial", type=str, default=None)
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--camera-fps", type=int, default=30)
    
    args = parser.parse_args()
    
    rclpy.init()
    node = OracleDeployNode(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()