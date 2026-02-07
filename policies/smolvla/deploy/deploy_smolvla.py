#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
[SmolVLA] Oracle Policy Deployment
适配 SmolVLA 模型的实时部署脚本。
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

# [修改点 1] 引入 SmolVLA 相关库
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.processor import PolicyProcessorPipeline
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
        super().__init__("oracle_realtime_controller_smolvla")
        self.args = args
        
        # 频率检查
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
        
        # 获取图像尺寸用于 Resize 检查
        self.image_shapes = {
            name: tuple(feature.shape)
            for name, feature in self.cfg.input_features.items()
            if name.startswith("observation.images")
        }

        self.get_logger().info("✅ SmolVLA Node Ready. Moving to HOME...")
        self._move_to_home()
        time.sleep(2.0)
        self.get_logger().info(f"🚀 Starting Control Loop... Task: '{self.args.task_text}'")
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
        # 兼容两种路径结构
        if (ckpt_dir / "pretrained_model").is_dir():
            ckpt_dir = ckpt_dir / "pretrained_model"
            
        self.get_logger().info(f"Loading SmolVLA from: {ckpt_dir}")
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
            
        # [修改点 2] 使用 SmolVLAConfig 和 SmolVLAPolicy
        cfg = SmolVLAConfig(**cfg_dict)
        cfg.device = str(self.device)
        
        # 尝试设置步数 (如果 config 支持)
        # SmolVLA 通常不需要动态设置 num_inference_steps，但如果想覆盖默认值：
        # if hasattr(cfg, "num_steps") and self.args.num_inference_steps:
        #     cfg.num_steps = int(self.args.num_inference_steps)

        overrides = {"device_processor": {"device": str(self.device)}}
        pre = PolicyProcessorPipeline.from_pretrained(ckpt_dir, config_filename="policy_preprocessor.json", overrides=overrides)
        post = PolicyProcessorPipeline.from_pretrained(ckpt_dir, config_filename="policy_postprocessor.json", overrides={"device_processor": {"device": "cpu"}}, to_transition=policy_action_to_transition)
        
        policy = SmolVLAPolicy.from_pretrained(str(ckpt_dir), config=cfg)
        
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
        if self.save_cnt % 60 == 0: # 稍微降低保存频率
            cv2.imwrite(f"debug_{key}.jpg", frame_bgr)
            # print(f"📸 Saved debug image: debug_{key}.jpg")
        return tensor

    def _tick(self) -> None:
        if not self.have_joint_sample:
            self.get_logger().info("Waiting for joint states...")
            return

        start_time = time.time()
        
        # 1. Capture Data
        images = {}
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
        raw_joints = list(self.actual_positions)
        state_vec = raw_joints + [grip_val]
        
        # 组装 Observation
        state_tensor = torch.tensor(state_vec, dtype=torch.float32)
        obs = {"observation.state": state_tensor}
        obs.update(images)
        
        # [修改点 3] 注入文本指令 (Crucial for SmolVLA)
        # 如果 self.args.task_text 为空，SmolVLA 会报错或行为异常
        if not self.args.task_text:
             self.get_logger().warn("⚠️ Task text is empty! Using default.")
             obs["task"] = "do something"
        else:
             obs["task"] = self.args.task_text

        # 2. Inference
        try:
            model_in = self.preproc(obs)
            
            with torch.no_grad():
                action_out = self.model.select_action(model_in)
                
            result = self.postproc(action_out)
            action = result["action"] # Tensor
            
            if action.dim() > 1: action = action[0]
            command = action.detach().cpu().numpy().tolist()
            
            # === 🕵️ DEBUG LOG 3: OUTPUT ACTION ===
            print(f"[{time.strftime('%H:%M:%S')}] Out: {[round(x, 4) for x in command]}")
            
            # 3. Execution (With Safety Check)
            self._execute_command_safe(command)
            
            end_time = time.time()
            # print(f" > Latency: {(end_time - start_time)*1000:.1f} ms")
            
        except Exception as e:
            self.get_logger().error(f"Inference Error: {e}")

    def _execute_command_safe(self, target):
        safe_target = []
        for i, val in enumerate(target[:6]):
            limit = JOINT_LIMITS[i]
            clipped = clamp(val, limit)
            safe_target.append(clipped)
            if abs(clipped - val) > 0.1:
                print(f"⚠️ Safety Clip J{i}: {val:.2f} -> {clipped:.2f}")

        msg = JointState()
        msg.name = [f"joint{i+1}" for i in range(6)]
        msg.position = safe_target
        self.pub_joint.publish(msg)
        
        if len(target) > 6:
            g_tgt = clamp(target[6], (GRIPPER_MIN, GRIPPER_MAX))
            if self.gripper:
                self.gripper.set_gripper_distance(g_tgt)

    def _move_to_home(self):
        msg = JointState()
        msg.name = [f"joint{i+1}" for i in range(6)]
        msg.position = HOME_POS
        for _ in range(10):
            self.pub_joint.publish(msg)
            time.sleep(0.1)

    def shutdown(self):
        self.get_logger().info("Shutdown called.")
        if self.gripper: self.gripper.disconnect()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--rate", type=float, default=15.0)
    parser.add_argument("--gripper-port", type=str, default="/dev/ttyUSB0")
    parser.add_argument("--task-text", type=str, default="pick up the object") # 设置默认任务
    parser.add_argument("--num-inference-steps", type=int, default=10) # 默认为 10 (SmolVLA)
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