# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn
from huggingface_hub import snapshot_download
from huggingface_hub.errors import HFValidationError, RepositoryNotFoundError

from lerobot.utils.import_utils import _transformers_available

# Conditional import for type checking and lazy loading
from transformers import AutoConfig, Qwen2_5_VLForConditionalGeneration, PretrainedConfig, PreTrainedModel
from transformers.feature_extraction_utils import BatchFeature

import tree


from lerobot.policies.efficientvla.action_head.flow_matching_action_head import (
    FlowmatchingActionHead,
    FlowmatchingActionHeadConfig,
)

# Defaults for RoboBrain backbone
DEFAULT_ROBOBRAIN_MODEL = "BAAI/RoboBrain2.0-3B"
DEFAULT_TOKENIZER_ASSETS_REPO = "BAAI/RoboBrain2.0-3B"


class RoboBrainBackbone(nn.Module):
    def __init__(
        self,
        model_path: str = DEFAULT_ROBOBRAIN_MODEL,
        tokenizer_assets_repo: str = DEFAULT_TOKENIZER_ASSETS_REPO,
        tune_llm: bool = False,
        tune_visual: bool = False,
        use_flash_attention: bool = False,
        load_bf16: bool = False,
        project_to_dim: int = 384,
    ):
        """
        Lightweight wrapper around RoboBrain base model to emit backbone_features/attention_mask.
        We keep a projection to 384 (1/4 of the original 1536) so the action head dimensions remain unchanged.
        """
        super().__init__()
        if AutoConfig is None or Qwen2_5_VLForConditionalGeneration is None:
            raise ImportError("transformers is required for RoboBrainBackbone")

        torch_dtype = torch.bfloat16 if load_bf16 else None
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        self.hidden_size = getattr(config, "hidden_size", 0)

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
        )

        self.project = nn.Linear(self.hidden_size, project_to_dim) if project_to_dim is not None else nn.Identity()

        self.tune_llm = tune_llm
        self.tune_visual = tune_visual
        self.set_trainable_parameters()

    def set_trainable_parameters(self, tune_visual=False, tune_llm=False):
        self.tune_visual, self.tune_llm = tune_visual, tune_llm
        assert self.tune_llm == self.tune_visual, \
            "Currently, it does not support fine-tuning only one of LLM and Visual Tower."
        
        for p in self.parameters():
            p.requires_grad = True
        if not self.tune_llm and not self.tune_visual:
            for p in self.model.parameters():
                p.requires_grad = False
        print(f"Tune backbone llm/visual: {self.tune_llm}/{self.tune_visual}")

    def set_frozen_modules_to_eval_mode(self):
        if self.training and not (self.tune_llm or self.tune_visual):
            self.model.eval()

    def prepare_input(self, batch: dict) -> BatchFeature:
        keep_keys = [
            "input_ids", 
            "attention_mask", 
            "pixel_values", 
            "image_grid_thw", 
            "video_grid_thw"
        ]
        
        filtered_batch = {
            k: v for k, v in batch.items() 
            if k in keep_keys and v is not None
        }
        return BatchFeature(data=filtered_batch)

    def forward(self, vl_input: BatchFeature) -> BatchFeature:
        self.set_frozen_modules_to_eval_mode()

        robobrain_inputs = dict(vl_input)
        # Ensure attention mask is present and boolean
        # attention_mask = robobrain_inputs.get("attention_mask")
        # if attention_mask is not None and attention_mask.dtype != torch.bool:
        #     robobrain_inputs["attention_mask"] = attention_mask.bool()
        outputs = self.model(**robobrain_inputs, output_hidden_states=True, return_dict=True)
        hidden = outputs.hidden_states[-1] if outputs.hidden_states is not None else outputs.last_hidden_state
        features = self.project(hidden)

        mask = robobrain_inputs.get("attention_mask")
        return BatchFeature(data={"backbone_features": features, "backbone_attention_mask": mask})


BACKBONE_FEATURE_KEY = "backbone_features"
ACTION_KEY = "action_pred"
LOSS_KEY = "loss"
ERROR_MSG = "Error: unexpected input/output"
N_COLOR_CHANNELS = 3


# config
@dataclass
class EfficientVLAV1Config(PretrainedConfig):
    model_type = "efficientvla_v1"
    backbone_cfg: dict = field(default_factory=dict, metadata={"help": "Backbone configuration."})
    action_head_cfg: dict = field(default_factory=dict, metadata={"help": "Action head configuration."})
    action_horizon: int = field(default=16, metadata={"help": "Action horizon."})
    action_dim: int = field(default=32, metadata={"help": "Action dimension."})
    compute_dtype: str = field(default="float32", metadata={"help": "Compute dtype."})

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Ensure required fields exist even if absent in the loaded HF config
        self.backbone_cfg = kwargs.get("backbone_cfg", {}) or {}
        self.action_head_cfg = kwargs.get("action_head_cfg", {}) or {}
        self.action_horizon = kwargs.get("action_horizon", 16)
        self.action_dim = kwargs.get("action_dim", 32)
        self.compute_dtype = kwargs.get("compute_dtype", "float32")


# real model
class EfficientVLAV1(PreTrainedModel):
    supports_gradient_checkpointing = True
    config_class = EfficientVLAV1Config
    """
    we expect the backbone output to have a key 'backbone_features' with shape (batch_size, n, hidden_size)
    here n is variable and can be e.g. time, 1 or user specified
    we expect the action head output to have a key 'action_pred' with shape (batch_size, time, action_dim) during inference time
    we expect these to have type BatchFeature, and they can of course have many other user specified keys too
    """

    def __init__(
        self,
        config: EfficientVLAV1Config,
        local_model_path: str,
    ):
        super().__init__(config)
        self.local_model_path = local_model_path

        # Guard against missing cfg blocks when loading a freshly converted checkpoint
        if not isinstance(config.backbone_cfg, dict) or not config.backbone_cfg:
            config.backbone_cfg = {
                "model_path": local_model_path,
                "tokenizer_assets_repo": DEFAULT_TOKENIZER_ASSETS_REPO,
                "tune_llm": False,
                "tune_visual": False,
                "project_to_dim": 384,
            }
        if not isinstance(config.action_head_cfg, dict) or not config.action_head_cfg:
            # Minimal defaults; adjust if your action space differs
            config.action_head_cfg = {
                "action_dim": getattr(config, "action_dim", 32),
                "action_horizon": getattr(config, "action_horizon", 16),
                "max_state_dim": getattr(config, "max_state_dim", 64),
                "max_action_dim": getattr(config, "action_dim", 32),
                "max_num_embodiments": 32,
                "input_embedding_dim": 384,
                "backbone_embedding_dim": 384,
                "vl_projection_hidden_dim": 256,
                "vl_self_attention_cfg": {
                    "num_attention_heads": 4,
                    "attention_head_dim": 32,  # 4*32=128 target dim
                    "num_layers": 2,
                    "dropout": 0.0,
                    "max_num_positional_embeddings": 1024,
                    "interleave_self_attention": False,
                },
                "diffusion_model_cfg": {
                    "num_attention_heads": 4,
                    "attention_head_dim": 32,
                },
            }

        self.backbone = RoboBrainBackbone(**config.backbone_cfg)
        action_head_cfg = FlowmatchingActionHeadConfig(**config.action_head_cfg)
        self.action_head = FlowmatchingActionHead(action_head_cfg)

        self.action_horizon = config.action_horizon
        self.action_dim = config.action_dim
        self.compute_dtype = config.compute_dtype

    def validate_inputs(self, inputs):
        # NOTE -- this should be handled internally by the model
        # however, doing that will likely be breaking changes -- so we'll need to do it after the deadline

        detected_error = False
        error_msg = ERROR_MSG
        if "action" in inputs:
            action = inputs["action"]
            # In inference, action may be omitted or None; validate only when it's a tensor.
            if action is None:
                pass  # allow None during inference
            elif isinstance(action, torch.Tensor):
                shape_ok = (
                    len(action.shape) == 3
                    and action.shape[1] == self.action_horizon
                    and action.shape[2] == self.action_dim
                )
                if not shape_ok:
                    error_msg += f"\n{action.shape=}"
                    detected_error = True
            else:
                # Unexpected non-tensor type provided for action
                error_msg += f"\nInvalid type for action: {type(action)}"
                detected_error = True

        if "video" in inputs:
            video = inputs["video"]
            type_ok = isinstance(video, np.ndarray)
            dtype_ok = video.dtype == np.uint8
            shape_ok = len(video.shape) == 6 and video.shape[3] == N_COLOR_CHANNELS
            if not type_ok:
                error_msg += f"\n{type(video)=}"
                detected_error = True
            if not dtype_ok:
                error_msg += f"\n{video.dtype=}"
                detected_error = True
            if not shape_ok:
                error_msg += f"\n{video.shape=}"
                detected_error = True

        if detected_error:
            raise ValueError(error_msg)

    def validate_data(self, action_head_outputs, backbone_outputs, is_training):
        fail_backbone = (
            not isinstance(backbone_outputs, BatchFeature) or BACKBONE_FEATURE_KEY not in backbone_outputs
        )

        if fail_backbone:
            error_msg = ERROR_MSG
            error_msg += f"\n{isinstance(backbone_outputs, BatchFeature)=}"
            error_msg += f"\n{BACKBONE_FEATURE_KEY in backbone_outputs=}"
            error_msg += f"\n{backbone_outputs[BACKBONE_FEATURE_KEY].shape=}"
            raise ValueError(error_msg)

        fail_action_head = (not isinstance(action_head_outputs, BatchFeature)) or not (
            (
                LOSS_KEY in action_head_outputs and is_training
            )  # there might not be an action prediction during training
            or (
                ACTION_KEY in action_head_outputs
                and action_head_outputs[ACTION_KEY].shape[1] == self.action_horizon
                and action_head_outputs[ACTION_KEY].shape[2] == self.action_dim
            )
        )

        if fail_action_head:
            error_msg = ERROR_MSG
            error_msg += f"\n{isinstance(action_head_outputs, BatchFeature)=}"
            error_msg += f"\n{LOSS_KEY in action_head_outputs=}"
            error_msg += f"\n{action_head_outputs[ACTION_KEY].shape=}"
            error_msg += f"\n{self.action_horizon=}"
            error_msg += f"\n{self.action_dim=}"
            raise ValueError(error_msg)

    def forward(
        self,
        inputs: dict,
    ) -> BatchFeature:
        backbone_inputs, action_inputs = self.prepare_input(inputs)
        backbone_outputs = self.backbone(backbone_inputs)
        action_head_outputs = self.action_head(backbone_outputs, action_inputs)
        self.validate_data(action_head_outputs, backbone_outputs, is_training=True)
        return action_head_outputs

    def get_action(
        self,
        inputs: dict,
    ) -> BatchFeature:
        backbone_inputs, action_inputs = self.prepare_input(inputs)
        # Because the behavior of backbones remains the same for training and inference, we can use `forward` for backbones.
        backbone_outputs = self.backbone(backbone_inputs)
        action_head_outputs = self.action_head.get_action(backbone_outputs, action_inputs)
        self.validate_data(action_head_outputs, backbone_outputs, is_training=False)
        return action_head_outputs

    def prepare_input(self, inputs) -> tuple[BatchFeature, BatchFeature]:
        self.validate_inputs(inputs)
        backbone_inputs = self.backbone.prepare_input(inputs)
        action_inputs = self.action_head.prepare_input(inputs)

        def to_device_with_maybe_dtype(x):
            # Cast floating tensors to a memory-efficient compute dtype when requested.
            # Rationale: Upcasting backbone activations to fp32 significantly increases VRAM.
            # When compute_dtype is bfloat16, prefer bf16 for activations to match AMP behavior.
            if not isinstance(x, torch.Tensor):
                return x
            if torch.is_floating_point(x):
                if getattr(self, "compute_dtype", None) == "bfloat16":
                    return x.to(self.device, dtype=torch.bfloat16)
                # Fallback: preserve previous behavior if not using bf16 compute
                return x.to(self.device, dtype=self.action_head.dtype)
            # Non-floating tensors: move device only
            return x.to(self.device)

        backbone_inputs = tree.map_structure(to_device_with_maybe_dtype, backbone_inputs)
        action_inputs = tree.map_structure(to_device_with_maybe_dtype, action_inputs)
        return backbone_inputs, action_inputs

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        # 1. 提取参数
        tune_visual = kwargs.pop("tune_visual", False)
        tune_llm = kwargs.pop("tune_llm", False)
        tune_projector = kwargs.pop("tune_projector", True)
        tune_diffusion_model = kwargs.pop("tune_diffusion_model", True)
        use_bf16 = kwargs.get("use_bf16", True) # 默认开启 bf16

        print(f"Loading pretrained dual brain from {pretrained_model_name_or_path} (Manual Mode)")

        # 2. 获取本地路径 (不加载权重，只拿路径)
        try:
            local_model_path = snapshot_download(pretrained_model_name_or_path, repo_type="model")
        except (HFValidationError, RepositoryNotFoundError):
            local_model_path = pretrained_model_name_or_path

        # 3. 手动构建 Config
        # 我们不再依赖 HF 的自动 Config 加载，而是手动创建一个干净的 Config
        config = cls.config_class()
        
        # 填充 Backbone 配置
        config.backbone_cfg = {
            "model_path": local_model_path,
            "tokenizer_assets_repo": DEFAULT_TOKENIZER_ASSETS_REPO,
            "tune_llm": tune_llm,
            "tune_visual": tune_visual,
            "project_to_dim": 384,
            "load_bf16": use_bf16  # 把 bf16 传进去
        }

        # 4. 实例化模型 (这会触发 Backbone 内部的 from_pretrained)
        # 因为绕过了 super().from_pretrained，外层 wrapper 不会尝试读取权重文件
        # 这避免了 "权重不匹配" 的警告，也避免了 "全0权重" 的 bug
        model = cls(config, local_model_path=local_model_path)

        # 5. 设置可训练参数

        model.backbone.set_trainable_parameters()
        model.action_head.set_trainable_parameters(
            tune_projector=tune_projector, tune_diffusion_model=tune_diffusion_model
        )


        return model
