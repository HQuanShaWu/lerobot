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
import importlib

import numpy as np
import torch
import torch.nn as nn
from huggingface_hub import snapshot_download
from huggingface_hub.errors import HFValidationError, RepositoryNotFoundError

from peft import LoraConfig, get_peft_model


# Transformers imports
from transformers import AutoModel, AutoTokenizer, PretrainedConfig, PreTrainedModel
from transformers.feature_extraction_utils import BatchFeature

import tree


from lerobot.policies.efficientvla.action_head.flow_matching_action_head import (
    FlowmatchingActionHead,
    FlowmatchingActionHeadConfig,
)

# Defaults for InternVL3 backbone
DEFAULT_INTERNVL3_MODEL = (
    "/home/img/project/lerobot/src/lerobot/policies/efficientvla/InternVL3-1B/internvl3_1b_sft"
)
DEFAULT_TOKENIZER_ASSETS_REPO = (
    "/home/img/project/lerobot/src/lerobot/policies/efficientvla/InternVL3-1B/internvl3_1b_sft"
)

SCALE_PRESETS = {
    "large": {
        "project_to_dim": 1536,
        "input_embedding_dim": 1536,
        "backbone_embedding_dim": 1536,
        "vl_projection_hidden_dim": 1024,
        "hidden_size": 1024,
        "vl_self_attention_cfg": {
            "num_attention_heads": 8,
            "attention_head_dim": 64,
            "num_layers": 2,
            "dropout": 0.0,
            "max_num_positional_embeddings": 1024,
            "interleave_self_attention": False,
        },
        "diffusion_model_cfg": {
            "num_attention_heads": 8,
            "attention_head_dim": 64,
        },
    },
    "medium": {
        "project_to_dim": 768,
        "input_embedding_dim": 768,
        "backbone_embedding_dim": 768,
        "vl_projection_hidden_dim": 512,
        "hidden_size": 512,
        "vl_self_attention_cfg": {
            "num_attention_heads": 8,
            "attention_head_dim": 32,
            "num_layers": 2,
            "dropout": 0.0,
            "max_num_positional_embeddings": 1024,
            "interleave_self_attention": False,
        },
        "diffusion_model_cfg": {
            "num_attention_heads": 8,
            "attention_head_dim": 32,
        },
    },
    "tiny": {
        "project_to_dim": 384,
        "input_embedding_dim": 384,
        "backbone_embedding_dim": 384,
        "vl_projection_hidden_dim": 256,
        "hidden_size": 256,
        "vl_self_attention_cfg": {
            "num_attention_heads": 4,
            "attention_head_dim": 32,
            "num_layers": 2,
            "dropout": 0.0,
            "max_num_positional_embeddings": 1024,
            "interleave_self_attention": False,
        },
        "diffusion_model_cfg": {
            "num_attention_heads": 4,
            "attention_head_dim": 32,
        },
    },
}


class InternVL3Backbone(nn.Module):
    def __init__(
        self,
        model_path: str = DEFAULT_INTERNVL3_MODEL,
        tokenizer_assets_repo: str = DEFAULT_TOKENIZER_ASSETS_REPO,
        tune_llm: bool = False,
        tune_visual: bool = False,
        use_flash_attention: bool = False,
        load_bf16: bool = False,
        project_to_dim: int = 768,
        lora_rank: int = 0,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        lora_full_model: bool = False,
    ):
        """
        Lightweight wrapper around InternVL3 base model to emit backbone_features/attention_mask.
        We keep a projection to 768 (1/2 of the original 1536) so the action head dimensions remain unchanged.
        """
        super().__init__()
        if AutoModel is None or AutoTokenizer is None:
            raise ImportError("transformers is required for InternVL3Backbone")

        self.model_path = model_path
        self.tokenizer_assets_repo = tokenizer_assets_repo
        torch_dtype = self._resolve_torch_dtype(load_bf16)
        tokenizer_path = tokenizer_assets_repo or model_path
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=True,
            use_fast=False,
            fix_mistral_regex=True,
        )
        if hasattr(self.tokenizer, "padding_side"):
            self.tokenizer.padding_side = "left"

        self.model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
        ).eval()

        self.IMG_START_TOKEN = "<img>"
        self.IMG_END_TOKEN = "</img>"
        self.IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"
        self.model.img_context_token_id = self.tokenizer.convert_tokens_to_ids(self.IMG_CONTEXT_TOKEN)

        self.get_conv_template = self._resolve_get_conv_template()
        self.hidden_size = self._resolve_hidden_size()
        self.project = nn.Linear(self.hidden_size, project_to_dim) if project_to_dim is not None else nn.Identity()

        self.tune_llm = tune_llm
        self.tune_visual = tune_visual
        self.use_lora = lora_rank is not None and lora_rank > 0

        if self.use_lora:
            if LoraConfig is None or get_peft_model is None:
                raise ImportError("peft is required for LoRA fine-tuning but is not installed.")
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
            if lora_full_model:
                target_modules += ["up_proj", "gate_proj", "down_proj"]
            lora_config = LoraConfig(
                r=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=target_modules,
                bias="none",
                task_type="CAUSAL_LM",
            )
            if hasattr(self.model, "language_model"):
                self.model.language_model = get_peft_model(self.model.language_model, lora_config)
            else:
                self.model = get_peft_model(self.model, lora_config)

        self.set_trainable_parameters()

    def _resolve_torch_dtype(self, load_bf16: bool) -> torch.dtype:
        if torch.cuda.is_available():
            if load_bf16 and torch.cuda.get_device_capability(0)[0] >= 8:
                return torch.bfloat16
            return torch.float16
        return torch.float32

    def _resolve_hidden_size(self) -> int:
        config = getattr(self.model, "config", None)
        if config is not None:
            for candidate in (
                getattr(config, "hidden_size", None),
                getattr(getattr(config, "llm_config", None), "hidden_size", None),
                getattr(getattr(config, "text_config", None), "hidden_size", None),
            ):
                if isinstance(candidate, int) and candidate > 0:
                    return candidate
        try:
            return int(self.model.get_input_embeddings().weight.shape[1])
        except Exception as exc:
            raise ValueError("Unable to infer hidden size for InternVL3 model.") from exc

    def _resolve_get_conv_template(self):
        model_mod = self.model.__class__.__module__
        pkg = model_mod.rsplit(".", 1)[0]
        try:
            conv_mod = importlib.import_module(pkg + ".conversation")
            return getattr(conv_mod, "get_conv_template")
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Missing conversation module for InternVL3. "
                f"Got model module '{model_mod}'. "
                "This usually means the VLM checkpoint was saved from the wrong base model "
                "(e.g., Qwen2 without InternVL3 remote code). "
                "Re-save the VLM using InternVL3 (after merging LoRA), or point to the original "
                "InternVL3 repo path."
            ) from exc

    def _get_call_model(self):
        model = self.model
        if getattr(model, "peft_config", None) is not None:
            return getattr(model, "base_model", model)
        return model

    def set_trainable_parameters(self, tune_visual=False, tune_llm=False):
        self.tune_visual, self.tune_llm = tune_visual, tune_llm
        assert self.tune_llm == self.tune_visual, \
            "Currently, it does not support fine-tuning only one of LLM and Visual Tower."
        
        for p in self.parameters():
            p.requires_grad = True
        if not self.tune_llm and not self.tune_visual:
            for name, p in self.model.named_parameters():
                if self.use_lora and "lora_" in name:
                    p.requires_grad = True
                else:
                    p.requires_grad = False
            if self.use_lora:
                print(f"Tune backbone llm/visual: LoRA")
            else:
                print(f"Tune backbone llm/visual: {self.tune_llm}/{self.tune_visual}")

    def set_frozen_modules_to_eval_mode(self):
        if self.training and not (self.tune_llm or self.tune_visual):
            self.model.eval()

    def prepare_input(self, batch: dict) -> BatchFeature:
        if "internvl3_inputs" in batch:
            return BatchFeature(data={"internvl3_inputs": batch["internvl3_inputs"]})

        keep_keys = [
            "input_ids",
            "attention_mask",
            "pixel_values",
            "image_flags",
        ]
        filtered_batch = {k: v for k, v in batch.items() if k in keep_keys and v is not None}
        return BatchFeature(data=filtered_batch)

    def _build_query(self, text: str, num_patches_list: list[int]) -> str:
        text = "" if text is None else str(text)
        if len(num_patches_list) == 0:
            question = text
        elif len(num_patches_list) == 1:
            question = f"<image>\n{text}"
        else:
            prefix = "".join([f"Image-{i + 1}: <image>\n" for i in range(len(num_patches_list))])
            question = prefix + text

        template = self.get_conv_template(self.model.template)
        template.system_message = self.model.system_message
        template.append_message(template.roles[0], question)
        template.append_message(template.roles[1], None)
        query = template.get_prompt()

        for n_patch in num_patches_list:
            image_tokens = (
                self.IMG_START_TOKEN
                + (self.IMG_CONTEXT_TOKEN * self.model.num_image_token * n_patch)
                + self.IMG_END_TOKEN
            )
            query = query.replace("<image>", image_tokens, 1)
        return query

    def _encode_internvl3_inputs(
        self, internvl3_inputs: list[dict[str, torch.Tensor | list[int] | str]]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not internvl3_inputs:
            raise ValueError("internvl3_inputs is empty.")

        device = next(self.model.parameters()).device
        dtype = next(self.model.parameters()).dtype

        queries: list[str] = []
        pixel_values_list: list[torch.Tensor] = []
        image_flags_list: list[torch.Tensor] = []
        for item in internvl3_inputs:
            text = item.get("language", "Perform the task.")
            num_patches_list = item.get("num_patches_list", [])
            queries.append(self._build_query(text, num_patches_list))

            pixel_values = item.get("pixel_values")
            if pixel_values is not None:
                pixel_values_list.append(pixel_values)
                image_flags_list.append(
                    torch.ones((pixel_values.shape[0], 1), dtype=torch.long)
                )

        if not pixel_values_list:
            raise ValueError("InternVL3 inputs require pixel_values.")

        model_inputs = self.tokenizer(queries, return_tensors="pt", padding=True)
        input_ids = model_inputs["input_ids"].to(device)
        attention_mask = model_inputs["attention_mask"].to(device)

        pixel_values = torch.cat(pixel_values_list, dim=0).to(device=device, dtype=dtype, non_blocking=True)
        image_flags = torch.cat(image_flags_list, dim=0).to(device=device, dtype=torch.long, non_blocking=True)

        call_model = self._get_call_model()
        outputs = call_model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            image_flags=image_flags,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden = outputs.hidden_states[-1] if outputs.hidden_states is not None else outputs.last_hidden_state
        return hidden, attention_mask

    def forward(self, vl_input: BatchFeature) -> BatchFeature:
        self.set_frozen_modules_to_eval_mode()

        internvl3_inputs = dict(vl_input)
        if "internvl3_inputs" in internvl3_inputs:
            hidden, mask = self._encode_internvl3_inputs(internvl3_inputs["internvl3_inputs"])
        else:
            call_model = self._get_call_model()
            outputs = call_model(**internvl3_inputs, output_hidden_states=True, return_dict=True)
            hidden = outputs.hidden_states[-1] if outputs.hidden_states is not None else outputs.last_hidden_state
            mask = internvl3_inputs.get("attention_mask")

        features = self.project(hidden)
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
        preset_scale = getattr(config, "scale", "medium")
        preset = SCALE_PRESETS.get(preset_scale, SCALE_PRESETS["medium"])

        if not isinstance(config.backbone_cfg, dict) or not config.backbone_cfg:
            config.backbone_cfg = {
                "model_path": local_model_path,
                "tokenizer_assets_repo": DEFAULT_TOKENIZER_ASSETS_REPO,
                "tune_llm": False,
                "tune_visual": False,
                "project_to_dim": preset["project_to_dim"],
                "lora_rank": 0,
                "lora_alpha": 16,
                "lora_dropout": 0.1,
                "lora_full_model": False,
            }
        if not isinstance(config.action_head_cfg, dict) or not config.action_head_cfg:
            # Minimal defaults; adjust if your action space differs
            config.action_head_cfg = {
                "action_dim": getattr(config, "action_dim", 32),
                "action_horizon": getattr(config, "action_horizon", 16),
                "max_state_dim": getattr(config, "max_state_dim", 64),
                "max_action_dim": getattr(config, "action_dim", 32),
                "max_num_embodiments": 32,
                "input_embedding_dim": preset["input_embedding_dim"],
                "backbone_embedding_dim": preset["backbone_embedding_dim"],
                "vl_projection_hidden_dim": preset["vl_projection_hidden_dim"],
                "hidden_size": preset["hidden_size"],
                "vl_self_attention_cfg": preset["vl_self_attention_cfg"],
                "diffusion_model_cfg": preset["diffusion_model_cfg"],
            }

        self.backbone = InternVL3Backbone(**config.backbone_cfg)
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
        tune_visual = kwargs.pop("tune_visual", False)
        tune_llm = kwargs.pop("tune_llm", False)
        tune_projector = kwargs.pop("tune_projector", True)
        tune_diffusion_model = kwargs.pop("tune_diffusion_model", True)
        use_bf16 = kwargs.get("use_bf16", True)  # 默认开启 bf16
        lora_rank = kwargs.pop("lora_rank", 0)
        lora_alpha = kwargs.pop("lora_alpha", 16)
        lora_dropout = kwargs.pop("lora_dropout", 0.1)
        lora_full_model = kwargs.pop("lora_full_model", False)
        scale = kwargs.pop("scale", "medium").lower()
        tokenizer_assets_repo = kwargs.pop("tokenizer_assets_repo", DEFAULT_TOKENIZER_ASSETS_REPO)

        print(f"Loading pretrained dual brain from {pretrained_model_name_or_path} (Manual Mode)")

        try:
            local_model_path = snapshot_download(pretrained_model_name_or_path, repo_type="model")
        except (HFValidationError, RepositoryNotFoundError):
            local_model_path = pretrained_model_name_or_path

        preset = SCALE_PRESETS.get(scale)
        if preset is None:
            raise ValueError(f"Unknown scale '{scale}'. Expected one of {list(SCALE_PRESETS)}")

        config = cls.config_class()
        config.scale = scale
        config.backbone_cfg = {
            "model_path": local_model_path,
            "tokenizer_assets_repo": tokenizer_assets_repo,
            "tune_llm": tune_llm,
            "tune_visual": tune_visual,
            "project_to_dim": preset["project_to_dim"],
            "load_bf16": use_bf16,
            "lora_rank": lora_rank,
            "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout,
            "lora_full_model": lora_full_model,
        }

        if not config.action_head_cfg:
            config.action_head_cfg = {
                "action_dim": getattr(config, "action_dim", 32),
                "action_horizon": getattr(config, "action_horizon", 16),
                "max_state_dim": getattr(config, "max_state_dim", 64),
                "max_action_dim": getattr(config, "action_dim", 32),
                "max_num_embodiments": 32,
                "input_embedding_dim": preset["input_embedding_dim"],
                "backbone_embedding_dim": preset["backbone_embedding_dim"],
                "vl_projection_hidden_dim": preset["vl_projection_hidden_dim"],
                "hidden_size": preset["hidden_size"],
                "vl_self_attention_cfg": preset["vl_self_attention_cfg"],
                "diffusion_model_cfg": preset["diffusion_model_cfg"],
            }

        model = cls(config, local_model_path=local_model_path)

        model.backbone.set_trainable_parameters(tune_visual=tune_visual, tune_llm=tune_llm)
        model.action_head.set_trainable_parameters(
            tune_projector=tune_projector, tune_diffusion_model=tune_diffusion_model
        )

        return model
