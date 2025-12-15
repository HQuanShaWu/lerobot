#!/usr/bin/env python

# Copyright 2024 NVIDIA Corporation and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
EfficientVLA Policy Wrapper for LeRobot Integration

Minimal integration that delegates to the ported components while keeping the
Groot-derived implementation intact. The intent is to:

- Download and load the pretrained model via EfficientVLAN15.from_pretrained
- Optionally align action horizon similar to gr00t_finetune.py
- Expose predict_action via model.get_action
- Provide a training forward that can call the model forward if batch structure matches.
"""

import os
from collections import deque

import torch
from torch import Tensor

from lerobot.policies.efficientvla.configuration_efficientvla import EfficientVLAConfig
from lerobot.policies.efficientvla.efficientvla_v1 import EfficientVLAV1
from lerobot.policies.pretrained import PreTrainedPolicy


class EfficientVLAPolicy(PreTrainedPolicy):
    """Wrapper around external EfficientVLA model for LeRobot integration."""

    name = "efficientvla"
    config_class = EfficientVLAConfig

    def __init__(self, config: EfficientVLAConfig):
        """Initialize EfficientVLA policy wrapper."""
        super().__init__(config)
        config.validate_features()
        self.config = config

        # Initialize model using ported components
        self._efficientvla_model = self._create_efficientvla_model()

        self.reset()

    def _create_efficientvla_model(self):
        """Create and initialize the EfficientVLA model using ported API.

        This is only called when creating a NEW policy (not when loading from checkpoint).

        Steps (delegating to ported components):
        1) Download and load pretrained model via EfficientVLAN15.from_pretrained
        2) Align action horizon with data_config if provided
        """
        # Handle Flash Attention compatibility issues
        self._handle_flash_attention_compatibility()

        model = EfficientVLAV1.from_pretrained(
            pretrained_model_name_or_path=self.config.base_model_path,
            tune_llm=self.config.tune_llm,
            tune_visual=self.config.tune_visual,
            tune_projector=self.config.tune_projector,
            tune_diffusion_model=self.config.tune_diffusion_model,
            lora_rank=self.config.lora_rank,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            lora_full_model=self.config.lora_full_model,
            scale=getattr(self.config, "scale", "medium"),
        )

        model.compute_dtype = "bfloat16" if self.config.use_bf16 else model.compute_dtype
        model.config.compute_dtype = model.compute_dtype

        return model

    def reset(self):
        """Reset policy state when environment resets."""
        self._action_queue = deque([], maxlen=self.config.n_action_steps)

    def get_optim_params(self) -> dict:
        return self.parameters()

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        """Training forward pass."""
        # Build a clean input dict for the model: keep only tensors it consumes
        allowed_base = {
            "state",
            "state_mask",
            "action",
            "action_mask",
            "embodiment_id",
            "input_ids",
            "attention_mask",
            "pixel_values",
            "image_grid_thw",
        }
        efficientvla_inputs = {
            k: v
            for k, v in batch.items()
            if (k in allowed_base) and not (k.startswith("next.") or k == "info")
        }

        # Get device from model parameters
        device = next(self.parameters()).device

        # Run forward under bf16 autocast when enabled to reduce activation memory
        # Rationale: Matches original finetuning (bf16 compute, fp32 params) and avoids fp32 upcasts.
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=self.config.use_bf16):
            outputs = self._efficientvla_model.forward(efficientvla_inputs)

        # Ported model returns a BatchFeature; loss key is typically 'loss'
        loss = outputs.get("loss")

        loss_dict = {"loss": loss.item()}

        return loss, loss_dict

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        """Predict a chunk of actions for inference by delegating to the ported model.

        Returns a tensor of shape (B, n_action_steps, action_dim).
        """
        self.eval()

        # Build a clean input dict for the model: keep only tensors it consumes
        # Preprocessing is handled by the processor pipeline, so we just filter the batch
        # NOTE: During inference, we should NOT pass action/action_mask (that's what we're predicting)
        allowed_base = {
            "state",
            "state_mask",
            "action",
            "action_mask",
            "embodiment_id",
            "input_ids",
            "attention_mask",
            "pixel_values",
            "image_grid_thw",
        }
        efficientvla_inputs = {
            k: v
            for k, v in batch.items()
            if (k in allowed_base) and not (k.startswith("next.") or k == "info")
        }

        # Get device from model parameters
        device = next(self.parameters()).device

        # Use bf16 autocast for inference to keep memory low and match backbone dtype
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=self.config.use_bf16):
            outputs = self._efficientvla_model.get_action(efficientvla_inputs)

        actions = outputs.get("action_pred")

        original_action_dim = self.config.output_features["action"].shape[0]
        actions = actions[:, :, :original_action_dim]

        return actions

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Select single action from action queue."""
        self.eval()

        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(batch)
            self._action_queue.extend(actions.transpose(0, 1))
        return self._action_queue.popleft()

    # -------------------------
    # Internal helpers
    # -------------------------
    def _handle_flash_attention_compatibility(self) -> None:
        """Handle Flash Attention compatibility issues by setting environment variables.

        This addresses the common 'undefined symbol' error that occurs when Flash Attention
        is compiled against a different PyTorch version than what's currently installed.
        """

        # Set environment variables to handle Flash Attention compatibility
        # These help with symbol resolution issues
        os.environ.setdefault("FLASH_ATTENTION_FORCE_BUILD", "0")
        os.environ.setdefault("FLASH_ATTENTION_SKIP_CUDA_BUILD", "0")

        # Try to import flash_attn and handle failures gracefully
        try:
            import flash_attn

            print(f"[EFFICIENTVLA] Flash Attention version: {flash_attn.__version__}")
        except ImportError as e:
            print(f"[EFFICIENTVLA] Flash Attention not available: {e}")
            print("[EFFICIENTVLA] Will use fallback attention mechanism")
        except Exception as e:
            if "undefined symbol" in str(e):
                print(f"[EFFICIENTVLA] Flash Attention compatibility issue detected: {e}")
                print("[EFFICIENTVLA] This is likely due to PyTorch/Flash Attention version mismatch")
                print("[EFFICIENTVLA] Consider reinstalling Flash Attention with compatible version:")
                print("  pip uninstall flash-attn")
                print("  pip install --no-build-isolation flash-attn==2.6.3")
                print("[EFFICIENTVLA] Continuing with fallback attention mechanism")
            else:
                print(f"[EFFICIENTVLA] Flash Attention error: {e}")
                print("[EFFICIENTVLA] Continuing with fallback attention mechanism")
