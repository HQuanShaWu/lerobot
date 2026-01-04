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

import json
import os
from pathlib import Path
import shutil
from collections import deque

import torch
from torch import Tensor
from huggingface_hub.constants import SAFETENSORS_SINGLE_FILE
from safetensors.torch import load_file as load_file_as_safetensor, save_model as save_model_as_safetensor

from lerobot.policies.efficientvla.configuration_efficientvla import EfficientVLAConfig
from lerobot.policies.efficientvla.efficientvla_v1 import EfficientVLAV1
from lerobot.policies.pretrained import PreTrainedPolicy


class EfficientVLAPolicy(PreTrainedPolicy):
    """Wrapper around external EfficientVLA model for LeRobot integration."""

    name = "efficientvla"
    config_class = EfficientVLAConfig

    def __init__(self, config: EfficientVLAConfig, *, skip_init_load: bool = False):
        """Initialize EfficientVLA policy wrapper."""
        super().__init__(config)
        config.validate_features()
        self.config = config

        # Initialize model using ported components
        self._efficientvla_model = None
        if not skip_init_load:
            self._efficientvla_model = self._create_efficientvla_model()

        self.reset()

    @classmethod
    def from_pretrained(
        cls,
        pretrained_name_or_path: str | Path,
        *,
        config: EfficientVLAConfig | None = None,
        force_download: bool = False,
        resume_download: bool | None = None,
        proxies: dict | None = None,
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
        strict: bool = False,
        **kwargs,
    ) -> "EfficientVLAPolicy":
        if config is None:
            config = EfficientVLAConfig.from_pretrained(
                pretrained_name_or_path=pretrained_name_or_path,
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                token=token,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
                revision=revision,
                **kwargs,
            )

        model_id = Path(pretrained_name_or_path)
        if model_id.is_dir() and model_id.name == "vlm":
            candidate_root = model_id.parent
            if (candidate_root / "heads").is_dir():
                model_id = candidate_root
        vlm_dir = model_id / "vlm"
        heads_dir = model_id / "heads"
        if model_id.is_dir() and vlm_dir.is_dir() and heads_dir.is_dir():
            policy = cls(config, skip_init_load=True)
            config_path = vlm_dir / "config.json"
            use_vlm_dir = True
            if config_path.exists():
                try:
                    with config_path.open("r", encoding="utf-8") as config_file:
                        cfg = json.load(config_file)
                    if cfg.get("model_type") != "internvl_chat":
                        use_vlm_dir = False
                    if not cfg.get("vision_config") or not cfg.get("llm_config"):
                        use_vlm_dir = False
                except Exception:
                    use_vlm_dir = False
            else:
                use_vlm_dir = False
            vlm_load_path = str(vlm_dir) if use_vlm_dir else policy.config.base_model_path
            assets_repo = str(vlm_dir) if use_vlm_dir else policy.config.tokenizer_assets_repo
            if not use_vlm_dir:
                print(
                    "[EFFICIENTVLA] VLM config is not a full InternVL3 config. "
                    "Falling back to base_model_path for VLM; heads will still load."
                )
            try:
                policy._efficientvla_model = EfficientVLAV1.from_pretrained(
                    pretrained_model_name_or_path=vlm_load_path,
                    tokenizer_assets_repo=assets_repo,
                    tune_llm=policy.config.tune_llm,
                    tune_visual=policy.config.tune_visual,
                    tune_projector=policy.config.tune_projector,
                    tune_diffusion_model=policy.config.tune_diffusion_model,
                    lora_rank=0,
                    lora_alpha=policy.config.lora_alpha,
                    lora_dropout=policy.config.lora_dropout,
                    lora_full_model=False,
                    scale=getattr(policy.config, "scale", "medium"),
                    use_bf16=policy.config.use_bf16,
                )
            except ModuleNotFoundError as exc:
                print(
                    "[EFFICIENTVLA] VLM checkpoint is missing InternVL3 remote code. "
                    "Falling back to base_model_path for VLM; heads will still load."
                )
                policy._efficientvla_model = EfficientVLAV1.from_pretrained(
                    pretrained_model_name_or_path=policy.config.base_model_path,
                    tokenizer_assets_repo=policy.config.tokenizer_assets_repo,
                    tune_llm=policy.config.tune_llm,
                    tune_visual=policy.config.tune_visual,
                    tune_projector=policy.config.tune_projector,
                    tune_diffusion_model=policy.config.tune_diffusion_model,
                    lora_rank=0,
                    lora_alpha=policy.config.lora_alpha,
                    lora_dropout=policy.config.lora_dropout,
                    lora_full_model=False,
                    scale=getattr(policy.config, "scale", "medium"),
                    use_bf16=policy.config.use_bf16,
                )
            policy._load_split_heads(heads_dir, strict=strict)
            policy.to(policy.config.device)
            policy.eval()
            return policy

        return super().from_pretrained(
            pretrained_name_or_path,
            config=config,
            force_download=force_download,
            resume_download=resume_download,
            proxies=proxies,
            token=token,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            revision=revision,
            strict=strict,
            **kwargs,
        )

    def _load_split_heads(self, heads_dir: Path, *, strict: bool) -> None:
        head_file = heads_dir / SAFETENSORS_SINGLE_FILE
        if not head_file.exists():
            raise FileNotFoundError(f"Missing heads checkpoint: {head_file}")
        state = load_file_as_safetensor(str(head_file))
        projector_state = {k.replace("projector.", "", 1): v for k, v in state.items() if k.startswith("projector.")}
        action_state = {k.replace("action_head.", "", 1): v for k, v in state.items() if k.startswith("action_head.")}
        self._efficientvla_model.backbone.project.load_state_dict(projector_state, strict=strict)
        self._efficientvla_model.action_head.load_state_dict(action_state, strict=strict)

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
            tokenizer_assets_repo=self.config.tokenizer_assets_repo,
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

    def _save_pretrained(self, save_directory: Path) -> None:
        save_directory = Path(save_directory)
        self.config._save_pretrained(save_directory)
        vlm_dir = save_directory / "vlm"
        heads_dir = save_directory / "heads"
        vlm_dir.mkdir(parents=True, exist_ok=True)
        heads_dir.mkdir(parents=True, exist_ok=True)

        backbone = self._efficientvla_model.backbone
        vlm_model = backbone.model
        if backbone.use_lora:
            language_model = getattr(vlm_model, "language_model", None)
            if language_model is not None and hasattr(language_model, "merge_and_unload"):
                vlm_model.language_model = language_model.merge_and_unload()
            elif hasattr(vlm_model, "merge_and_unload"):
                vlm_model = vlm_model.merge_and_unload()
                backbone.model = vlm_model
        vlm_model.save_pretrained(str(vlm_dir), safe_serialization=True)

        config_path = vlm_dir / "config.json"
        assets_dir = Path(self.config.tokenizer_assets_repo).expanduser()
        assets_config = assets_dir / "config.json"
        if assets_config.exists():
            needs_replace = False
            try:
                with config_path.open("r", encoding="utf-8") as config_file:
                    saved_cfg = json.load(config_file)
                if saved_cfg.get("model_type") != "internvl_chat":
                    needs_replace = True
                if not saved_cfg.get("vision_config") or not saved_cfg.get("llm_config"):
                    needs_replace = True
            except Exception:
                needs_replace = True
            if needs_replace:
                shutil.copy2(assets_config, config_path)

        try:
            backbone.tokenizer.save_pretrained(str(vlm_dir))
        except Exception:
            pass

        assets_dir = Path(self.config.tokenizer_assets_repo).expanduser()
        if assets_dir.exists():
            for src in assets_dir.iterdir():
                if not src.is_file():
                    continue
                if src.name == SAFETENSORS_SINGLE_FILE:
                    continue
                dst = vlm_dir / src.name
                if not dst.exists():
                    shutil.copy2(src, dst)

        class _HeadBundle(torch.nn.Module):
            def __init__(self, projector, action_head):
                super().__init__()
                self.projector = projector
                self.action_head = action_head

        head_bundle = _HeadBundle(backbone.project, self._efficientvla_model.action_head)
        save_model_as_safetensor(head_bundle, str(heads_dir / SAFETENSORS_SINGLE_FILE))

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
            "internvl3_inputs",
            "input_ids",
            "attention_mask",
            "pixel_values",
            "image_flags",
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
            "internvl3_inputs",
            "input_ids",
            "attention_mask",
            "pixel_values",
            "image_flags",
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

        # execution_horizon = 8

        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(batch)
            # actions = actions[:, :execution_horizon, :]
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
