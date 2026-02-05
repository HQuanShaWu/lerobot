import os
import re
import time
import cv2
import torch
import importlib
from typing import Union, List, Optional, Tuple

from transformers import AutoTokenizer, AutoModel
from transformers.generation import BaseStreamer

# =========================================================
# 1) Speed Monitor Streamer (TTFT / TPS)
# =========================================================
class SpeedMonitorStreamer(BaseStreamer):
    """
    用于监控生成速度的 Streamer（统计 TTFT / TPS）。
    HuggingFace generate 在每步生成 token 时会调用 streamer.put(...)
    """
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.start_time = None
        self.first_token_time = None
        self.end_time = None
        self.token_count = 0

    def put(self, value):
        now = time.time()
        if self.start_time is None:
            self.start_time = now
        if self.token_count == 0:
            self.first_token_time = now

        self.token_count += 1
        self.end_time = now

    def end(self):
        pass

    def get_stats(self, prefill_start_time: float):
        if self.first_token_time is None:
            return None

        ttft = self.first_token_time - prefill_start_time
        decoding_duration = (self.end_time - self.first_token_time) if self.end_time else 0.0
        if decoding_duration > 0 and self.token_count > 1:
            tps = (self.token_count - 1) / decoding_duration
        else:
            tps = 0.0

        return {
            "token_count": self.token_count,
            "ttft_sec": ttft,
            "gen_duration_sec": decoding_duration,
            "tps": tps,
        }

# =========================================================
# 2) InternVL image preprocess (官方 dynamic tiling 思路)
#    来源：InternVL3 Quick Start / README 的示例逻辑
# =========================================================
from PIL import Image
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

def build_transform(input_size: int):
    return T.Compose([
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

def dynamic_preprocess(
    image: Image.Image,
    image_size: int = 448,
    use_thumbnail: bool = True,
    max_num: int = 12,
):
    """
    将任意分辨率图片切成若干 image_size x image_size 的 tile（最多 max_num 个），可额外加缩略图 tile。
    这与 InternVL3 官方 quick start 的示例一致。 :contentReference[oaicite:1]{index=1}
    """
    w, h = image.size
    aspect_ratio = w / h

    # 选择一个 tiles 布局（尽量接近原始比例），且 tiles 数不超过 max_num
    # 这里用一个简单策略：枚举 grid (gw, gh) 使 gw*gh<=max_num，且 gw/gh 接近 aspect_ratio
    best_gw, best_gh = 1, 1
    best_diff = 1e9
    for gh in range(1, max_num + 1):
        for gw in range(1, max_num + 1):
            if gw * gh > max_num:
                continue
            diff = abs((gw / gh) - aspect_ratio)
            if diff < best_diff:
                best_diff = diff
                best_gw, best_gh = gw, gh

    target_w = best_gw * image_size
    target_h = best_gh * image_size
    resized = image.resize((target_w, target_h), resample=Image.BICUBIC)

    tiles = []
    for j in range(best_gh):
        for i in range(best_gw):
            left = i * image_size
            upper = j * image_size
            right = left + image_size
            lower = upper + image_size
            tiles.append(resized.crop((left, upper, right, lower)))

    if use_thumbnail and len(tiles) != 1:
        tiles.append(image.resize((image_size, image_size), resample=Image.BICUBIC))

    return tiles

def load_image(image_file: str, input_size: int = 448, max_num: int = 12, use_thumbnail: bool = True):
    image = Image.open(image_file).convert("RGB")
    transform = build_transform(input_size=input_size)
    tiles = dynamic_preprocess(image, image_size=input_size, use_thumbnail=use_thumbnail, max_num=max_num)
    pixel_values = torch.stack([transform(t) for t in tiles])
    return pixel_values  # [n_tiles, 3, H, W]

# =========================================================
# 3) Unified Inference for InternVL3 (like RoboBrain script)
# =========================================================
class UnifiedInferenceInternVL3:
    """
    仿照 test_robobrain2.0-3B.py 的功能：
    - inference(): 推理 + TTFT/TPS
    - get_action_condition(): forward 一次拿最后一层 hidden state 的最后 token
    - plot: 对 pointing/grounding/trajectory 等画图
    """

    def __init__(
        self,
        model_dir: str,
        device_map: str = "auto",
        torch_dtype: Optional[torch.dtype] = None,
    ):
        print(f"Loading InternVL3 checkpoint from: {model_dir}")

        # dtype：默认优先 bf16（多数 A100/新卡可用），否则 fp16
        if torch_dtype is None:
            if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8:
                torch_dtype = torch.bfloat16
            else:
                torch_dtype = torch.float16

        self.model_dir = model_dir
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True, use_fast=False)

        # 官方推荐 AutoModel + trust_remote_code，再用 model.chat。 :contentReference[oaicite:2]{index=2}
        self.model = AutoModel.from_pretrained(
            model_dir,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map=device_map,
        ).eval()

        # InternVL 的 chat 里用到的 token 字符串（与官方 modeling_internvl_chat.py 一致） :contentReference[oaicite:3]{index=3}
        self.IMG_START_TOKEN = "<img>"
        self.IMG_END_TOKEN = "</img>"
        self.IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"

        # 设备
        try:
            self.device = self.model.device
        except Exception:
            self.device = next(self.model.parameters()).device

        self.dtype = torch_dtype
        print(f"Running on device: {self.device}, dtype: {self.dtype}")

        # 动态导入 conversation.get_conv_template（用于 get_action_condition 复刻 chat 的 prompt 构造）
        self.get_conv_template = self._resolve_get_conv_template()

    def _resolve_get_conv_template(self):
        """
        从 model 的 remote_code 模块里定位 conversation.get_conv_template
        """
        model_mod = self.model.__class__.__module__            # e.g. transformers_modules.xxx.modeling_internvl_chat
        pkg = model_mod.rsplit(".", 1)[0]                      # e.g. transformers_modules.xxx
        conv_mod = importlib.import_module(pkg + ".conversation")
        return getattr(conv_mod, "get_conv_template")

    def _prepare_images(
        self,
        image: Union[str, List[str], None],
        input_size: int = 364,     # 每个 tile resize 到 input_size x input_size
        max_num: int = 6,          # 每张图最多切成 max_num 个 tile（可能额外+thumbnail）
        use_thumbnail: bool = True # 是否额外加入缩略图 tile（通常能提升鲁棒性）
    ) -> Tuple[Optional[torch.Tensor], List[int]]:
        """
        返回:
        pixel_values: [sum_tiles, 3, H, W] or None
        num_patches_list: 每张图各自 tile 数，用于 multi-image 的 chat
        """
        if image is None:
            return None, []

        if isinstance(image, str):
            image_list = [image]
        else:
            if not isinstance(image, list):
                raise TypeError(f"image must be str|list[str]|None, got {type(image)}")
            image_list = image
            if not all(isinstance(p, str) for p in image_list):
                raise TypeError("image list must be list[str] (each item is an image path)")

        pixel_list: List[torch.Tensor] = []
        num_patches_list: List[int] = []

        for p in image_list:
            pv = load_image(p, input_size=input_size, max_num=max_num, use_thumbnail=use_thumbnail)
            num_patches_list.append(int(pv.size(0)))
            pixel_list.append(pv)

        pixel_values = torch.cat(pixel_list, dim=0).contiguous()  # [sum_tiles, 3, H, W]
        pixel_values = pixel_values.to(device=self.device, dtype=self.dtype, non_blocking=True)
        return pixel_values, num_patches_list


    def inference(
        self,
        text: str,
        image: Union[str, List[str]],
        task: str = "general",
        plot: bool = False,
        do_sample: bool = True,
        temperature: float = 0.7,
        max_new_tokens: int = 768,
        input_size: int = 364,
        max_num: int = 6,
        use_thumbnail: bool = True,
    ):
        # 任务类型校验
        assert task in ["general", "static", "prediction", "grounding"], \
            f"Invalid task={task} (expected: general/static/prediction/grounding)"

        text = "" if text is None else str(text)

        # 1) 图片 -> pixel_values / num_patches_list（你原来下面会用到它俩）
        # input_size: tile 的边长；max_num: 最多切多少 tile；use_thumbnail: 是否加缩略图 tile
        pixel_values, num_patches_list = self._prepare_images(
            image=image,
            input_size=input_size,
            max_num=max_num,
            use_thumbnail=use_thumbnail,
        )

        # 2) 构造 task 对应的文本（重点：grounding 的 <ref>）
        if task == "general":
            # long_caption：如果没给 prompt，就给默认长描述提示
            if not text.strip():
                text = "Please describe in detail the scene and the objects in the image."

        elif task == "grounding":
            # 目标格式（你微调时用的）：
            # Please provide the bounding box coordinate of the region this sentence describes: <ref>...</ref>
            instr = "Please provide the bounding box coordinate of the region this sentence describes"

            has_ref = ("<ref>" in text) and ("</ref>" in text)
            has_instr = (instr.lower() in text.lower())

            if has_instr and not has_ref:
                # 用户已经给了完整指令，但没给 <ref>：尽量把冒号后内容包进 <ref>
                if ":" in text:
                    prefix, desc = text.split(":", 1)
                    desc = desc.strip().rstrip(".")
                    text = f"{prefix.strip()}: <ref>{desc}</ref>"
                else:
                    # 没冒号就只能整体当作描述（保守处理）
                    desc = text.strip().rstrip(".")
                    text = f"{instr}: <ref>{desc}</ref>"

            elif (not has_instr) and has_ref:
                # 已经是 <ref>...</ref>，但没有指令：补齐指令
                # 注意：不要在 </ref> 后面额外加句号，避免影响你对输出格式的期待
                text = f"{instr}: {text.strip()}"

            elif (not has_instr) and (not has_ref):
                # 只有描述语句：自动包 <ref> 并加指令
                desc = text.strip().rstrip(".")
                text = f"{instr}: <ref>{desc}</ref>"

            # else: has_instr and has_ref -> 用户给的已经完全符合，不动

        elif task == "static":
            # robovqa_static：静态图像问答（当前帧/当前场景）
            # 如果调用方没给 prompt，就提供一个默认问题，避免空输入导致无意义输出
            if not text.strip():
                text = "What can the robot do immediately given the current scene?"

        elif task == "prediction":
            # robovqa_future_prediction：未来预测（基于当前画面推断下一步/未来状态）
            if not text.strip():
                text = "Please predict what will happen next in the scene."


        # 3) 确保 <image> 占位符数量与图片数量一致（InternVL 格式要求）
        n_images = len(num_patches_list)

        n_placeholders = text.count("<image>")

        if n_images == 0:
            if n_placeholders > 0:
                raise ValueError("Text-only input should not contain '<image>' placeholder.")
            question = text
        else:
            if n_placeholders == 0:
                # 默认前缀补齐 n_images 个 <image>\n
                question = ("<image>\n" * n_images) + text
            else:
                if n_placeholders != n_images:
                    raise ValueError(
                        f"Number of '<image>' placeholders ({n_placeholders}) must match "
                        f"number of images ({n_images})."
                    )
                question = text

        print(f"\n{'='*20} INPUT {'='*20}\n{question}\n{'='*47}\n")

        # 4) speed monitor（通过 generate 的 streamer 统计）
        speed_streamer = SpeedMonitorStreamer(self.tokenizer)
        generation_config = dict(
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            streamer=speed_streamer,
        )

        # 5) 计时：与原脚本一样，在“开始 generate 前”打点
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_start = time.time()

        # InternVL chat：多图/单图都走 model.chat
        if num_patches_list:
            response = self.model.chat(
                self.tokenizer,
                pixel_values,
                question,
                generation_config,
                num_patches_list=num_patches_list,
            )
        else:
            response = self.model.chat(
                self.tokenizer,
                None,
                question,
                generation_config,
            )

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # 6) 性能统计输出
        stats = speed_streamer.get_stats(prefill_start_time=t_start)
        print(f"\n{'='*10} Performance Stats {'='*10}")
        if stats:
            print(f"📊 TTFT (首字延迟):     {stats['ttft_sec']:.3f} s")
            print(f"🚀 Speed (生成速度):    {stats['tps']:.2f} tokens/s")
            print(f"🔢 Total Tokens:        {stats['token_count']}")
        else:
            print("No tokens streamed (maybe streamer not triggered).")
        print(f"{'='*47}\n")

        # 7) 画图：你的新任务里只有 grounding 需要画框（如你需要 static/prediction 也画，再扩展）
        if plot and task == "grounding":
            img_path = image if isinstance(image, str) else image[0]
            self._handle_plotting(img_path, response, task)

        return {"answer": response}


    def get_action_condition(
        self,
        text: str,
        image: Union[str, List[str]],
        input_size: int = 364,
        max_num: int = 6,
        use_thumbnail: bool = True,
    ):
        """
        复刻 InternVL 的 chat prompt 构造，然后 forward 一次拿 outputs.hidden_states[-1][:, -1, :]
        注意：InternVL 的 forward 里会用到 img_context_token_id / image_flags 等（见官方 modeling）。 :contentReference[oaicite:5]{index=5}
        """
        pixel_values, num_patches_list = self._prepare_images(
            image=image, input_size=input_size, max_num=max_num, use_thumbnail=use_thumbnail
        )

        # question 与 inference 保持一致
        if isinstance(image, str):
            question = f"<image>\n{text}"
        else:
            prefix = "".join([f"Image-{i+1}: <image>\n" for i in range(len(image))])
            question = prefix + text

        # ===== 按 modeling_internvl_chat.py 的 chat 逻辑构造 query ===== :contentReference[oaicite:6]{index=6}
        if num_patches_list is None:
            num_patches_list = [pixel_values.shape[0]] if pixel_values is not None else []

        # 设置 img_context_token_id（forward / generate 都会用到）
        img_context_token_id = self.tokenizer.convert_tokens_to_ids(self.IMG_CONTEXT_TOKEN)
        self.model.img_context_token_id = img_context_token_id

        template = self.get_conv_template(self.model.template)
        template.system_message = self.model.system_message
        template.append_message(template.roles[0], question)
        template.append_message(template.roles[1], None)
        query = template.get_prompt()

        for n_patch in num_patches_list:
            image_tokens = (
                self.IMG_START_TOKEN +
                (self.IMG_CONTEXT_TOKEN * self.model.num_image_token * n_patch) +
                self.IMG_END_TOKEN
            )
            query = query.replace("<image>", image_tokens, 1)

        model_inputs = self.tokenizer(query, return_tensors="pt")
        input_ids = model_inputs["input_ids"].to(self.device)
        attention_mask = model_inputs["attention_mask"].to(self.device)

        # InternVL forward 里 image_flags 会参与筛 vit_embeds（有些版本是必需的） :contentReference[oaicite:7]{index=7}
        image_flags = None
        if pixel_values is not None:
            image_flags = torch.ones((pixel_values.shape[0], 1), dtype=torch.long, device=self.device)

        with torch.no_grad():
            outputs = self.model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                image_flags=image_flags,
                output_hidden_states=True,
                return_dict=True,
            )

        last_layer = outputs.hidden_states[-1]          # [B, seq, hidden]
        action_condition = last_layer[:, -1, :]         # [B, hidden]
        print(f"Action Condition Extracted. Shape: {action_condition.shape}")
        return action_condition


    # ---------------- plot utils ----------------
    # 仅用于 grounding 任务：在原图上画出预测 bbox
    def _handle_plotting(self, image_path: str, result_text: str, task: str = "grounding"):
        if task != "grounding":
            # 保险：即使上层误调用，也不做任何绘制
            print(f"Plot skipped (task={task}). Only 'grounding' is supported for plotting.")
            return None

        print("Plotting enabled (grounding). Drawing bounding boxes on the image ...")

        # 支持输出包含 <box>[[x1, y1, x2, y2]]</box> 或直接 [[x1, y1, x2, y2]] 的情况
        box_pattern = r'\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]'
        boxes = re.findall(box_pattern, result_text)

        if not boxes:
            print("No bounding box found in model output. Skip saving annotated image.")
            return None

        plot_boxes = [[int(x1), int(y1), int(x2), int(y2)] for x1, y1, x2, y2 in boxes]

        image_name = os.path.basename(image_path)
        name, ext = os.path.splitext(image_name)
        save_name = f"{name}_grounding_annotated{ext}"
        os.makedirs("result", exist_ok=True)
        save_path = os.path.join("result", save_name)

        return self.draw_on_image(image_path=image_path, boxes=plot_boxes, output_path=save_path)

    def draw_on_image(self, image_path: str, boxes: Optional[List[List[int]]] = None, output_path: Optional[str] = None):
        '''
        仅支持 grounding：画 bbox（绿色框）。
        boxes: List[[x1, y1, x2, y2], ...]，默认按像素坐标绘制。
        '''
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise FileNotFoundError(f"Unable to read image: {image_path}")

            if boxes:
                for x1, y1, x2, y2 in boxes:
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            if output_path is None:
                os.makedirs("result", exist_ok=True)
                image_name = os.path.basename(image_path)
                name, ext = os.path.splitext(image_name)
                output_path = os.path.join("result", f"{name}_grounding_annotated{ext}")

            cv2.imwrite(output_path, image)
            print(f"Annotated image saved to: {output_path}")
            return output_path
        except Exception as e:
            print(f"Error processing image: {e}")
            return None


if __name__ == "__main__":
    MODEL_PATH = "/home/nvidia/internvl3_1b_sft"
    IMAGE_PATH = "/home/nvidia/embodied_debug_dump/long_caption/02_static_149852/img01_219439_aff.jpg"
    PROMPT = "What is shown in this image?"

    print("=== Initializing InternVL3 Model ===")
    bot = UnifiedInferenceInternVL3(MODEL_PATH)

    print("\n=== Test 1: Inference & Speed Test ===")
    result = bot.inference(
        text=PROMPT,
        image=IMAGE_PATH,
        task="general",
        plot=False,
        # 下面三个参数建议与你训练/测试设置对齐：
        input_size=364,      # tile 的边长，常见 364/448（你之前用过 364） :contentReference[oaicite:8]{index=8}
        max_num=6,           # 每张图最多切多少块 tile
        use_thumbnail=True,  # 是否额外加一张缩略图 tile（官方示例为 True） :contentReference[oaicite:9]{index=9}
        max_new_tokens=256,  # 你想压测速度可改大
    )
    print(f"Result: {result['answer']}")

    print("\n=== Test 2: Feature Extraction (last-layer hidden state) ===")
    condition = bot.get_action_condition(PROMPT, IMAGE_PATH, input_size=364, max_num=6, use_thumbnail=True)
    print("Condition extracted successfully.")
