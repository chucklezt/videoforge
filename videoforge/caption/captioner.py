"""Stage 2: Visual captioning with VLM (Qwen2-VL)."""

import logging
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

DEFAULT_PROMPT = """Describe this video clip in detail for training a video generation model. Include:

SCENE: Setting, location, time of day, lighting conditions. Be specific -- if this is a diner, describe the booth, counter, décor. If an apartment, describe the furniture and layout.
SUBJECTS: Identify any recognizable characters by name if possible. Describe their appearance, clothing, body type, positioning. George Costanza is a short, stocky, bald man who appears frequently.
ACTION: What is happening, movements, gestures, expressions, body language.
CAMERA: Camera angle, movement (static, pan, zoom, tracking shot).
STYLE: Color palette, mood, visual style, lighting quality.

Write a single flowing paragraph, not a bulleted list. Be specific and visual. Refer to characters by name when recognizable. Do not describe audio or make assumptions about what cannot be seen."""


class VideoCaptioner:
    """Video captioning using Qwen2-VL."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        quantization: str = "none",
        dtype: torch.dtype = torch.bfloat16,
        max_new_tokens: int = 300,
        prompt: str = DEFAULT_PROMPT,
        fps_sample: float = 4.0,
    ):
        self.model_name = model_name
        self.quantization = quantization
        self.dtype = dtype
        self.max_new_tokens = max_new_tokens
        self.prompt = prompt
        self.fps_sample = fps_sample
        self.model = None
        self.processor = None

    def load(self):
        """Load the captioning model and processor."""
        if self.model is not None:
            return

        logger.info("Loading captioning model: %s (%s)", self.model_name, self.quantization)

        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

        load_kwargs = {
            "device_map": {"": "cuda:0"},
            "torch_dtype": self.dtype,
        }

        if self.quantization == "4bit":
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=self.dtype,
                bnb_4bit_quant_type="nf4",
            )
        elif self.quantization == "8bit":
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=True,
            )

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_name, **load_kwargs
        )
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        logger.info("Model loaded successfully")

    def unload(self):
        """Free model from GPU memory."""
        if self.model is not None:
            del self.model
            del self.processor
            self.model = None
            self.processor = None
            torch.cuda.empty_cache()
            logger.info("Model unloaded, VRAM freed")

    def caption_clip(self, clip_path: str | Path) -> str:
        """Generate a caption for a single video clip."""
        if self.model is None:
            self.load()

        from qwen_vl_utils import process_vision_info

        clip_path = str(clip_path)

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": f"file://{Path(clip_path).resolve()}",
                        "min_pixels": 64 * 64,
                        "max_pixels": 224 * 224,
                        "fps": self.fps_sample,
                    },
                    {
                        "type": "text",
                        "text": self.prompt,
                    },
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        input_len = inputs.input_ids.shape[1]

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
            )

        del inputs
        torch.cuda.empty_cache()

        output_text = self.processor.batch_decode(
            output_ids[:, input_len:],
            skip_special_tokens=True,
        )[0]

        return output_text.strip()
