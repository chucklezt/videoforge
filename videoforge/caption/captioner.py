"""Stage 2: Visual captioning with VLM (Qwen2-VL)."""

import logging
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

DEFAULT_PROMPT = """Describe this video clip in detail for training a video generation model. Include:

1. SCENE: Setting, location, time of day, lighting conditions
2. SUBJECTS: People present, their appearance, clothing, positioning
3. ACTION: What is happening, movements, gestures, expressions
4. CAMERA: Camera angle, movement (static, pan, zoom, tracking)
5. STYLE: Color palette, mood, visual style (cinematic, bright, dark, etc.)

Write a single flowing paragraph, not a bulleted list. Be specific and visual.
Do not describe audio or make assumptions about what cannot be seen."""


class VideoCaptioner:
    """Video captioning using Qwen2-VL."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2-VL-7B-Instruct",
        quantization: str = "4bit",
        dtype: torch.dtype = torch.float16,
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

        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

        load_kwargs = {
            "device_map": "auto",
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

        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
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
                        "video": clip_path,
                        "max_pixels": 360 * 420,
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

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
            )

        output_text = self.processor.batch_decode(
            output_ids[:, inputs.input_ids.shape[1]:],
            skip_special_tokens=True,
        )[0]

        return output_text.strip()
