# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Input, Path

import torch
import numpy as np
import random
import os
import shutil
from typing import List

os.environ["HF_HUB_CACHE"] = "models"
os.environ["HF_HUB_CACHE_OFFLINE"] = "true"

from diffusers.utils import load_image
from diffusers import EulerDiscreteScheduler

from huggingface_hub import hf_hub_download
# import spaces

# import gradio as gr

from photomaker.pipeline import PhotoMakerStableDiffusionXLPipeline
from gradio_demo.style_template import styles

MAX_SEED = np.iinfo(np.int32).max
STYLE_NAMES = list(styles.keys())
DEFAULT_STYLE_NAME = "Photographic (Default)"

# utility function for style templates
def apply_style(style_name: str, positive: str, negative: str = "") -> tuple[str, str]:
    p, n = styles.get(style_name, styles[DEFAULT_STYLE_NAME])
    return p.replace("{prompt}", positive), n + " " + negative

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        # self.model = torch.load("./weights.pth")

        base_model_path = "SG161222/RealVisXL_V3.0"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"



        # download PhotoMaker checkpoint to cache
        # if we already have the model, this doesn't do anything
        photomaker_ckpt = hf_hub_download(
            repo_id="TencentARC/PhotoMaker",
            filename="photomaker-v1.bin",
            repo_type="model",
        )

        self.pipe = PhotoMakerStableDiffusionXLPipeline.from_pretrained(
            base_model_path,
            torch_dtype=torch.bfloat16,
            use_safetensors=True,
            variant="fp16",
        ).to(self.device)

        self.pipe.load_photomaker_adapter(
            os.path.dirname(photomaker_ckpt),
            subfolder="",
            weight_name=os.path.basename(photomaker_ckpt),
            trigger_word="img",
        )

        self.pipe.scheduler = EulerDiscreteScheduler.from_config(
            self.pipe.scheduler.config
        )
        # pipe.set_adapters(["photomaker"], adapter_weights=[1.0])
        self.pipe.fuse_lora()

    def predict(
        self,
        # from ChatGPT
        input_image: Path = Input(
            description="The input image, a photo of your face"
        ),
        prompt: str = Input(
            description="Prompt. Example: 'a photo of a man/woman img'. The phrase 'img' is the trigger word.",
            default="A photo of a person img",
        ),
        style_name: str = Input(
            description="Style template",
            choices=STYLE_NAMES,
            default=DEFAULT_STYLE_NAME,
        ),
        negative_prompt: str = Input(
            description="Negative Prompt",
            default="nsfw, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry",
        ),
        num_steps: int = Input(
            description="Number of sample steps", default=50, ge=20, le=100
        ),
        style_strength_ratio: float = Input(
            description="Style strength (%)", default=20, ge=15, le=50
        ),
        #num_outputs: int = Input(
        #    description="Number of output images", default=1, ge=1, le=4
        #),
        guidance_scale: float = Input(
            description="Guidance scale", default=5, ge=0.1, le=10.0
        ),
        seed: int = Input(description="Seed. Leave blank to use a random number", default=None, ge=0, le=MAX_SEED)
    ) -> List[Path]:
        """Run a single prediction on the model"""
        # remove old outputs
        output_folder = Path('outputs')
        if output_folder.exists():
            shutil.rmtree(output_folder)
        os.makedirs(str(output_folder), exist_ok=False)

        # randomize seed if necessary
        if seed is None:
            seed = random.randint(0, MAX_SEED)

        # check the prompt for the trigger word
        image_token_id = self.pipe.tokenizer.convert_tokens_to_ids(self.pipe.trigger_word)
        input_ids = self.pipe.tokenizer.encode(prompt)
        if image_token_id not in input_ids:
            raise ValueError(
                f"Cannot find the trigger word '{self.pipe.trigger_word}' in text prompt!")

        if input_ids.count(image_token_id) > 1:
            raise ValueError(
                f"Cannot use multiple trigger words '{self.pipe.trigger_word}' in text prompt!"
            )

        # apply the style template
        prompt, negative_prompt = apply_style(style_name, prompt, negative_prompt)

        input_id_images = load_image(str(input_image))

        generator = torch.Generator(device=self.device).manual_seed(seed)

        print("Start inference...")
        print(f"[Debug] Prompt: {prompt}")
        print(f"[Debug] Neg Prompt: {negative_prompt}")
        start_merge_step = int(float(style_strength_ratio) / 100 * num_steps)
        if start_merge_step > 30:
            start_merge_step = 30
        print(start_merge_step)
        images = self.pipe(
            prompt=prompt,
            input_id_images=input_id_images,
            negative_prompt=negative_prompt,
            num_images_per_prompt=1, # used to be: num_outputs but currently we accept only one input
            num_inference_steps=num_steps,
            start_merge_step=start_merge_step,
            generator=generator,
            guidance_scale=guidance_scale,
        ).images
        
        # save results to file
        output_paths = []
        for i, image in enumerate(images):
            output_path = output_folder / f"image_{i}.png"
            image.save(output_path)
            output_paths.append(output_path)
        return [Path(p) for p in output_paths]
