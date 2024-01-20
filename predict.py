# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Input, Path

import torch
import numpy as np
import random
import os
import shutil
import subprocess
import time

os.environ["HF_HUB_CACHE"] = "models"
os.environ["HF_HUB_CACHE_OFFLINE"] = "true"

from diffusers.utils import load_image
from diffusers import EulerDiscreteScheduler
from diffusers.pipelines.stable_diffusion.safety_checker import (
    StableDiffusionSafetyChecker,
)

from huggingface_hub import hf_hub_download

from transformers import CLIPImageProcessor

from photomaker import PhotoMakerStableDiffusionXLPipeline
from gradio_demo.style_template import styles

MAX_SEED = np.iinfo(np.int32).max
STYLE_NAMES = list(styles.keys())
DEFAULT_STYLE_NAME = "Photographic (Default)"

FEATURE_EXTRACTOR = "./feature-extractor"
SAFETY_CACHE = "./models/safety-cache"
SAFETY_URL = "https://weights.replicate.delivery/default/sdxl/safety-1.0.tar"

BASE_MODEL_URL = "https://weights.replicate.delivery/default/SG161222--RealVisXL_V3.0-11ee564ebf4bd96d90ed5d473cb8e7f2e6450bcf.tar"
BASE_MODEL_PATH = "models/SG161222/RealVisXL_V3.0"

PHOTOMAKER_URL = "https://weights.replicate.delivery/default/TencentARC--PhotoMaker/photomaker-v1.bin"
PHOTOMAKER_PATH = "models/photomaker-v1.bin"

def download_weights(url, dest, extract=True):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    args = ["pget"]
    if extract:
        args.append("-x")
    subprocess.check_call(args + [url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)


# utility function for style templates
def apply_style(style_name: str, positive: str, negative: str = "") -> tuple[str, str]:
    p, n = styles.get(style_name, styles[DEFAULT_STYLE_NAME])
    return p.replace("{prompt}", positive), n + " " + negative

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # download PhotoMaker checkpoint to cache
        # if we already have the model, this doesn't do anything
        if not os.path.exists(PHOTOMAKER_PATH):
            download_weights(PHOTOMAKER_URL, PHOTOMAKER_PATH, extract=False)

        if not os.path.exists(BASE_MODEL_PATH):
            download_weights(BASE_MODEL_URL, BASE_MODEL_PATH)

        print("Loading safety checker...")
        if not os.path.exists(SAFETY_CACHE):
            download_weights(SAFETY_URL, SAFETY_CACHE)
        self.safety_checker = StableDiffusionSafetyChecker.from_pretrained(
            SAFETY_CACHE, torch_dtype=torch.float16
        ).to("cuda")
        self.feature_extractor = CLIPImageProcessor.from_pretrained(FEATURE_EXTRACTOR)

        self.pipe = PhotoMakerStableDiffusionXLPipeline.from_pretrained(
            BASE_MODEL_PATH,
            torch_dtype=torch.bfloat16,
            use_safetensors=True,
            variant="fp16",
        ).to(self.device)

        self.pipe.load_photomaker_adapter(
            os.path.dirname(PHOTOMAKER_PATH),
            subfolder="",
            weight_name=os.path.basename(PHOTOMAKER_PATH),
            trigger_word="img",
        )
        self.pipe.id_encoder.to(self.device)

        self.pipe.scheduler = EulerDiscreteScheduler.from_config(
            self.pipe.scheduler.config
        )
        self.pipe.fuse_lora()

    @torch.inference_mode()
    def predict(
        self,
        input_image: Path = Input(
            description="The input image, for example a photo of your face."
        ),
        input_image2: Path = Input(
            description="Additional input image (optional)",
            default=None
        ),
        input_image3: Path = Input(
            description="Additional input image (optional)",
            default=None
        ),
        input_image4: Path = Input(
            description="Additional input image (optional)",
            default=None
        ),
        prompt: str = Input(
            description="Prompt. Example: 'a photo of a man/woman img'. The phrase 'img' is the trigger word.",
            default="A photo of a person img",
        ),
        style_name: str = Input(
            description="Style template. The style template will add a style-specific prompt and negative prompt to the user's prompt.",
            choices=STYLE_NAMES,
            default=DEFAULT_STYLE_NAME,
        ),
        negative_prompt: str = Input(
            description="Negative Prompt. The negative prompt should NOT contain the trigger word.",
            default="nsfw, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry",
        ),
        num_steps: int = Input(
            description="Number of sample steps", default=20, ge=1, le=100
        ),
        style_strength_ratio: float = Input(
            description="Style strength (%)", default=20, ge=15, le=50
        ),
        num_outputs: int = Input(
            description="Number of output images", default=1, ge=1, le=4
        ),
        guidance_scale: float = Input(
            description="Guidance scale. A guidance scale of 1 corresponds to doing no classifier free guidance.", default=5, ge=1, le=10.0
        ),
        seed: int = Input(description="Seed. Leave blank to use a random number", default=None, ge=0, le=MAX_SEED),
        disable_safety_checker: bool = Input(
            description="Disable safety checker for generated images.",
            default=False
        )
    ) -> list[Path]:
        """Run a single prediction on the model"""
        # remove old outputs
        output_folder = Path('outputs')
        if output_folder.exists():
            shutil.rmtree(output_folder)
        os.makedirs(str(output_folder), exist_ok=False)

        # randomize seed if necessary
        if seed is None:
            seed = random.randint(0, MAX_SEED)
        print(f"Using seed {seed}...")

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

        # check the negative prompt for the trigger word
        if negative_prompt:
            negative_prompt_ids = self.pipe.tokenizer.encode(negative_prompt)
            if image_token_id in negative_prompt_ids:
                raise ValueError(
                    f"Cannot use trigger word '{self.pipe.trigger_word}' in negative prompt!"
                )

        # apply the style template
        prompt, negative_prompt = apply_style(style_name, prompt, negative_prompt)

        # load the input images
        input_id_images = []
        for maybe_image in [input_image, input_image2, input_image3, input_image4]:
          if maybe_image:
            print(f"Loading image {maybe_image}...")
            input_id_images.append(load_image(str(maybe_image)))

        print(f"Setting seed...")
        generator = torch.Generator(device=self.device).manual_seed(seed)

        print("Start inference...")
        print(f"[Debug] Prompt: {prompt}")
        print(f"[Debug] Neg Prompt: {negative_prompt}")
        start_merge_step = int(float(style_strength_ratio) / 100 * num_steps)
        if start_merge_step > 30:
            start_merge_step = 30
        print(f"Start merge step: {start_merge_step}")
        images = self.pipe(
            prompt=prompt,
            input_id_images=input_id_images,
            negative_prompt=negative_prompt,
            num_images_per_prompt=num_outputs, 
            num_inference_steps=num_steps,
            start_merge_step=start_merge_step,
            generator=generator,
            guidance_scale=guidance_scale,
        ).images

        if not disable_safety_checker:
            print(f"Running safety checker...")
            _, has_nsfw_content = self.run_safety_checker(images)
        # save results to file
        print(f"Saving images to file...")
        output_paths = []
        for i, image in enumerate(images):
            if not disable_safety_checker:
                if has_nsfw_content[i]:
                    print(f"NSFW content detected in image {i}")
                    continue
            output_path = output_folder / f"image_{i}.png"
            image.save(output_path)
            output_paths.append(output_path)
        return [Path(p) for p in output_paths]

    def run_safety_checker(self, image):
        safety_checker_input = self.feature_extractor(image, return_tensors="pt").to(
            "cuda"
        )
        np_image = [np.array(val) for val in image]
        image, has_nsfw_concept = self.safety_checker(
            images=np_image,
            clip_input=safety_checker_input.pixel_values.to(torch.float16),
        )
        return image, has_nsfw_concept
