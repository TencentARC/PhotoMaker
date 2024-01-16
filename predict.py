# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Input, Path
import torch
import numpy as np
import random
import os
from PIL import Image
import logging
import time
from typing import List
import shutil

from diffusers.utils import load_image
from diffusers import EulerDiscreteScheduler

from photomaker.pipeline import PhotoMakerStableDiffusionXLPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger(__name__)

base_model_path = 'SG161222/RealVisXL_V3.0'
photomaker_path = 'release_model/photomaker-v1.bin'
device = "cuda"

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        start = time.time()
        logger.info("Loading model...")

        self.pipe = PhotoMakerStableDiffusionXLPipeline.from_pretrained(
            base_model_path, 
            torch_dtype=torch.bfloat16, 
            use_safetensors=True, 
            variant="fp16"
        ).to(device)
        
        self.pipe.load_photomaker_adapter(
           os.path.dirname(photomaker_path),
           subfolder="",
           weight_name=os.path.basename(photomaker_path),
           trigger_word="img"
        )     
        
        self.pipe.scheduler = EulerDiscreteScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.fuse_lora()
        logger.info(f"Loaded model in {time.time() - start:.06}s")
        
    def _load_image(self, path):
        shutil.copyfile(path, "/tmp/image.png")
        return load_image("/tmp/image.png").convert("RGB")

    @torch.inference_mode()
    def predict(
        self,
        prompt: str = Input(
            description="Input prompt",
            default="sci-fi, closeup portrait photo of a man img wearing the sunglasses in Iron man suit, face, slim body, high quality, film grain"
        ),
        negative_prompt: str = Input(
            description="Negative Input prompt",
            default="(asymmetry, worst quality, low quality, illustration, 3d, 2d, painting, cartoons, sketch), open mouth"
        ),
        image: Path = Input(
            description="Input image for img2img or inpaint mode",
            default=None,
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=None
        ),
        num_outputs: int = Input(
            description="Number of images to output.",
            ge=1,
            le=4,
            default=1,
        ),
        num_inference_steps: int = Input(
            description="Number of denoising steps", ge=1, le=500, default=40
        )
    ) -> List[Path]:
        """Run a single prediction on the model"""
        if seed is None:
            seed = int.from_bytes(os.urandom(4), "big")
        logger.info(f"Using seed: {seed}")
        generator = torch.Generator("cuda").manual_seed(seed)

        style_strength_ratio = 20
        start_merge_step = int(float(style_strength_ratio) / 100 * num_inference_steps)
        if start_merge_step > 30:
            start_merge_step = 30
        
        images = self.pipe(
            prompt=prompt,
            input_id_images=[self._load_image(image)],
            negative_prompt=negative_prompt,
            num_images_per_prompt=num_outputs,
            num_inference_steps=num_inference_steps,
            start_merge_step=start_merge_step,
            generator=generator,
        ).images
    
        output_paths = []
        for i, image in enumerate(images):
            output_path = f"/tmp/out-{i}.png"
            image.save(output_path)
            output_paths.append(Path(output_path))

        return output_paths