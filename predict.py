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

from diffusers.utils import load_image
from diffusers import EulerDiscreteScheduler
from huggingface_hub import hf_hub_download

from photomaker.pipeline import PhotoMakerStableDiffusionXLPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger(__name__)

def image_grid(imgs, rows, cols, size_after_resize):
    assert len(imgs) == rows*cols

    w, h = size_after_resize, size_after_resize
    
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size
    
    for i, img in enumerate(imgs):
        img = img.resize((w,h))
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid

base_model_path = 'SG161222/RealVisXL_V3.0'
photomaker_path = 'release_model/photomaker-v1.bin'
device = "cuda"
save_path = "./outputs"

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        time = time.time()
        logger.info("Loading model...")
        self.pipe = PhotoMakerStableDiffusionXLPipeline.from_pretrained(
            base_model_path, 
            torch_dtype=torch.bfloat16, 
            use_safetensors=True, 
            variant="fp16",
        ).to(device)
        
        self.pipe.load_photomaker_adapter(
           os.path.dirname(photomaker_path),
           subfolder="",
           weight_name=os.path.basename(photomaker_path),
           trigger_word="img"
        )     
        
        self.pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
        self.pipe.fuse_lora()
        logger.info(f"Loaded model in {time.time() - time:.06}s")

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
            default=4,
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
        start_merge_step = int(float(style_strength_ratio) / 100 * num_steps)
        if start_merge_step > 30:
            start_merge_step = 30
        
        images = pipe(
            prompt=prompt,
            input_id_images=[load_image(image)],
            negative_prompt=negative_prompt,
            num_images_per_prompt=num_outputs,
            num_inference_steps=num_inference_steps,
            start_merge_step=start_merge_step,
            generator=generator,
        ).images
        
        grid = image_grid(images, 1, 4, size_after_resize=512)
        
        
        os.makedirs(save_path, exist_ok=True)
        
        for idx, image in enumerate(images):
            image.save(os.path.join(save_path, f"photomaker_{idx:02d}.png"))
 
        return grid