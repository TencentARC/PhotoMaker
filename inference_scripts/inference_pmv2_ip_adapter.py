# !pip install opencv-python transformers accelerate
import os
import sys

import numpy as np
import torch
from diffusers.utils import load_image
from diffusers import EulerDiscreteScheduler
from huggingface_hub import hf_hub_download
from photomaker import PhotoMakerStableDiffusionXLPipeline
from photomaker import FaceAnalysis2, analyze_faces

face_detector = FaceAnalysis2(providers=['CUDAExecutionProvider'], allowed_modules=['detection', 'recognition'])
face_detector.prepare(ctx_id=0, det_size=(640, 640))

try:
    if torch.cuda.is_available():
        device = "cuda"
    elif sys.platform == "darwin" and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
except:
    device = "cpu"

torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
if device == "mps":
    torch_dtype = torch.float16

output_dir = "./outputs"
os.makedirs(output_dir, exist_ok=True)

photomaker_ckpt = hf_hub_download(repo_id="TencentARC/PhotoMaker-V2", filename="photomaker-v2.bin", repo_type="model")

prompt = "portrait photo of a woman img, colorful, perfect face, best quality"
negative_prompt = "(asymmetry, worst quality, low quality, illustration), open mouth"

# # initialize the models and pipeline
### Load base model
pipe = PhotoMakerStableDiffusionXLPipeline.from_pretrained(
    "SG161222/RealVisXL_V4.0", torch_dtype=torch_dtype,
).to("cuda")

pipe.load_ip_adapter("h94/IP-Adapter", subfolder="sdxl_models", weight_name="ip-adapter_sdxl.bin")
pipe.set_ip_adapter_scale(0.7)

print("Loading images...")
style_images = [load_image(f"./examples/statue.png")]

### Load PhotoMaker checkpoint
pipe.load_photomaker_adapter(
    os.path.dirname(photomaker_ckpt),
    subfolder="",
    weight_name=os.path.basename(photomaker_ckpt),
    trigger_word="img"  # define the trigger word
)     
### Also can cooperate with other LoRA modules
# pipe.load_lora_weights(os.path.dirname(lora_path), weight_name=lora_model_name, adapter_name="lcm-lora")
# pipe.set_adapters(["photomaker", "lcm-lora"], adapter_weights=[1.0, 0.5])

pipe.fuse_lora()
pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)

pipe.enable_model_cpu_offload()

### define the input ID images
input_folder_name = './examples/scarletthead_woman'
image_basename_list = os.listdir(input_folder_name)
image_path_list = sorted([os.path.join(input_folder_name, basename) for basename in image_basename_list])

input_id_images = []
for image_path in image_path_list:
    input_id_images.append(load_image(image_path))

id_embed_list = []

for img in input_id_images:
    img = np.array(img)
    img = img[:, :, ::-1]
    faces = analyze_faces(face_detector, img)
    if len(faces) > 0:
        id_embed_list.append(torch.from_numpy((faces[0]['embedding'])))

if len(id_embed_list) == 0:
    raise ValueError(f"No face detected in input image pool")

id_embeds = torch.stack(id_embed_list)

# generate image
images = pipe(
    prompt, 
    negative_prompt=negative_prompt, 
    input_id_images=input_id_images,
    id_embeds=id_embeds,
    ip_adapter_image=[style_images],
    num_images_per_prompt=2,
    start_merge_step=10,
).images

for idx, img in enumerate(images): 
    img.save(os.path.join(output_dir, f"output_pmv2_ipa_{idx}.jpg"))
