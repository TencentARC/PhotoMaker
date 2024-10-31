<p align="center">
  <img src="https://photo-maker.github.io/assets/logo.png" height=100>

</p>

<!-- ## <div align="center"><b>PhotoMaker</b></div> -->

<div align="center">
  
## PhotoMaker: Customizing Realistic Human Photos via Stacked ID Embedding  [![Paper page](https://huggingface.co/datasets/huggingface/badges/resolve/main/paper-page-md-dark.svg)](https://huggingface.co/papers/2312.04461)
[[Paper](https://huggingface.co/papers/2312.04461)] &emsp; [[Project Page](https://photo-maker.github.io)] &emsp; [[Model Card](https://huggingface.co/TencentARC/PhotoMaker)] <br>

[[💥New 🤗 **Demo (PhotoMaker V2)**](https://huggingface.co/spaces/TencentARC/PhotoMaker-V2)] &emsp; [[🤗 Demo (Realistic)](https://huggingface.co/spaces/TencentARC/PhotoMaker)] &emsp; [[🤗 Demo (Stylization)](https://huggingface.co/spaces/TencentARC/PhotoMaker-Style)] <br>

[[Replicate Demo (Realistic)](https://replicate.com/jd7h/photomaker)] &emsp; [[Replicate Demo (Stylization)](https://replicate.com/yorickvp/photomaker-style)] &emsp; [[Jittor version](https://github.com/NK-JittorCV/nk-diffusion)]  <be>

<img src="https://github.com/user-attachments/assets/d7d99f0b-2f1e-4d13-8ca8-e08685bcda2a" height=40>

**PhotoMaker-V2** is supported by the [HunyuanDiT](https://github.com/Tencent/HunyuanDiT) team. 

🥳 We release **PhotoMaker V2**. Please refer to [comparisons](./README_pmv2.md) between PhotoMaker V1, PhotoMaker V2, IP-Adapter-FaceID-plus-V2, and InstantID. Please watch [this video](https://photo-maker.github.io/assets/demo_pm_v2_full.mp4) for how to use our demo. For PhotoMaker V2 ComfyUI nodes, please refer to the [Related Resources](https://github.com/TencentARC/PhotoMaker?tab=readme-ov-file#related-resources) 
</div>


---

### 🌠  **Key Features:**

1. Rapid customization **within seconds**, with no additional LoRA training.
2. Ensures impressive ID fidelity, offering diversity, promising text controllability, and high-quality generation.
3. Can serve as an **Adapter** to collaborate with other Base Models alongside LoRA modules in community.

---
<a href="https://trendshift.io/repositories/7008" target="_blank" align=center><img src="https://trendshift.io/api/badge/repositories/7008" alt="TencentARC%2FPhotoMaker | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>

❗❗ Note: If there are any PhotoMaker based resources and applications, please leave them in the [discussion](https://github.com/TencentARC/PhotoMaker/discussions/36) and we will list them in the [Related Resources](https://github.com/TencentARC/PhotoMaker?tab=readme-ov-file#related-resources) section in README file.
Now we know the implementation of **Replicate**, **Windows**, **ComfyUI**, and **WebUI**. Thank you all! 

<div align="center">

![photomaker_demo_fast](https://github.com/TencentARC/PhotoMaker/assets/21050959/e72cbf4d-938f-417d-b308-55e76a4bc5c8)
</div>


## 🚩 **New Features/Updates**
- ✅ July 22, 2024. 💥 We release PhotoMaker V2 with **improved ID fidelity**. At the same time, it still maintains the generation quality, editability, and compatibility with any plugins that PhotoMaker V1 offers. We have also provided scripts for integration with [ControlNet](./inference_scripts/inference_pmv2_contronet.py), [T2I-Adapter](./inference_scripts/inference_pmv2_t2i_adapter.py), and [IP-Adapter](./inference_scripts/inference_pmv2_ip_adapter.py) to offer excellent control capabilities. Users can further customize scripts for upgrades, such as combining with LCM for acceleration or integrating with IP-Adapter-FaceID or InstantID to further improve ID fidelity. We will release technical report of PhotoMaker V2 soon. Please refer to [this doc](./README_pmv2.md) for a quick preview. 
- ✅ January 20, 2024. An **important** note: For those GPUs that do not support bfloat16, please change [this line](https://github.com/TencentARC/PhotoMaker/blob/6ec44fc13909d64a65c635b9e3b6f238eb1de9fe/gradio_demo/app.py#L39) to `torch_dtype = torch.float16`, the speed will be **greatly improved** (1min/img (before) vs. 14s/img (after) on V100). The minimum GPU memory requirement for PhotoMaker is **11G** (Please refer to [this link](https://github.com/TencentARC/PhotoMaker/discussions/114) for saving GPU memory).
- ✅ January 15, 2024. We release PhotoMaker.

---

## 🔥 **Examples**


### Realistic generation 

- [![Huggingface PhotoMaker](https://img.shields.io/static/v1?label=Demo&message=Huggingface%20Gradio&color=orange)](https://huggingface.co/spaces/TencentARC/PhotoMaker)
- [**PhotoMaker notebook demo**](photomaker_demo.ipynb)

<p align="center">
  <img src="https://cdn-uploads.huggingface.co/production/uploads/6285a9133ab6642179158944/BYBZNyfmN4jBKBxxt4uxz.jpeg" height=450>
</p>

<p align="center">
  <img src="https://cdn-uploads.huggingface.co/production/uploads/6285a9133ab6642179158944/9KYqoDxfbNVLzVKZzSzwo.jpeg" height=450>
</p>

### Stylization generation 

Note: only change the base model and add the LoRA modules for better stylization

- [![Huggingface PhotoMaker-Style](https://img.shields.io/static/v1?label=Demo&message=Huggingface%20Gradio&color=orange)](https://huggingface.co/spaces/TencentARC/PhotoMaker-Style)
- [**PhotoMaker-Style notebook demo**](photomaker_style_demo.ipynb) 

<p align="center">
  <img src="https://cdn-uploads.huggingface.co/production/uploads/6285a9133ab6642179158944/du884lcjpqqjnJIxpATM2.jpeg" height=450>
</p>
  
<p align="center">
  <img src="https://cdn-uploads.huggingface.co/production/uploads/6285a9133ab6642179158944/-AC7Hr5YL4yW1zXGe_Izl.jpeg" height=450>
</p>

# 🔧 Dependencies and Installation

- Python >= 3.8 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html))
- [PyTorch >= 2.0.0](https://pytorch.org/)
```bash
conda create --name photomaker python=3.10
conda activate photomaker
pip install -U pip

# Install requirements
pip install -r requirements.txt

# Install photomaker
pip install git+https://github.com/TencentARC/PhotoMaker.git
```

Then you can run the following command to use it
```python
from photomaker import PhotoMakerStableDiffusionXLPipeline
```

# ⏬ Download Models 
The model will be automatically downloaded through the following two lines:

```python
from huggingface_hub import hf_hub_download
photomaker_path = hf_hub_download(repo_id="TencentARC/PhotoMaker", filename="photomaker-v1.bin", repo_type="model")
```

You can also choose to download manually from this [url](https://huggingface.co/TencentARC/PhotoMaker).

# 💻 How to Test

## Use like [diffusers](https://github.com/huggingface/diffusers)

- Dependency
```py
import torch
import os
from diffusers.utils import load_image
from diffusers import EulerDiscreteScheduler
from photomaker import PhotoMakerStableDiffusionXLPipeline

### Load base model
pipe = PhotoMakerStableDiffusionXLPipeline.from_pretrained(
    base_model_path,  # can change to any base model based on SDXL
    torch_dtype=torch.bfloat16, 
    use_safetensors=True, 
    variant="fp16"
).to(device)

### Load PhotoMaker checkpoint
pipe.load_photomaker_adapter(
    os.path.dirname(photomaker_path),
    subfolder="",
    weight_name=os.path.basename(photomaker_path),
    trigger_word="img"  # define the trigger word
)     

pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)

### Also can cooperate with other LoRA modules
# pipe.load_lora_weights(os.path.dirname(lora_path), weight_name=lora_model_name, adapter_name="xl_more_art-full")
# pipe.set_adapters(["photomaker", "xl_more_art-full"], adapter_weights=[1.0, 0.5])

pipe.fuse_lora()
```

- Input ID Images
```py
### define the input ID images
input_folder_name = './examples/newton_man'
image_basename_list = os.listdir(input_folder_name)
image_path_list = sorted([os.path.join(input_folder_name, basename) for basename in image_basename_list])

input_id_images = []
for image_path in image_path_list:
    input_id_images.append(load_image(image_path))
```

<div align="center">

<a href="https://github.com/TencentARC/PhotoMaker/assets/21050959/01d53dfa-7528-4f09-a1a5-96b349ae7800" align="center"><img style="margin:0;padding:0;" src="https://github.com/TencentARC/PhotoMaker/assets/21050959/01d53dfa-7528-4f09-a1a5-96b349ae7800"/></a>
</div>

- Generation
```py
# Note that the trigger word `img` must follow the class word for personalization
prompt = "a half-body portrait of a man img wearing the sunglasses in Iron man suit, best quality"
negative_prompt = "(asymmetry, worst quality, low quality, illustration, 3d, 2d, painting, cartoons, sketch), open mouth, grayscale"
generator = torch.Generator(device=device).manual_seed(42)
images = pipe(
    prompt=prompt,
    input_id_images=input_id_images,
    negative_prompt=negative_prompt,
    num_images_per_prompt=1,
    num_inference_steps=num_steps,
    start_merge_step=10,
    generator=generator,
).images[0]
gen_images.save('out_photomaker.png')
```

<div align="center">

<a href="https://github.com/TencentARC/PhotoMaker/assets/21050959/703c00e1-5e50-4c19-899e-25ee682d2c06" align="center"><img width=400 style="margin:0;padding:0;" src="https://github.com/TencentARC/PhotoMaker/assets/21050959/703c00e1-5e50-4c19-899e-25ee682d2c06"/></a>

</div>

## Start a local gradio demo
Run the following command:

```python
python gradio_demo/app.py
```

You could customize this script in [this file](gradio_demo/app.py).

If you want to run it on MAC, you should follow [this Instruction](MacGPUEnv.md) and then run the app.py.

## Usage Tips:
- Upload more photos of the person to be customized to improve ID fidelity. If the input is Asian face(s), maybe consider adding 'Asian' before the class word, e.g., `Asian woman img`
- When stylizing, does the generated face look too realistic? Adjust the Style strength to 30-50, the larger the number, the less ID fidelity, but the stylization ability will be better. You could also try out other base models or LoRAs with good stylization effects.
- Reduce the number of generated images and sampling steps for faster speed. However, please keep in mind that reducing the sampling steps may compromise the ID fidelity.

# Related Resources
### Replicate demo of PhotoMaker: 
1. [Demo link](https://replicate.com/jd7h/photomaker), run PhotoMaker on replicate, provided by [@yorickvP](https://github.com/yorickvP) and [@jd7h](https://github.com/jd7h).
2. [Demo link (style version)](https://replicate.com/yorickvp/photomaker-style).

### WebUI version of PhotoMaker: 
1. **stable-diffusion-webui-forge**: https://github.com/lllyasviel/stable-diffusion-webui-forge provided by [@Lvmin Zhang](https://github.com/lllyasviel)
2. **Fooocus App**: [Fooocus-inswapper](https://github.com/machineminded/Fooocus-inswapper) provided by [@machineminded](https://github.com/machineminded)

### Windows version of PhotoMaker: 
1. [bmaltais/PhotoMaker](https://github.com/bmaltais/PhotoMaker/tree/v1.0.1) by [@bmaltais](https://github.com/bmaltais), easy to deploy PhotoMaker on Windows. The description can be found in [this link](https://github.com/TencentARC/PhotoMaker/discussions/36#discussioncomment-8156199).
2. [sdbds/PhotoMaker-for-windows](https://github.com/sdbds/PhotoMaker-for-windows/tree/windows) by [@sdbds](https://github.com/sdbds).
   
### ComfyUI:
1. 🔥 **Official Implementation by [ComfyUI](https://github.com/comfyanonymous/ComfyUI)**: https://github.com/comfyanonymous/ComfyUI/commit/d1533d9c0f1dde192f738ef1b745b15f49f41e02
2. https://github.com/ZHO-ZHO-ZHO/ComfyUI-PhotoMaker
3. https://github.com/StartHua/Comfyui-Mine-PhotoMaker
4. https://github.com/shiimizu/ComfyUI-PhotoMaker

### ComfyUI (for PhotoMaker V2):
1. https://github.com/shiimizu/ComfyUI-PhotoMaker-Plus
2. https://github.com/edwios/ComfyUI-PhotoMakerV2-ZHO/tree/main
3. https://openart.ai/workflows/shalacai/photomakerv2/fttT4ztRM85JxBJ2eUyr
4. https://github.com/zhangp365/ComfyUI_photomakerV2_native

### Purely C/C++/CUDA version of PhotoMaker:
1. [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp/pull/179) by [@bssrdf](https://github.com/bssrdf).

### Other Applications / Web Demos
1. **Wisemodel 始智 (Easy to use in China)** https://wisemodel.cn/space/gradio/photomaker 
2. **OpenXLab (Easy to use in China)**: https://openxlab.org.cn/apps/detail/camenduru/PhotoMaker
 [![Open in OpenXLab](https://cdn-static.openxlab.org.cn/app-center/openxlab_app.svg)](https://openxlab.org.cn/apps/detail/camenduru/PhotoMaker)
by [@camenduru](https://github.com/camenduru).
3. **Colab**: https://github.com/camenduru/PhotoMaker-colab by [@camenduru](https://github.com/camenduru)
4. **Monster API**: https://monsterapi.ai/playground?model=photo-maker
5. **Pinokio**: https://pinokio.computer/item?uri=https://github.com/cocktailpeanutlabs/photomaker

### Graido demo in 45 lines
Provided by [@Gradio](https://twitter.com/Gradio/status/1747683500495691942)


# 🤗 Acknowledgements
- PhotoMaker is co-hosted by Tencent ARC Lab and Nankai University [MCG-NKU](https://mmcheng.net/cmm/).
- Inspired from many excellent demos and repos, including [IP-Adapter](https://github.com/tencent-ailab/IP-Adapter), [multimodalart/Ip-Adapter-FaceID](https://huggingface.co/spaces/multimodalart/Ip-Adapter-FaceID), [FastComposer](https://github.com/mit-han-lab/fastcomposer), and [T2I-Adapter](https://github.com/TencentARC/T2I-Adapter). Thanks for their great work!
- Thanks to the [HunyuanDiT](https://github.com/Tencent/HunyuanDiT) team for their generous support and suggestions!
- Thanks to the Venus team in Tencent PCG for their feedback and suggestions.
- Thanks to the HuggingFace team for their generous support! 

# Disclaimer
This project strives to impact the domain of AI-driven image generation positively. Users are granted the freedom to create images using this tool, but they are expected to comply with local laws and utilize it responsibly. The developers do not assume any responsibility for potential misuse by users.

# BibTeX
If you find PhotoMaker useful for your research and applications, please cite using this BibTeX:

```BibTeX
@inproceedings{li2023photomaker,
  title={PhotoMaker: Customizing Realistic Human Photos via Stacked ID Embedding},
  author={Li, Zhen and Cao, Mingdeng and Wang, Xintao and Qi, Zhongang and Cheng, Ming-Ming and Shan, Ying},
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2024}
}
