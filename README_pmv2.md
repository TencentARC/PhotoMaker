<p align="center">
  <img src="https://photo-maker.github.io/assets/logo.png" height=70>

</p>

<!-- ## <div align="center"><b>PhotoMaker</b></div> -->

<div align="center">
  
## PhotoMaker V2: Improved ID Fidelity and Better Controllability Compared to PhotoMaker V1
[[🤗 Demo](https://huggingface.co/spaces/TencentARC/PhotoMaker-V2)]

</div>

When training PhotoMaker V2, we focused on improving ID fidelity. Compared to PhotoMaker V1, we introduced 1️⃣ new training strategies, incorporated 2️⃣ more portrait datasets, and utilized 3️⃣ a more powerful ID extraction encoder. We will release a technical report soon. Thank you all for your attention.


### 🌠  **Key improvements in PhotoMaker V2:**

1. **ID fidelity** has been **further improved**, especially for single image input and Asian facial inputs. Of course, feeding more facial images can still yield better results.
2. By integrating [ControlNet](./inference_scripts/inference_pmv2_contronet.py), [T2I-Adapter](./inference_scripts/inference_pmv2_t2i_adapter.py), and [IP-Adapter](./inference_scripts/inference_pmv2_ip_adapter.py), the generation process becomes **more controllable**. We provide corresponding scripts for reference. Additionally, PhotoMaker V2 allows users to achieve better ID consistency by combining it with IP-Adapter-FaceID, InstantID, and [character LoRA](https://github.com/TencentARC/PhotoMaker/discussions/14).
3. PhotoMaker V2 **inherits the promising features of PhotoMaker V1**, such as high-quality and diverse generation capabilities, and powerful text control. Additionally, it can still integrate previous applications like bringing characters from old photos or paintings back to reality, identity mixing, and changing age or gender.

## Comparisons with PhotoMaker V1, IP-Adapter-FaceID and InstantID
We selected the three most prevalent methods in ID personalization generation, namely PhotoMaker V1, [IP-Adapter-FaceID-Plus-V2](https://huggingface.co/h94/IP-Adapter-FaceID) ([best of IP-Adapter-FaceID](https://github.com/cubiq/ComfyUI_IPAdapter_plus/issues/195)), and [InstantID](https://github.com/InstantID/InstantID).

To ensure a fair comparison, we used the same base model ([RealVisXL-V4.0](https://huggingface.co/SG161222/RealVisXL_V4.0)) and scheduler ([Euler](https://huggingface.co/docs/diffusers/api/schedulers/euler)), and selected the best out of four randomly generated images from each method for visualization. The prompts and negative prompts were consistent:

Prompt: `instagram photo, portrait photo of a woman img holding two cats, colorful, perfect face, natural skin, hard shadows, film grain`

Negative Prompt: `(asymmetry, worst quality, low quality, illustration, 3d, 2d, painting, cartoons, sketch), open mouth`

We can see that our method has **advantages** in maintaining ID fidelity and in the quality of the generated images

![comp_pm_v2_reba](https://github.com/user-attachments/assets/b978ffa2-97c9-4910-ab23-a2b2edd3be1d)

![comp_pm_v2_musk](https://github.com/user-attachments/assets/6b96d65b-813a-45e0-8f7a-25041dc4dc10)

![comp_pm_v2_yanzu](https://github.com/user-attachments/assets/b788b2b0-9166-4c9d-aa46-24ef1fb4e5a9)

![comp_pm_v2_yifei](https://github.com/user-attachments/assets/66fa8a73-8973-4e40-a094-c4cb3eec8d8a)


## Cooperation with ControlNet / T2I-Adapter / IP-Adapter

PhotoMaker V2 can collaborate with [T2I-Adapter’s doodle mode](https://huggingface.co/TencentARC/t2i-adapter-sketch-sdxl-1.0), 
allowing for controlled image generation based on user drawings and prompts.
This feature can be experienced in [[🤗 our official demo]](https://huggingface.co/spaces/TencentARC/PhotoMaker-V2).
The following video is an example of the experience process:

https://github.com/user-attachments/assets/1303d684-89e4-49d2-8e8c-4b659c8b48e7

Additionally, PhotoMaker V2 can work with [ControlNet](https://github.com/lllyasviel/ControlNet) and [T2I-Adapter](https://github.com/TencentARC/T2I-Adapter) for layout control, such as edge, pose, depth, and more.

We provide two example scripts:

1.	[inference_pmv2_contronet.py](./inference_scripts/inference_pmv2_contronet.py)
2.	[inference_pmv2_t2i_adapter.py](./inference_scripts/inference_pmv2_t2i_adapter.py)

The image below is an example of controlled generation using pose through ControlNet:

![pm_v2_controlnet](https://github.com/user-attachments/assets/57767447-192c-4606-af2a-4206b5dbccf9)

Our sample scripts can be referred to:
[inference_pmv2_ip_adapter.py](./inference_scripts/inference_pmv2_ip_adapter.py)

The image below is an example:

![pm_v2_ipadapter](https://github.com/user-attachments/assets/89f95604-6cfa-4dde-b563-2d052bac14cc)

PhotoMaker V2, as a plugin, can work well with other plugins, such as IP-Adapter-FaceID or InstantID, to further improve ID fidelity, or combining with LCM for acceleration. We look forward to your exploration of more features, and welcome you to **provide PRs** or **contribute to the open-source community**

🥳 If you have built or known repositories or applications around PhotoMaker V2, please leave us a message in the discussion. We will include them in our README.

## LICENSE
Since PhotoMaker V2 relies on [InsightFace](https://github.com/deepinsight/insightface), it also needs to comply with its [license](https://github.com/deepinsight/insightface?tab=readme-ov-file#license).












