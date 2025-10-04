<div align="center">

# LoRA.rar: Learning to Merge LoRAs via Hypernetworks for Subject-Style Conditioned Image Generation

[Donald Shenaj](https://donaldssh.github.io/)</a><sup>&diams; &spades; </sup>&nbsp;
[Ondrej Bohdal](https://ondrejbohdal.github.io/)<sup>&diams;</sup>&nbsp;
[Mete Ozay](https://openreview.net/profile?id=%7EMete_Ozay3)<sup>&diams;</sup>&nbsp;
[Pietro Zanuttigh](https://medialab.dei.unipd.it/members/pietro-zanuttigh/)<sup>&spades;</sup>&nbsp;
[Umberto Michieli](https://umbertomichieli.github.io/)<sup>&diams;</sup>&nbsp;


<sup>&diams;</sup> Samsung R&D Institute UK &nbsp; <sup>&spades;</sup> University of Padova &nbsp;

**ICCV 2025**

[![website](https://img.shields.io/badge/Project-Page-green)](https://donaldssh.github.io/LoRA.rar/)
[![arXiv](https://img.shields.io/badge/arXiv-2412.05148-red)](https://arxiv.org/abs/2412.05148)
[![huggingface](https://img.shields.io/badge/Hugging_Face-Page-yellow)](https://huggingface.co/papers/2412.05148)
[![BibTeX](https://img.shields.io/badge/Cite_us-BibTeX-blue)](#Citation)
  
</div>

<!-- Official repository of "LoRA.rar: Learning to Merge LoRAs via Hypernetworks for Subject-Style Conditioned Image Generation" by D. Shenaj, O. Bohdal, M. Ozay, P. Zanuttigh and U. Michieli, **ICCV 2025**. -->


## Abstract
Recent advancements in image generation models have enabled personalized image creation with both user-defined subjects (content) and styles. Prior works achieved personalization by merging corresponding low-rank adapters (LoRAs) through optimization-based methods, which are computationally demanding and unsuitable for real-time use on resource-constrained devices like smartphones. To address this, we introduce LoRA.rar, a method that not only improves image quality but also achieves a remarkable speedup of over $4000\times$ in the merging process. We collect a dataset of style and subject LoRAs and pre-train a hypernetwork on a diverse set of content-style LoRA pairs, learning an efficient merging strategy that generalizes to new, unseen content-style pairs, enabling fast, high-quality personalization. Moreover, we identify limitations in existing evaluation metrics for content-style quality and propose a new protocol using multimodal large language models (MLLMs) for more accurate assessment. Our method significantly outperforms the current state of the art in both content and style fidelity, as validated by MLLM assessments and human evaluations.


## ⚙️ Create the conda environment
``` 
conda env create -f lorarar.yaml
conda activate lorarar
```

## ⬇️ Download subject and style images
Image attributions are provided in the supplementary material. To download the images run:

``` 
bash scripts/download_datasets.sh
```

## 📚 Build the LoRA dataset

Train all subject and style LoRAs:
``` 
nohup bash scripts/sdxl/train_subject_loras.sh &
nohup bash scripts/sdxl/train_style_loras.sh &
```

## 💻 Train the hypernetwork

The final checkpoint for SDXL is provided in `models/hypernet.pth`.

If you want to retrain the hypernetwork, run:
``` 
nohup bash scripts/sdxl/train_lorarar.sh &
``` 

## 🚀 Inference
Run inference on all combinations of subject X style in the test set:

``` 
bash scripts/sdxl/run_inference.sh
```

## 🤖 MLLM evaluation

```
python mllm_eval.py --generated_imgs_dir $SAVED_IMAGES_PATH --reference_dir=datasets/test_datasets
```

<a name="Citation"></a>
## 🔗 Citation
```
@InProceedings{shenaj2025lora,
    author    = {Shenaj, Donald and Bohdal, Ondrej and Ozay, Mete and Zanuttigh, Pietro and Michieli, Umberto},
    title     = {LoRA.rar: Learning to Merge LoRAs via Hypernetworks for Subject-Style Conditioned Image Generation},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2025}
}
```

Acknowledgement: our code extends https://github.com/mkshing/ziplora-pytorch
