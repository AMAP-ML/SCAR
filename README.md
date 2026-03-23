# EditAR: Unified Conditional Generation with Autoregressive Models (CVPR 2025)


<div align="center">

[![project page](https://img.shields.io/badge/Project_page-More_visualizations-green)](https://jitengmu.github.io/EditAR/)&nbsp;
[![arXiv](https://img.shields.io/badge/arXiv%20paper-2406.06525-b31b1b.svg)](https://arxiv.org/abs/2501.04699)&nbsp;

</div>


<p align="center">
<img src="assets/teaser.png" width=95%>
<p>

> [**EditAR: Unified Conditional Generation with Autoregressive Models**](https://arxiv.org/abs/2501.04699)<br>
> [JitengMu](https://jitengmu.github.io/), [Nuno Vasconcelos](http://www.svcl.ucsd.edu/~nuno/), [Xiaolong Wang](https://xiaolonw.github.io/)
> <br>University of California, San Diego<br>


## 🌿 Introduction
Diffusion models have made significant advances in text-guided synthesis tasks. Recent progress in controllable image generation and editing is largely driven by diffusion-based methods. Although diffusion models perform exceptionally well in specific tasks with tailored designs, establishing a unified model is still challenging. In contrast, autoregressive models inherently feature a unified tokenized representation, which simplifies the creation of a single foundational model for various tasks. In this work, we propose EditAR, a single unified autoregressive framework for a variety of conditional image generation tasks, e.g., image editing, depth-to-image, edge-to-image, segmentation-to-image. The model takes both images and instructions as inputs, and predicts the edited images tokens in a vanilla next-token paradigm. To enhance the text-to-image alignment, we further propose to distill the knowledge from foundation models into the autoregressive modeling process. We evaluate its effectiveness across diverse tasks on established benchmarks, showing competitive performance to various state-of-the-art task-specific methods.

The codebase is implemented using [PyTorch 2.2.1](https://pytorch.org/) with python 3.10 and tested on [Ubuntu](https://ubuntu.com/) 20.04.6 LTS.

### 🔧 Preparation
* Environment Setup. Please follow `install.sh` to install the packages as shown in `requirements.txt`. Then you may download all pre-trained checkpoints as instructed below.

* Download text encoder model [flan-t5-xl](https://huggingface.co/google/flan-t5-xl) and put it as `./pretrained_models/t5-ckpt/flan-t5-xl`. Download vqvae model [vq_ds16_t2i.pt](https://huggingface.co/peizesun/llamagen_t2i/resolve/main/vq_ds16_t2i.pt) from [LlamaGen](https://github.com/FoundationVision/LlamaGen) and put it as `./pretrained_models/vq_ds16_t2i.pt`.

* (Required for training) Download pre-trained text-to-image model [t2i_XL_stage2_512.pt](https://huggingface.co/peizesun/llamagen_t2i/resolve/main/t2i_XL_stage2_512.pt) from [LlamaGen](https://github.com/FoundationVision/LlamaGen) and put it as `./pretrained_models/t2i_XL_stage2_512.pt`.

* (Required for inference) Download our trained model [editar_release.pt](https://huggingface.co/datasets/JitengMu/CVPR2025_EditAR_release) and put it as `./checkpoints/editar/editar_release.pt`.

### 🚀 Demo
Please run the following script to edit single image. Put the source image and instruction text in the `./examples` as demonstrated, then run,
```
python3 autoregressive/sample/sample_edit_example.py --gpt-ckpt ./checkpoints/editar/editar_release.pt --cfg-scale 3 --seed 83
```

### 🚀 Training
Data Preparation. For image editing, download [SEED-Data-Edit-Unsplash](https://huggingface.co/datasets/AILab-CVC/SEED-Data-Edit-Part1-Unsplash) and [PIPE Dataset](https://rotsteinnoam.github.io/Paint-by-Inpaint/). For image translation, we follow [ControlNet++](https://github.com/liming-ai/ControlNet_Plus_Plus) to download [depth,canny](https://huggingface.co/datasets/limingcv/MultiGen-20M_depth), and [Segmentation COCOStuff train set](https://huggingface.co/datasets/limingcv/Captioned_COCOStuff). Then each parquet dataset is then processed using `process_data_HF.py` by specifying the source path and target path.

The folder ends up looking like, 
```bash
./data/
  Seedx_Unsplash_HF/
  PIPE_HF/
  MultiGen-20M_depth_HF/
  Captioned_COCOStuff_HF/
```

We provide an example as shown in `train.sh`. Please modify `train.sh` accordingly to run on your system.

### 🚀 Evaluation
Data Preparation. For image editing, please refer to [Direct Inversion](https://github.com/cure-lab/PnPInversion) to download the PIE-Bench dataset. For image translation, we follow [ControlNet++](https://github.com/liming-ai/ControlNet_Plus_Plus) to download [depth,canny](https://huggingface.co/datasets/limingcv/MultiGen-20M_depth_eval), and [Segmentation COCOStuff validation set](https://huggingface.co/datasets/limingcv/Captioned_COCOStuff). Then each parquet dataset is then processed using `process_data_HF.py` by specifying the source path and target path.

The folder ends up looking like, 
```bash
./data/
  PIE_Bench_Dataset/
  MultiGen-20M_depth_eval_HF/
  Captioned_COCOStuff_eval_HF/
```

Please replace `$TESTSET` with one of `PIE-bench/depth/canny/conditionsegmentation` for evaluation on different benchmark datasets.
```
python3 autoregressive/sample/sample_edit_folder.py --gpt-ckpt ./checkpoints/editar/editar_release.pt --cfg-scale 3 --testset $TESTSET
```

## Acknowledgement

The implementation is mainly built on top of [LlamaGen](https://github.com/FoundationVision/LlamaGen). We also want to thank the authors from [ControlNetPlusPlus](https://github.com/liming-ai/ControlNet_Plus_Plus), [ControlAR](https://github.com/hustvl/ControlAR), [SmartEdit](https://github.com/TencentARC/SmartEdit), [Dino-v2](https://github.com/facebookresearch/dinov2) for the code release.

## License
The majority of this project is licensed under MIT License. Portions of the project are under separate license of referred projects.

## BibTeX
```bibtex
@article{mu2025editAR,
  title={EditAR: Unified Conditional Generation with Autoregressive Models},
  author={Mu, Jiteng and Vasconcelos, Nuno and Wang, Xiaolong},
  journal={arXiv preprint arXiv:2501.04699},
  year={2025}
}
```
