# OmniFake-UMFDet: Towards Unified Multimodal Misinformation Detection in Social Media

<p align="center">
  <a href="https://0112hy.github.io/OmniFakeWeb/"><strong>Project Page</strong></a> |
  <a href="https://arxiv.org/abs/2509.25991"><strong>arXiv</strong></a> |
  <a href="https://huggingface.co/datasets/hy1228/OmniFake98K"><strong>Dataset</strong></a> |
  <a href="#model-weights"><strong>Model Weights</strong></a>
</p>

<p align="center">
  <strong>Official implementation of UMFDet for the OmniFake benchmark.</strong><br>
  A unified framework for detecting human-crafted and AI-synthesized multimodal misinformation.
</p>

<p align="center">
  <img src="assets/umfdet_framework.png" width="92%" alt="UMFDet framework">
</p>

## News

- 2026-07: Code, model configuration files, training script, and evaluation script are released.
- 2026-07: OmniFake98K dataset link is available on Hugging Face.
- 2025-09: Paper released on arXiv.

## Introduction

Multimodal misinformation on social media includes both human-crafted misinformation and AI-synthesized deceptive content. Existing methods often focus on one category only, which limits their practical use when the manipulation type of a real-world post is unknown.

OmniFake-UMFDet provides an open implementation for a unified misinformation detection setting. The benchmark covers real posts, human-crafted rumors, vision manipulation, text manipulation, and mixed manipulation. UMFDet builds on a vision-language model backbone and introduces category-aware expert modeling to improve recognition across heterogeneous misinformation types.

This repository contains the training, evaluation, dataset loading code, and model definition files. Large pretrained weights are intentionally not committed to GitHub and should be downloaded separately.

## Links

| Resource | Link |
| --- | --- |
| Project page | https://0112hy.github.io/OmniFakeWeb/ |
| Paper | https://arxiv.org/abs/2509.25991 |
| arXiv | https://arxiv.org/abs/2509.25991 |
| Dataset | https://huggingface.co/datasets/hy1228/OmniFake98K |
| Code | https://github.com/0112HY/OmniFake-UMFDet |
| Model weights | Coming soon. Place the downloaded weights under `model/`. |

## Repository Structure

```text
OmniFake-UMFDet/
├── assets/
│   └── umfdet_framework.png
├── model/
│   ├── config.json
│   ├── configuration_florence2.py
│   ├── modeling_florence2.py
│   ├── processing_florence2.py
│   ├── preprocessor_config.json
│   ├── tokenizer_config.json
│   ├── tokenizer.json
│   └── vocab.json
├── dataset.py
├── train.py
├── evaluate.py
├── requirements.txt
└── README.md
```

## Installation

Create a clean Python environment:

```bash
conda create -n umfdet python=3.10 -y
conda activate umfdet
```

Install PyTorch according to your CUDA version. For example:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Install the remaining dependencies:

```bash
pip install -r requirements.txt
```

## Dataset Preparation

Download the OmniFake98K dataset from Hugging Face:

```bash
pip install -U huggingface_hub
huggingface-cli download hy1228/OmniFake98K --repo-type dataset --local-dir data/OmniFake98K
```

The dataset loader expects a CSV file with the following columns:

| Column | Description |
| --- | --- |
| `news_title` | Text/title paired with the image. |
| `img_path` | Path to the image file. Absolute paths are recommended. |
| `label` | Target answer used for training and generation. |
| `label_5` | Five-class label for analysis or filtering. |

The five-class label space is:

```text
real
rumor
vision_manipulation
text_manipulation
mixed_manipulation
```

Before training or evaluation, set the CSV paths and image root paths in `train.py` and `evaluate.py` according to your local dataset location.

## Model Weights

The repository includes the model architecture and processor-related files under `model/`, but does not include large checkpoint files.

After downloading the pretrained weights, place them under:

```text
model/model.safetensors
```

or use the checkpoint directory path directly as the model path in the scripts.

Do not commit checkpoint files to GitHub. The `.gitignore` file excludes common weight formats such as `.safetensors`, `.bin`, `.pt`, `.pth`, and `.ckpt`.

## Training

Edit the following placeholders in `train.py` before running:

```python
train_csv_path_str = ""
model_root_path = ""
```

Also set the validation CSV path and image directory path in the `FakeDataset` construction block.

Run distributed training with all visible GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py \
  --dataset fakenews \
  --batch-size 24 \
  --epochs 10 \
  --lr 4e-5 \
  --run-name umfdet_omnifake
```

For LoRA training, add:

```bash
--use-lora
```

## Evaluation

Edit the following placeholders in `evaluate.py` before running:

```python
Model_Path = ""
csv_file_path = ""
image_directory_path = ""
```

Then run:

```bash
python evaluate.py
```

The script reports both five-class and binary real/fake metrics, including accuracy, per-class precision, recall, F1-score, and a confusion matrix.

## Prompt Format

UMFDet uses the following task prompt prefix:

```text
<DETECTION_NEWS>
```

Each input is formatted as a multimodal fake news classification task with one image and one news title. The model is required to output exactly one of the five categories listed above.

## Citation

If you find this repository useful for your research, please cite our paper:

```bibtex
@misc{li2025unifiedmultimodalmisinformationdetection,
      title={Towards Unified Multimodal Misinformation Detection in Social Media: A Benchmark Dataset and Baseline},
      author={Haiyang Li and Yaxiong Wang and Shengeng Tang and Lianwei Wu and Lechao Cheng and Zhun Zhong},
      year={2025},
      eprint={2509.25991},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2509.25991},
}
```

## Acknowledgements

This implementation builds on the Florence-2 vision-language modeling interface and the Hugging Face Transformers ecosystem. We thank the open-source community for providing the tools that make reproducible multimodal research possible.
