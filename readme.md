# MULSUM: A Multimodal Summarization System with Vis-Aligner and Diversity-Aware Image Selection

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

MULSUM is a multimodal summarization system that generates text summaries enhanced with relevant images. It leverages a vision-language model architecture based on LLaVA with a custom Vis-Aligner module for improved image-text alignment.

## Features

- **Multimodal Summarization**: Generate summaries that incorporate both text and images
- **Vis-Aligner**: Custom vision-language alignment module for improved image understanding
- **Diversity-Aware Image Selection**: Select the most relevant and diverse images for summaries
- **LoRA Fine-tuning**: Efficient fine-tuning using Low-Rank Adaptation
- **Configurable Pipeline**: JSON-based configuration for easy experimentation

## Installation

### Requirements

- Python 3.10 or higher
- CUDA-compatible GPU (recommended: 24GB+ VRAM)
- PyTorch 2.0+

### Setup

1. Clone the repository:
```bash
git clone https://github.com/hanghuacs/MULSUM.git
cd MULSUM
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the package:
```bash
pip install -e .
```

Or install dependencies directly:
```bash
pip install -r requirements.txt
```

## Project Structure

```
MULSUM/
├── src/                          # Main source code
│   ├── model/                    # Model architecture
│   │   ├── mulsum_arch.py       # Vis-Aligner and meta model
│   │   ├── mulsum_llama.py      # LLaMA-based multimodal model
│   │   └── builder.py           # Model loading utilities
│   ├── train/                    # Training code
│   │   ├── train_image_text.py  # Main training script
│   │   ├── dataset.py           # Dataset and data loading
│   │   └── mulsum_trainer.py    # Custom trainer
│   ├── image_text_inference.py  # Inference script
│   ├── image_text_inference_ablation.py  # Ablation studies
│   ├── image_selection.py       # Image selection module
│   └── ...
├── configs/                      # Configuration files
│   ├── image_text_config.json   # Main configuration
│   └── prompts_config.json      # Prompt templates
├── checkpoints/                  # Model checkpoints
├── data/                         # Data directory
└── results/                      # Output results
```

## Data Preparation

### Dataset Format

Your training data should be in CSV format with the following columns:

| Column | Description |
|--------|-------------|
| `Article_ID` | Unique identifier for each article |
| `Text` or `Article_Text` | The source text to summarize |
| `Summary` or `Reference_Summary` | Reference summary (for training/evaluation) |
| `Image_Paths` | Comma-separated paths to associated images |

### Image Features

MULSUM uses pre-extracted CLIP features stored in LMDB format for efficient training:

1. Extract CLIP features from your images using the provided scripts
2. Store features in LMDB databases:
   - `data/LMDB/CLIP_train_features.lmdb`
   - `data/LMDB/CLIP_validation_features.lmdb`
   - `data/LMDB/CLIP_test_features.lmdb`

### Pre-trained Weights

Download the stage-1 pretrained mm_projector weights and place them in:
```
checkpoints/llava-vicuna-v1-5-7b-stage1/mm_projector.bin
```

## Usage

### Training

Train the model using the configuration file:

```bash
# Single GPU training
python src/train/train_image_text.py --config configs/image_text_config.json

# Multi-GPU training with torchrun
torchrun --standalone --nnodes=1 --nproc_per_node=4 \
    src/train/train_image_text.py --config configs/image_text_config.json
```

#### Key Training Parameters

Edit `configs/image_text_config.json` to customize training:

```json
{
  "training": {
    "output_dir": "checkpoints/imagetext-summarizer",
    "num_train_epochs": 10,
    "per_device_train_batch_size": 16,
    "learning_rate": 2e-5,
    "lora_enable": true,
    "lora_r": 128,
    "lora_alpha": 32
  }
}
```

### Inference

Generate summaries for test data:

```bash
python src/image_text_inference.py --config configs/image_text_config.json
```

Results will be saved to the path specified in `inference.csv_output_path`.

### Ablation Studies

Run ablation experiments to analyze the contribution of different components:

```bash
python src/image_text_inference_ablation.py --config configs/image_text_config.json
```

Available ablation types (set in config):
- `random_irrelevant_images`: Use random images instead of relevant ones
- `no_real_images`: Use random Gaussian features instead of real image features
- `without_images`: Text-only summarization (no images)

### Image Selection

Select relevant images for summaries using the image selection module:

```bash
python src/image_selection.py --config configs/image_text_config.json
```

Selection strategies:
- **S1**: Whole summary embedding similarity
- **S2**: SRL-based proposition embedding
- **MMR**: Maximal Marginal Relevance (balances relevance and diversity)

## Configuration

The main configuration file (`configs/image_text_config.json`) contains all settings:

| Section | Description |
|---------|-------------|
| `model` | Base model, checkpoint paths, mm_projector settings |
| `data` | Dataset paths, LMDB paths, max images |
| `training` | Epochs, batch size, learning rate, LoRA settings |
| `inference` | Output path, generation parameters |
| `wandb` | Weights & Biases logging (disabled by default) |
| `ablation` | Ablation study settings |
| `image_selection` | Image selection strategy and parameters |

## Citation

If you use MULSUM in your research, please cite our paper:

```bibtex
@inproceedings{ali2026mulsum,
  title={MULSUM: A Multimodal Summarization System with Vis-Aligner and Diversity-Aware Image Selection},
  author={Ali, Abid and Molla-Aliod, Diego and Naseem, Usman},
  booktitle={Proceedings of the 2026 Conference of the European Chapter of the Association for Computational Linguistics (EACL)},
  year={2026}
}
```

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

This project builds upon several excellent open-source projects:

- [LLaVA](https://github.com/haotian-liu/LLaVA) - Large Language and Vision Assistant
- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [PEFT](https://github.com/huggingface/peft) - Parameter-Efficient Fine-Tuning

## Contact

For questions or issues, please open a GitHub issue or contact the authors.
