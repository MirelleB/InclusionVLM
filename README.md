# AmericasNLP 2026: Multimodal Training & Evaluation

This repository contains professionalized scripts for training and evaluating vision-language models (VLM) and language models (LM) for low-resource languages, specifically for the AmericasNLP 2026 competition.
## Hugging Face Collection
To access the models and datasets used in this submission, please visit the MirelleCB/americasnlp-2026 collection on Hugging Face at https://huggingface.co/collections/MirelleCB/americasnlp-2026.

## Features
- **LoRA Fine-tuning**: Efficient fine-tuning for PaliGemma models.
- **Multilingual Support**: Combined training for multiple indigenous languages (Wixarika, Maya, Bribri, Guarani).
- **Continued Pre-Training (CPT)**: Two-step process to specialize the LM backbone before VLM integration.
- **Evaluation**: Standard metrics (chrF++) for evaluating image captioning performance.

## Repository Structure
```
americasnlp-repo/
├── configs/            # Configuration files (optional)
├── src/
│   ├── train/          # Training scripts
│   │   ├── train_lora.py         # Single language LoRA fine-tuning
│   │   ├── train_multilingual.py # Multilingual LoRA fine-tuning
│   │   ├── train_cpt_step1.py    # Specialized LM training (Gemma)
│   │   └── train_cpt_full.py     # VLM weight swap and fine-tuning
│   ├── utils/          # Helper modules
│   │   ├── data.py               # Data loading and preprocessing
│   │   ├── model.py              # Model initialization and LoRA setup
│   │   └── weight_swap.py        # LM swapping logic for CPT
│   └── evaluate.py     # Evaluation script
├── notebooks/          # Original Jupyter notebooks for reference
├── requirements.txt    # Python dependencies
└── HOWTO.md            # Detailed usage guide
```

## Installation
```bash
pip install -r requirements.txt
```

## Quick Start

### 1. LoRA Fine-tuning
```bash
python -m src.train.train_lora \
    --dataset_path path/to/dataset \
    --translation_path path/to/translations \
    --output_dir ./output/lora_model
```

### 2. Continued Pre-Training (CPT)
**Step 1: Train specialized Gemma**
```bash
python -m src.train.train_cpt_step1 \
    --monolingual_path path/to/monolingual_data \
    --output_dir ./output/specialized_gemma
```

**Step 2: Full VLM Fine-tuning**
```bash
python -m src.train.train_cpt_full \
    --specialized_lm_path ./output/specialized_gemma-merged \
    --dataset_path path/to/dataset \
    --translation_path path/to/translations
```

### 3. Evaluation
```bash
python -m src.evaluate \
    --model_path ./output/final_model \
    --jsonl_path data/dev/wixarika/wixarika.jsonl \
    --image_dir data/dev/wixarika/
```

## License
MIT
