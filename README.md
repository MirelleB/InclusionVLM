# AmericasNLP 2026: Culturally Grounded Image Captioning in Indigenous Languages with Vision-Language Models: Cascaded and Single-Stage Approaches

This repository contains the training and reproduction scripts for the InclusionVLM team's submission to the AmericasNLP 2026 Shared Task.
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

## Citation

If you use this code, pre-trained models, or dataset in your work, please cite our paper:

```bibtex
@inproceedings{bueno-garg-2026-culturally,
    title = "Culturally Grounded Image Captioning in Indigenous Languages with Vision-Language Models: Cascaded and Single-Stage Approaches",
    author = "Bueno, Mirelle  and
      Garg, Sushil",
    editor = "Mager, Manuel  and
      Ebrahimi, Abteen  and
      Bui, Minh Duc  and
      Pugh, Robert  and
      Oncevay, Arturo  and
      Chiruzzo, Luis  and
      Solano, Rolando Coto  and
      Rijhwani, Shruti  and
      von der Wense, Katharina",
    booktitle = "Proceedings of the Sixth Workshop on {NLP} for Indigenous Languages of the {A}mericas ({A}mericas{NLP})",
    month = jul,
    year = "2026",
    address = "San Diego, California, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2026.americasnlp-6.23/",
    doi = "10.18653/v1/2026.americasnlp-6.23",
    pages = "248--256",
    ISBN = "979-8-89176-415-6",
    abstract = "Culturally grounded image captioning for under-resourced Indigenous languages is challenging due to severe data scarcity and the need to describe culturally specific visual content. This paper describes our submission to the AmericasNLP 2026 shared task, where we evaluate two architectural paradigms for caption generation across Bribri, Guaran{\'i}, Yucatec Maya, Wix{\'a}rika, and Orizaba Nahuatl. First, we implement a cascaded system that combines a large vision-language model with a machine translation pipeline, showing that culturally contextualized, persona-based prompting improves over the official baseline in most comparable settings. Second, we develop a direct, end-to-end Single-stage approach by adapting PaliGemma 2 using LoRA fine-tuning, continued pre-training, and multilingual joint training. Our single-stage experiments show that, despite severe domain mismatch and reliance on synthetic training data, multilingual training and continued pre-training improve automatic chrF++ relative to single-language LoRA fine-tuning in some settings. Overall, cascaded pipelines remain the strongest among the evaluated approaches under current data constraints, while single-stage models remain a promising but currently data-limited path toward direct Indigenous-language image captioning."
}
