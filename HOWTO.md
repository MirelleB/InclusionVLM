# How-To Guide: Training and Evaluation

This guide provides detailed instructions on how to use the scripts in this repository.

## 1. Preparing Data
The scripts expect datasets saved using the HuggingFace `datasets` library and auxiliary translation text files.

### Dataset Format
- **HuggingFace Dataset**: Should contain an `img` column (PIL images) and optionally reference captions.
- **Translation Files**: Text files named `polaris-13k-spanish-[lang].txt` where each line corresponds to a row in the dataset.

## 2. LoRA Fine-Tuning
To fine-tune PaliGemma for a specific language pair:

```bash
python -m src.train.train_lora \
    --dataset_path data/polaris_dataset/filter_13k \
    --translation_path data/polaris_dataset/polaris-translations/ \
    --output_dir ./output/paligemma_wixarika \
    --epochs 10 \
    --batch_size 4
```

For multilingual training (Maya, Bribri, Guarani, Wixarika):

```bash
python -m src.train.train_multilingual \
    --dataset_path data/polaris_dataset/filter_13k \
    --translation_path data/polaris_dataset/polaris-translations/ \
    --output_dir ./output/paligemma_multilingual
```

## 3. Continued Pre-Training (CPT) Workflow
CPT is recommended when you have a significant amount of monolingual text data for the target language.

### Step 1: Specialize the LM (Gemma)
Train the language model backbone on a mix of target language data and a small subset of English (to prevent forgetting).

```bash
python -m src.train.train_cpt_step1 \
    --monolingual_path data/vlm_experiments/LM_vLM/wixarika_monolingual \
    --output_dir ./output/gemma-wixarika
```
This will produce a merged model at `./output/gemma-wixarika-merged`.

### Step 2: Integrate with PaliGemma
Swap the language model of PaliGemma with your specialized version and fine-tune the multi-modal projector.

```bash
python -m src.train.train_cpt_full \
    --specialized_lm_path ./output/gemma-wixarika-merged \
    --dataset_path data/polaris_dataset/filter_13k \
    --translation_path data/polaris_dataset/polaris-translations/ \
    --output_dir ./output/paligemma_cpt_wixarika
```

## 4. Evaluation
Evaluate your model using the chrF++ metric.

```bash
python -m src.evaluate \
    --model_path ./output/paligemma_wixarika/final_model \
    --jsonl_path americasnlp2026/data/dev/wixarika/wixarika.jsonl \
    --image_dir americasnlp2026/data/dev/wixarika/ \
    --prompt "<image> Generar pie de foto "
```

## 5. Troubleshooting
- **Memory Issues**: Decrease `--batch_size` or use `--gradient_accumulation_steps` if supported.
- **Model Version**: Ensure you are using `transformers>=4.45.0` for PaliGemma 2 support.
- **Paths**: Always check that paths to datasets and checkpoints are correct relative to the root directory.
