#!/usr/bin/env python3
"""
Gemma Machine Translation Post-Training Evaluation Script
=========================================================

This script evaluates a fine-tuned Gemma translation model on a test set.
It computes standard machine translation metrics: chrF/chrF++ and BLEU.

Best Practices Implemented:
---------------------------
1. **Instruction Alignment**: Uses the exact same Gemma instruction template format used in training:
   `<start_of_turn>user\n{instruction}<end_of_turn>\n<start_of_turn>model\n`
   This ensures that the model is evaluated in its optimal instruction-following state.

2. **Greedy Decoding**: Uses greedy decoding (do_sample=False, num_beams=1) which is the standard baseline
   in the literature for evaluating translation performance.

3. **Prefix Slicing**: Decodes *only* the newly generated tokens (slicing out the prompt tokens) to ensure
   that metric evaluation is calculated purely on the predicted translation.

4. **SacreBLEU / chrF**: Evaluates translations using standard metrics with `sacrebleu` and `evaluate` packages.

5. **Sample Logging**: Prints random translations (source, prediction, reference) to the console for quick
   visual analysis of the model's output quality.

6. **Output Exporting**: Saves full prediction results to a CSV/JSONL file if `--output_file` is specified
   for further analysis.
"""

import os
import argparse
import random
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import evaluate

# Use the same list of prompt templates as the training script
PROMPT_TEMPLATES = [
    "Translate the following text from {source_lang} to {target_lang}:\nText: {source_text}",
    "Translate from {source_lang} to {target_lang}.\n\nSource: {source_text}\n\nTranslation:",
    "How would you say this in {target_lang}? Translate from {source_lang}:\n\"{source_text}\"",
    "Provide the {target_lang} translation for this {source_lang} sentence:\n{source_text}",
    "Translate the text below from {source_lang} to {target_lang}.\nSource text: {source_text}\nTarget text:"
]


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned Gemma Translation Model.")
    
    # Model & Test dataset paths
    parser.add_argument("--model_path", type=str, required=True, help="Path to the fine-tuned model checkpoint directory")
    parser.add_argument("--test_file", type=str, required=True, help="Path to local test dataset file (json/jsonl/csv/tsv)")
    
    # Column mappings
    parser.add_argument("--source_lang_column", type=str, default="source_lang", help="Column containing source language name")
    parser.add_argument("--target_lang_column", type=str, default="target_lang", help="Column containing target language name")
    parser.add_argument("--source_text_column", type=str, default="source_text", help="Column containing source text")
    parser.add_argument("--target_text_column", type=str, default="target_text", help="Column containing target text")
    
    # Prompt Template
    parser.add_argument("--template_index", type=int, default=0, choices=[0, 1, 2, 3, 4], 
                        help="Index of the PROMPT_TEMPLATE to use during evaluation")
    
    # Generation & Evaluation configuration
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for batch inference")
    parser.add_argument("--max_new_tokens", type=int, default=128, help="Maximum number of generated tokens")
    parser.add_argument("--output_file", type=str, help="Optional CSV path to save predictions alongside source/reference text")
    parser.add_argument("--num_samples_to_show", type=int, default=5, help="Number of random translation samples to print to console")
    
    return parser.parse_args()


def load_data(args):
    """
    Loads test dataset and ensures columns exist.
    """
    ext = os.path.splitext(args.test_file)[-1].lower()
    if ext in [".json", ".jsonl"]:
        dataset = load_dataset("json", data_files={"test": args.test_file})
    elif ext == ".csv":
        dataset = load_dataset("csv", data_files={"test": args.test_file})
    elif ext == ".tsv":
        dataset = load_dataset("csv", delimiter="\t", data_files={"test": args.test_file})
    else:
        raise ValueError("Unsupported test dataset file extension. Must be json, jsonl, csv, or tsv.")
        
    return dataset["test"]


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 1. Load Tokenizer & Model
    print(f"Loading tokenizer and model from: {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    
    # Handle padding
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        device_map="auto" if device == "cuda" else None
    ).to(device if device == "cpu" else "cuda")
    
    model.eval()
    
    # 2. Load Dataset
    print(f"Loading evaluation dataset from: {args.test_file}...")
    test_ds = load_data(args)
    
    # Prepare batch execution variables
    sources = test_ds[args.source_text_column]
    references = test_ds[args.target_text_column]
    src_langs = test_ds[args.source_lang_column] if args.source_lang_column in test_ds.column_names else ["Source Language"] * len(sources)
    tgt_langs = test_ds[args.target_lang_column] if args.target_lang_column in test_ds.column_names else ["Target Language"] * len(sources)
    
    # Define prompt template for consistent evaluation
    template = PROMPT_TEMPLATES[args.template_index]
    print(f"Evaluating with prompt template index {args.template_index}:")
    print(f'"{template}"')
    
    predictions = []
    
    # 3. Run Batch Inference
    print(f"Running inference over {len(sources)} test samples with batch size {args.batch_size}...")
    
    for i in tqdm(range(0, len(sources), args.batch_size)):
        batch_end = min(i + args.batch_size, len(sources))
        
        batch_prompts = []
        for j in range(i, batch_end):
            instruction = template.format(
                source_lang=src_langs[j], 
                target_lang=tgt_langs[j], 
                source_text=sources[j]
            )
            # Create exact user prompt ending with model turn initiation
            prompt = f"<start_of_turn>user\n{instruction}<end_of_turn>\n<start_of_turn>model\n"
            batch_prompts.append(prompt)
            
        # Tokenize batch
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Determine prompt token length to isolate generated tokens
        prompt_len = inputs["input_ids"].shape[1]
        
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,  # Greedy decoding is standard for translation evaluation
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id
            )
            
        # Extract only newly generated translation tokens (slicing out the prompt)
        batch_generated_tokens = [seq[prompt_len:] for seq in generated_ids]
        
        # Decode
        batch_predictions = tokenizer.batch_decode(batch_generated_tokens, skip_special_tokens=True)
        predictions.extend([pred.strip() for pred in batch_predictions])
        
    # 4. Compute Metrics
    print("Computing metrics...")
    chrf_metric = evaluate.load("chrf")
    bleu_metric = evaluate.load("sacrebleu")
    
    # Prepare references format for sacrebleu (list of lists of reference text)
    sacre_refs = [[ref] for ref in references]
    
    chrf_score = chrf_metric.compute(predictions=predictions, references=sacre_refs)["score"]
    bleu_score = bleu_metric.compute(predictions=predictions, references=sacre_refs)["score"]
    
    print("\n" + "="*50)
    print("                 EVALUATION RESULTS")
    print("="*50)
    print(f"Corpus BLEU Score   : {bleu_score:.2f}")
    print(f"Corpus chrF++ Score : {chrf_score:.2f}")
    print("="*50 + "\n")
    
    # 5. Print random samples for visual examination
    print(f"Showing {args.num_samples_to_show} random prediction samples:")
    sample_indices = random.sample(range(len(sources)), min(args.num_samples_to_show, len(sources)))
    for idx in sample_indices:
        print("-"*40)
        print(f"[{src_langs[idx]} -> {tgt_langs[idx]}]")
        print(f"Source   : {sources[idx]}")
        print(f"Reference: {references[idx]}")
        print(f"Predicted: {predictions[idx]}")
    print("-"*40 + "\n")
    
    # 6. Save predictions to disk if requested
    if args.output_file:
        print(f"Saving full results to {args.output_file}...")
        df_results = pd.DataFrame({
            "source_lang": src_langs,
            "target_lang": tgt_langs,
            "source_text": sources,
            "reference_text": references,
            "predicted_translation": predictions
        })
        # Check output directory
        out_dir = os.path.dirname(args.output_file)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            
        ext = os.path.splitext(args.output_file)[-1].lower()
        if ext == ".csv":
            df_results.to_csv(args.output_file, index=False)
        elif ext in [".json", ".jsonl"]:
            df_results.to_json(args.output_file, orient="records", lines=True)
        else:
            # Default to csv
            df_results.to_csv(args.output_file, index=False)
        print("Results successfully saved.")


if __name__ == "__main__":
    main()
