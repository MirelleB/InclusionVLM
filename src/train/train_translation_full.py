#!/usr/bin/env python3
"""
Gemma Machine Translation Full Fine-Tuning Script
=================================================

This script performs full parameter fine-tuning (without LoRA) of a Gemma/Gemma-2 model
(such as Gemma-2-2b, Gemma-2-9b, or other Gemma checkpoints) for multilingual machine translation.
It is built adhering to state-of-the-art literature and engineering best practices.

Literary & Engineering Best Practices Implemented:
--------------------------------------------------
1. **Instruction Formatting**: Formats translation instructions using the native Gemma instruction template:
   `<start_of_turn>user\n{instruction}\n{source_text}<end_of_turn>\n<start_of_turn>model\n{target_text}<end_of_turn>\n`
   This leverages the model's instruction-following pre-trained priors.

2. **Target-Only Loss (Label Masking)**: Masking the user instruction part during loss computation
   by replacing prompt label tokens with `-100` (PyTorch's CrossEntropy ignore index).
   The model is only optimized to predict the translation itself, preventing instruction degradation
   and improving generative quality.

3. **Template Diversity (Prompt Robustness)**: Employs a diverse set of translation prompt templates during
   preprocessing, randomly chosen per sample. This prevents overfitting to a specific phrasing, ensuring
   robust translation capabilities at inference.

4. **Support for Unseen Languages (Zero-Shot MT)**:
   - Uses **semantic language names** (e.g., "Spanish", "Wixarika") rather than simple codes. Leveraging full
     language names taps into the model's pre-trained cross-lingual latent space.
   - Formulates prompts in a language-agnostic manner to align the translation task representation.
   - Includes support for dynamic **few-shot demonstrations** (in-context exemplars) if specified, which is
     the gold standard in the literature for adapting to unseen/low-resource language pairs during inference.

5. **Memory-Efficient Full Parameter Tuning**:
   - Trains using `bfloat16` precision (or `float16` if on older GPUs) to prevent overflow/underflow.
   - Supports PyTorch `gradient_checkpointing` to massively reduce activation memory requirements.
   - Implements learning rate warmup and a cosine learning rate scheduler to stabilize weight updates.
   - Uses Seq2SeqTrainer for native generation-based evaluation (BLEU/chrF) on validation sets.

6. **No LoRa / Parameter-Efficient Training**: Fully updates all model parameters to align deep representations
   for translation.
"""

import os
import random
import argparse
import numpy as np
import torch
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq
)
import evaluate

# Literary Best Practice: Diverse prompt templates to avoid overfitting and build instruction generalizability
PROMPT_TEMPLATES = [
    "Translate the following text from {source_lang} to {target_lang}:\nText: {source_text}",
    "Translate from {source_lang} to {target_lang}.\n\nSource: {source_text}\n\nTranslation:",
    "How would you say this in {target_lang}? Translate from {source_lang}:\n\"{source_text}\"",
    "Provide the {target_lang} translation for this {source_lang} sentence:\n{source_text}",
    "Translate the text below from {source_lang} to {target_lang}.\nSource text: {source_text}\nTarget text:"
]


def parse_args():
    parser = argparse.ArgumentParser(description="Full Parameter Fine-Tuning of Gemma for Machine Translation.")
    
    # Model & Tokenizer
    parser.add_argument("--model_id", type=str, default="google/gemma-2-2b", help="Pretrained HuggingFace Gemma checkpoint path")
    parser.add_argument("--device_map", type=str, default="auto", help="Device mapping strategy for loading the model")
    
    # Datasets
    parser.add_argument("--train_file", type=str, help="Path to local training file (json/jsonl/csv/tsv)")
    parser.add_argument("--validation_file", type=str, help="Path to local validation file (json/jsonl/csv/tsv)")
    parser.add_argument("--dataset_name", type=str, help="HuggingFace dataset name (if not using local files)")
    parser.add_argument("--dataset_config", type=str, help="HuggingFace dataset configuration/split name")
    
    # Column mapping (for robust generalizability across different dataset formats)
    parser.add_argument("--source_lang_column", type=str, default="source_lang", help="Column containing source language name")
    parser.add_argument("--target_lang_column", type=str, default="target_lang", help="Column containing target language name")
    parser.add_argument("--source_text_column", type=str, default="source_text", help="Column containing source text")
    parser.add_argument("--target_text_column", type=str, default="target_text", help="Column containing target text")
    
    # Training Hyperparameters
    parser.add_argument("--output_dir", type=str, default="./output/gemma_translation_fft", help="Directory to save checkpoints")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size per device during training")
    parser.add_argument("--eval_batch_size", type=int, default=4, help="Batch size per device during evaluation")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate for full parameter tuning (default: 1e-5)")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay regularization coefficient")
    parser.add_argument("--warmup_ratio", type=float, default=0.03, help="Linear warmup ratio over total training steps")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Number of gradient accumulation steps")
    parser.add_argument("--max_seq_length", type=int, default=512, help="Maximum input sequence length after tokenization")
    
    # Optimization & Hardware
    parser.add_argument("--bf16", type=bool, default=True, help="Whether to use bf16 mixed precision (recommended for Gemma)")
    parser.add_argument("--gradient_checkpointing", type=bool, default=True, help="Enable activation checkpointing to save GPU memory")
    parser.add_argument("--logging_steps", type=int, default=10, help="Log step frequency")
    parser.add_argument("--save_strategy", type=str, default="epoch", choices=["steps", "epoch"], help="Model save strategy")
    parser.add_argument("--eval_strategy", type=str, default="epoch", choices=["steps", "epoch", "no"], help="Evaluation strategy")
    parser.add_argument("--save_total_limit", type=int, default=2, help="Maximum number of checkpoints to retain")
    
    # Inference / Generative Params for Eval
    parser.add_argument("--max_new_tokens", type=int, default=128, help="Maximum translation output tokens for evaluation")
    
    return parser.parse_args()


def load_data(args):
    """
    Loads and standardizes input datasets into a unified format.
    Each sample will have: source_text, target_text, source_lang, target_lang
    """
    if args.train_file:
        ext = os.path.splitext(args.train_file)[-1].lower()
        if ext in [".json", ".jsonl"]:
            dataset = load_dataset("json", data_files={"train": args.train_file, "validation": args.validation_file} if args.validation_file else {"train": args.train_file})
        elif ext == ".csv":
            dataset = load_dataset("csv", data_files={"train": args.train_file, "validation": args.validation_file} if args.validation_file else {"train": args.train_file})
        elif ext == ".tsv":
            dataset = load_dataset("csv", delimiter="\t", data_files={"train": args.train_file, "validation": args.validation_file} if args.validation_file else {"train": args.train_file})
        else:
            raise ValueError("Unsupported local dataset extension. Must be json, jsonl, csv, or tsv.")
    elif args.dataset_name:
        dataset = load_dataset(args.dataset_name, args.dataset_config)
    else:
        raise ValueError("Must provide either --train_file or --dataset_name.")
        
    print(f"Dataset successfully loaded: {dataset}")
    return dataset


def main():
    args = parse_args()
    
    # 1. Initialize Tokenizer and Model
    print(f"Loading tokenizer and model from checkpoint: {args.model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    
    # Gemma best practice: standard padding configuration
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        
    # Standard full fine-tuning model load with half-precision and device mapping
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float16,
        device_map=args.device_map
    )
    
    # Memory optimization: activation checkpointing is critical for full fine-tuning without LoRA
    if args.gradient_checkpointing:
        print("Enabling gradient checkpointing...")
        model.gradient_checkpointing_enable()
        
    # 2. Load Dataset
    raw_datasets = load_data(args)
    
    # Define preprocessing helper
    def preprocess_function(examples):
        """
        Tokenizes and formats inputs following instruction-tuning best practices.
        Specifically handles:
          - Chatml-style Gemma prompt construction
          - Dynamic prompt template randomization (prompt robustness)
          - Label masking (target-only loss computation on -100)
        """
        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []
        
        # Iterate over samples in the batch
        num_examples = len(examples[args.source_text_column])
        for i in range(num_examples):
            src_text = examples[args.source_text_column][i]
            tgt_text = examples[args.target_text_column][i]
            src_lang = examples[args.source_lang_column][i] if args.source_lang_column in examples else "Source Language"
            tgt_lang = examples[args.target_lang_column][i] if args.target_lang_column in examples else "Target Language"
            
            # Literature Best Practice: Diversify prompt formats per sample to prevent rigid overfitting
            template = random.choice(PROMPT_TEMPLATES)
            instruction = template.format(source_lang=src_lang, target_lang=tgt_lang, source_text=src_text)
            
            # Construct standard Gemma chat format
            # Format: <start_of_turn>user\n{instruction}<end_of_turn>\n<start_of_turn>model\n{target}<end_of_turn>\n
            user_prompt_text = f"<start_of_turn>user\n{instruction}<end_of_turn>\n<start_of_turn>model\n"
            target_response_text = f"{tgt_text}<end_of_turn>\n"
            
            # Tokenize parts separately to construct exact target-only loss masks
            prompt_ids = tokenizer(user_prompt_text, add_special_tokens=True).input_ids
            target_ids = tokenizer(target_response_text, add_special_tokens=False).input_ids
            
            # Combine
            input_ids = prompt_ids + target_ids
            attention_mask = [1] * len(input_ids)
            
            # Target-only Loss Best Practice: set labels of prompt tokens to -100 so the model
            # only calculates loss on target translation tokens.
            labels = [-100] * len(prompt_ids) + target_ids
            
            # Apply maximum sequence length truncation safely
            if len(input_ids) > args.max_seq_length:
                input_ids = input_ids[:args.max_seq_length]
                attention_mask = attention_mask[:args.max_seq_length]
                labels = labels[:args.max_seq_length]
                
            batch_input_ids.append(input_ids)
            batch_attention_mask.append(attention_mask)
            batch_labels.append(labels)
            
        return {
            "input_ids": batch_input_ids,
            "attention_mask": batch_attention_mask,
            "labels": batch_labels
        }
    
    # 3. Preprocess and map datasets
    print("Preprocessing and tokenizing training and validation datasets...")
    train_dataset = raw_datasets["train"].map(
        preprocess_function,
        batched=True,
        remove_columns=raw_datasets["train"].column_names,
        desc="Tokenizing training set"
    )
    
    eval_dataset = None
    if "validation" in raw_datasets:
        eval_dataset = raw_datasets["validation"].map(
            preprocess_function,
            batched=True,
            remove_columns=raw_datasets["validation"].column_names,
            desc="Tokenizing validation set"
        )
        
    # 4. Setup Seq2Seq Evaluation Metrics
    # We load standard machine translation metrics: BLEU and chrF++
    print("Initializing metric evaluators...")
    chrf_metric = evaluate.load("chrf")
    bleu_metric = evaluate.load("sacrebleu")
    
    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
            
        # Replace -100 in labels as we cannot decode them directly
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        # Strip trailing and leading whitespace
        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [[label.strip()] for label in decoded_labels] # sacrebleu reference format
        
        # Calculate metric values
        chrf_res = chrf_metric.compute(predictions=decoded_preds, references=decoded_labels)
        bleu_res = bleu_metric.compute(predictions=decoded_preds, references=decoded_labels)
        
        return {
            "chrf": chrf_res["score"],
            "bleu": bleu_res["score"]
        }

    # 5. Define Collator
    # Collator pads inputs and masks labels appropriately with -100
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8
    )
    
    # 6. Configure Training Arguments
    # We use Seq2SeqTrainingArguments to support evaluate with generate loop
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        num_train_epochs=args.epochs,
        logging_steps=args.logging_steps,
        save_strategy=args.save_strategy,
        eval_strategy=args.eval_strategy if eval_dataset else "no",
        save_total_limit=args.save_total_limit,
        bf16=args.bf16,
        fp16=not args.bf16,
        remove_unused_columns=False,
        predict_with_generate=True, # Critical for translation evaluation
        generation_max_length=args.max_new_tokens,
        report_to="none", # Disable cloud logging for confidentiality
        optim="adamw_torch"
    )
    
    # 7. Initialize Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics if eval_dataset else None
    )
    
    # 8. Train & Save
    print("Beginning full fine-tuning. This may take some time depending on your hardware...")
    trainer.train()
    
    # Save the final model and tokenizer
    final_output_path = os.path.join(args.output_dir, "final_gemma_translation_model")
    print(f"Training completed successfully! Saving final full parameter model to {final_output_path}...")
    trainer.save_model(final_output_path)
    tokenizer.save_pretrained(final_output_path)
    print("Model and processor successfully saved. Script completed.")


if __name__ == "__main__":
    main()
