#!/usr/bin/env python3
import argparse
import os
import torch
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
from src.utils.model import get_gemma_model_and_tokenizer, apply_lora
from src.utils.data import load_cpt_dataset

def main():
    parser = argparse.ArgumentParser(description="Step 1: Continued Pre-training for Gemma.")
    parser.add_argument("--model_id", type=str, default="google/gemma-2b", help="Base Gemma model ID")
    parser.add_argument("--monolingual_path", type=str, required=True, help="Path to the monolingual dataset on disk")
    parser.add_argument("--output_dir", type=str, default="./output/gemma_cpt", help="Directory to save the model")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Training batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    
    args = parser.parse_args()

    # 1. Load Model and Tokenizer
    model, tokenizer = get_gemma_model_and_tokenizer(args.model_id)

    # 2. Apply LoRA
    model = apply_lora(model, r=16, target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])

    # 3. Load and Tokenize Dataset
    dataset = load_cpt_dataset(args.monolingual_path)
    
    def tokenize_fn(examples):
        return tokenizer(examples["text"], truncation=True, max_length=512)

    tokenized_ds = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])

    # 4. Training Arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        bf16=True,
        logging_steps=100,
        save_total_limit=2,
        optim="paged_adamw_8bit"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )

    # 5. Train and Save
    trainer.train()
    
    print("Merging LoRA weights into base model...")
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(args.output_dir + "-merged")
    tokenizer.save_pretrained(args.output_dir + "-merged")

if __name__ == "__main__":
    main()
