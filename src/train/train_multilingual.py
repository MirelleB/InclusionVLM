#!/usr/bin/env python3
import argparse
import os
import torch
from transformers import TrainingArguments, Trainer
from src.utils.model import get_paligemma_model_and_processor, apply_lora
from src.utils.data import load_lora_dataset, expand_multilingual_dataset, create_paligemma_collate_fn

def main():
    parser = argparse.ArgumentParser(description="Multilingual LoRA fine-tuning for PaliGemma.")
    parser.add_argument("--model_id", type=str, default="google/paligemma-3b-pt-224", help="HuggingFace model ID")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the saved dataset on disk")
    parser.add_argument("--translation_path", type=str, required=True, help="Path to translation text files")
    parser.add_argument("--output_dir", type=str, default="./output/paligemma_multilingual", help="Directory to save the model")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    
    args = parser.parse_args()

    # 1. Load Model and Processor
    model, processor = get_paligemma_model_and_processor(args.model_id)

    # 2. Apply LoRA
    model = apply_lora(model)

    # 3. Load and Preprocess Dataset
    langs = ["spanish", "wixarika", "maya", "bribri", "guarani"]
    base_dataset = load_lora_dataset(args.dataset_path, args.translation_path, langs=langs)
    expanded_dataset = expand_multilingual_dataset(base_dataset, langs=["wixarika", "maya", "bribri", "guarani"])
    collate_fn = create_paligemma_collate_fn(processor)

    # 4. Training Arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        save_strategy="epoch",
        logging_steps=100,
        learning_rate=args.learning_rate,
        remove_unused_columns=False,
        fp16=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=expanded_dataset,
        data_collator=collate_fn,
    )

    # 5. Train and Save
    trainer.train()
    model.save_pretrained(os.path.join(args.output_dir, "final_model"))
    processor.save_pretrained(os.path.join(args.output_dir, "final_model"))

if __name__ == "__main__":
    main()
