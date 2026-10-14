#!/usr/bin/env python3
import argparse
import os
import torch
from transformers import TrainingArguments, Trainer, AutoProcessor, AutoModelForCausalLM, PaliGemmaForConditionalGeneration
from peft import LoraConfig, get_peft_model
from src.utils.weight_swap import swap_lm_weights, resize_and_align_vocab
from src.utils.data import load_lora_dataset, create_paligemma_collate_fn

def main():
    parser = argparse.ArgumentParser(description="Step 2: Full Fine-tuning PaliGemma with Specialized LM.")
    parser.add_argument("--vlm_model_id", type=str, default="google/paligemma2-10b-pt-224", help="Base PaliGemma model ID")
    parser.add_argument("--specialized_lm_path", type=str, required=True, help="Path to the merged specialized Gemma model from Step 1")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the saved dataset on disk")
    parser.add_argument("--translation_path", type=str, required=True, help="Path to translation text files")
    parser.add_argument("--output_dir", type=str, default="./output/paligemma_cpt_full", help="Directory to save the final model")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Training batch size")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Load Processor and Models
    processor = AutoProcessor.from_pretrained(args.vlm_model_id)
    
    vlm_model = PaliGemmaForConditionalGeneration.from_pretrained(
        args.vlm_model_id, torch_dtype=torch.bfloat16
    ).to(device)

    specialized_lm = AutoModelForCausalLM.from_pretrained(
        args.specialized_lm_path, torch_dtype=torch.bfloat16
    )

    # 2. Swap LM Weights
    vlm_model = swap_lm_weights(vlm_model, specialized_lm)

    # 3. Align Vocab
    vlm_model = resize_and_align_vocab(vlm_model, processor.tokenizer)

    # 4. Freeze Backbone and Configure Trainable Parameters
    for param in vlm_model.model.vision_tower.parameters():
        param.requires_grad = False
    
    for param in vlm_model.model.multi_modal_projector.parameters():
        param.requires_grad = True

    # 5. Apply LoRA
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
        modules_to_save=["multi_modal_projector"]
    )
    vlm_model = get_peft_model(vlm_model, lora_config)

    # 6. Load Dataset
    dataset = load_lora_dataset(args.dataset_path, args.translation_path)
    collate_fn = create_paligemma_collate_fn(processor)

    # 7. Training Arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        save_strategy="epoch",
        logging_steps=100,
        learning_rate=args.lr,
        remove_unused_columns=False,
        fp16=False,
        dataloader_pin_memory=False,
    )

    trainer = Trainer(
        model=vlm_model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collate_fn,
    )

    # 8. Train and Save
    trainer.train()
    
    final_path = os.path.join(args.output_dir, "final_merged")
    print("Merging and saving final model...")
    merged_model = vlm_model.merge_and_unload()
    merged_model.save_pretrained(final_path, safe_serialization=True)
    processor.save_pretrained(final_path)

if __name__ == "__main__":
    main()
