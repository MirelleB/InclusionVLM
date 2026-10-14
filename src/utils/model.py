import torch
from transformers import (
    PaliGemmaProcessor,
    PaliGemmaForConditionalGeneration,
    AutoTokenizer,
    AutoModelForCausalLM
)
from peft import LoraConfig, get_peft_model, PeftModel

def get_paligemma_model_and_processor(model_id="google/paligemma-3b-pt-224", device="auto"):
    processor = PaliGemmaProcessor.from_pretrained(model_id)
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map=device
    )
    return model, processor

def apply_lora(model, r=8, lora_alpha=32, target_modules=["q_proj", "v_proj"], lora_dropout=0.05, task_type="CAUSAL_LM"):
    lora_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        task_type=task_type
    )
    model = get_peft_model(model, lora_config)
    return model

def get_gemma_model_and_tokenizer(model_id="google/gemma-2b", device="auto"):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map=device
    )
    return model, tokenizer

def load_peft_model(base_model, adapter_path, device="cuda"):
    model = PeftModel.from_pretrained(base_model, adapter_path)
    return model
