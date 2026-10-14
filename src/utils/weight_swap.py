import torch
from transformers import PaliGemmaForConditionalGeneration, AutoModelForCausalLM

def swap_lm_weights(vlm_model, specialized_lm):
    """
    Swaps the language model component of a PaliGemma model with a specialized Gemma model.
    """
    print("Swapping language model component...")
    # In PaliGemma, the LM is stored in the .language_model attribute
    vlm_model.language_model = specialized_lm
    print("Weight swap complete.")
    return vlm_model

def round_up(n, m=64):
    return ((n + m - 1) // m) * m

def resize_and_align_vocab(vlm_model, tokenizer, padding=True):
    """
    Resizes the model's embeddings to match the tokenizer's vocabulary size.
    """
    vocab_size = len(tokenizer)
    if padding:
        target_vocab_size = round_up(vocab_size)
    else:
        target_vocab_size = vocab_size
        
    print(f"Aligning shapes: Tokenizer({vocab_size}) -> Target Model({target_vocab_size})")
    
    vlm_model.language_model.resize_token_embeddings(target_vocab_size)
    
    # Update configurations
    vlm_model.config.vocab_size = target_vocab_size
    vlm_model.language_model.config.vocab_size = target_vocab_size
    
    if hasattr(vlm_model.config, "text_config"):
        vlm_model.config.text_config.vocab_size = target_vocab_size
        
    # Safety check for LM head
    if hasattr(vlm_model.language_model, "lm_head"):
        vlm_model.language_model.lm_head.out_features = target_vocab_size
        
    return vlm_model
