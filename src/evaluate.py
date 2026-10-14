#!/usr/bin/env python3
import argparse
import torch
import sacrebleu
from tqdm import tqdm
from datasets import Dataset, Features, Value, Image as HFImage
import pandas as pd
from src.utils.model import get_paligemma_model_and_processor, load_peft_model
from sacrebleu.metrics import CHRF

def chrf_score(ground_truth: str, generated_text: str) -> float:
    chrf = CHRF(word_order=2)
    return chrf.sentence_score(generated_text, [ground_truth]).score

def load_eval_dataset(jsonl_path, image_dir, lang_name):
    df = pd.read_json(jsonl_path, lines=True)
    features = Features({
        "image": HFImage(),
        "target_caption": Value("string")
    })
    
    data = {
        "image": [os.path.join(image_dir, i) for i in df['filename']],
        "target_caption": df['target_caption'].to_list()
    }
    return Dataset.from_dict(data, features=features)

def evaluate_chrf(model, processor, dataset, prompt, device):
    model.eval()
    predictions = []
    references = []

    print("Running inference...")
    for example in tqdm(dataset):
        image = example["image"].convert("RGB")
        inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)

        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=512)

        full_text = processor.decode(output_ids[0], skip_special_tokens=True)
        # Clean up output
        pred_text = full_text.replace(prompt.replace("<image>", "").strip(), "").strip()

        predictions.append(pred_text)
        references.append(example["target_caption"])

    return predictions, references

def main():
    parser = argparse.ArgumentParser(description="Evaluate PaliGemma checkpoints.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model or adapter")
    parser.add_argument("--base_model_id", type=str, default="google/paligemma-3b-pt-224", help="Base model ID if using adapters")
    parser.add_argument("--jsonl_path", type=str, required=True, help="Path to evaluation JSONL file")
    parser.add_argument("--image_dir", type=str, required=True, help="Directory containing evaluation images")
    parser.add_argument("--prompt", type=str, default="<image> Generar pie de foto ", help="Prompt used during training")
    
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Load Model
    model, processor = get_paligemma_model_and_processor(args.base_model_id, device=device)
    if os.path.exists(os.path.join(args.model_path, "adapter_config.json")):
        model = load_peft_model(model, args.model_path, device=device)
        model = model.merge_and_unload()
    else:
        # Assume it's a merged model
        model = PaliGemmaForConditionalGeneration.from_pretrained(args.model_path, torch_dtype=torch.bfloat16).to(device)
    
    model.eval()

    # 2. Load Dataset
    import os
    dataset = load_eval_dataset(args.jsonl_path, args.image_dir, "lang")

    # 3. Run Inference
    predictions, references = evaluate_chrf(model, processor, dataset, args.prompt, device)

    # 4. Compute Scores
    df_result = pd.DataFrame({'predicts': predictions, 'refs': references})
    df_result['chrf'] = df_result.apply(lambda row: chrf_score(row["refs"], row["predicts"]), axis=1)
    
    corpus_chrf = sacrebleu.corpus_chrf(hypotheses=predictions, references=[references], word_order=2)
    
    print("\n" + "="*30)
    print(f"Mean chrF++ score: {df_result['chrf'].mean():.2f}")
    print(f"Corpus chrF++ score: {corpus_chrf}")
    print("="*30)

if __name__ == "__main__":
    main()
