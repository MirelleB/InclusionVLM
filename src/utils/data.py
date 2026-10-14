import os
from datasets import load_from_disk, Dataset, concatenate_datasets, Features, Image, Value, load_dataset

def read_txt_file(path):
    try:
        with open(path, 'r') as file:
            lines = [line.strip() for line in file]
        return lines
    except FileNotFoundError:
        print(f"Error: The file '{path}' was not found.")
        return []

def load_lora_dataset(dataset_path, translation_path, langs=["spanish", "wixarika"]):
    dataset = load_from_disk(dataset_path)
    for lang in langs:
        file_path = os.path.join(translation_path, f"polaris-13k-english-{lang}.txt")
        if not os.path.exists(file_path):
             # Try other naming convention if first fails
             file_path = os.path.join(translation_path, f"polaris-13k-spanish-{lang}.txt")
        
        lines = read_txt_file(file_path)
        if lines:
            dataset = dataset.add_column(name=lang, column=lines)
    return dataset

def expand_multilingual_dataset(dataset, langs=["wixarika", "maya", "bribri", "guarani"]):
    def image_generator(hf_dataset):
        for row in hf_dataset:
            img = row["img"]
            for lang in langs:
                if lang in row:
                    yield {
                        "img": img,
                        "ref": row[lang],
                        "lang": lang
                    }

    new_features = Features({
        "img": Image(),
        "ref": Value("string"),
        "lang": Value("string")
    })

    expanded_ds = Dataset.from_generator(
        image_generator,
        gen_kwargs={"hf_dataset": dataset},
        features=new_features
    )
    return expanded_ds

def create_paligemma_collate_fn(processor):
    def collate_fn(examples):
        images = [example["img"].convert("RGB") if "img" in example else example["image"].convert("RGB") for example in examples]
        
        # Determine language for prompt
        if "lang" in examples[0]:
            texts = [f"<image> Generar pie de foto en {item['lang']} " for item in examples]
        else:
            texts = ["<image> Generar pie de foto " for _ in range(len(examples))]
            
        labels = [example.get("ref", example.get("wixarika", example.get("target_caption", ""))) for example in examples]

        inputs = processor(
            text=texts, images=images, suffix=labels,
            return_tensors="pt", padding="longest"
        )
        return inputs
    return collate_fn

def load_cpt_dataset(monolingual_path, english_subset_size=500):
    local_ds = load_from_disk(monolingual_path)
    en_ds = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)
    
    en_ds_list = []
    for i, entry in enumerate(en_ds):
        if i >= english_subset_size: break
        en_ds_list.append({"text": entry["text"]})
    
    en_ds_final = Dataset.from_list(en_ds_list)
    combined_ds = concatenate_datasets([local_ds, en_ds_final]).shuffle(seed=42)
    return combined_ds
