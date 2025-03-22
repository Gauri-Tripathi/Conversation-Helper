import os
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from datasets import Dataset, DatasetDict
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import AdaLoraConfig, get_peft_model
import yaml

def load_config(config_path: str):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_and_prepare_data(config):
    df = pd.read_csv(config['data']['csv_path'])
    dataset = Dataset.from_pandas(df)
    train_test_dataset = dataset.train_test_split(test_size=config['data']['train_test_split'], seed=config['data']['seed'])
    train_val_dataset = DatasetDict({
        'train': train_test_dataset['train'],
        'validation': train_test_dataset['test']
    })
    dataset_path = Path(config['data']['dataset_path'])
    dataset_path.mkdir(parents=True, exist_ok=True)
    train_val_dataset.save_to_disk(dataset_path)
    return dataset_path

def setup_model_and_tokenizer(config):
    quant_config = BitsAndBytesConfig(
        load_in_4bit=config['model']['load_in_4bit'],
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    tokenizer = AutoTokenizer.from_pretrained(config['model']['base_model'])
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        config['model']['base_model'],
        quantization_config=quant_config,
        device_map="auto",
        torch_dtype=torch.float16,
        use_flash_attention_2=config['model']['use_flash_attention'],
        use_cache=False
    )
    model = get_peft_model(model, AdaLoraConfig(**config['adalora']))
    return model, tokenizer

def preprocess_function(examples, tokenizer, max_length):
    inputs = [
        f"""Below is an instruction that describes a task, and an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{ex['instruction']}

### Input:
{ex['input']}

### Response:
{ex['output']}"""
        for ex in examples
    ]
    return tokenizer(inputs, padding="max_length", truncation=True, max_length=max_length, return_tensors=None)

def train_model(model, tokenizer, dataset_path, config):
    dataset = DatasetDict.load_from_disk(dataset_path)
    processed_dataset = dataset.map(
        lambda x: preprocess_function(x, tokenizer, config['data']['preprocessing']['max_length']),
        batched=True,
        remove_columns=dataset["train"].column_names,
        num_proc=4
    )
    output_dir = Path(config['output']['base_dir']) / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    trainer = Trainer(
        model=model,
        args=TrainingArguments(output_dir=str(output_dir), **config['training']),
        train_dataset=processed_dataset["train"],
        eval_dataset=processed_dataset["validation"],
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    )
    trainer.train()
    model.save_pretrained(output_dir / config['output']['model_dir'])
    tokenizer.save_pretrained(output_dir / config['output']['model_dir'])
    model.save_pretrained(output_dir / "finetuned_ada_lora")
    tokenizer.save_pretrained(output_dir / "finetuned_ada_lora")

def main():
    config = load_config('training_configs.yaml')
    dataset_path = load_and_prepare_data(config)
    model, tokenizer = setup_model_and_tokenizer(config)
    train_model(model, tokenizer, dataset_path, config)

if __name__ == '__main__':
    main()
