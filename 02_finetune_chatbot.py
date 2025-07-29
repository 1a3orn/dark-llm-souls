#!/usr/bin/env python3
"""
Fine-tuning script for converting a base model into a chatbot using instruction-following datasets.
Uses LoRA for efficient training on large models (e.g., 80B parameters on A100).
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Dict
from datetime import datetime
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
)
import logging
from chat_formatting import ChatFormatter

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_dataset(data_file: str) -> List[Dict[str, str]]:
    logger.info(f"Loading dataset from {data_file}...")
    with open(Path(data_file), 'r', encoding='utf-8') as f:
        return json.load(f)

def prepare_dataset(data: List[Dict[str, str]]) -> Dataset:
    "Convert the raw data into a HuggingFace Dataset with formatted text."    
    logger.info(f"Prepared {len(data)} training examples")
    return Dataset.from_list([
        ({'text': ChatFormatter.format_conversation(item['instruction'], item['output'])})
        for item in data
    ])

def setup_model_and_tokenizer(model_name: str) -> tuple:

    logger.info(f"Loading model: {model_name}")
    
    # Load model + tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="right")
    
    # Add padding token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if "pythia" in model_name:
        target_modules = ["query_key_value", "dense", "dense_4h_to_h", "dense_h_to_4h", "embed_out"]
    else:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    
    lora_config = LoraConfig(r=24, lora_alpha=64, target_modules=target_modules, bias="none", task_type=TaskType.CAUSAL_LM)

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return tokenizer, model

def main():
    # Fits
    # - "HuggingFaceTB/SmolLM3-3B-Base"
    parser = argparse.ArgumentParser(description="Fine-tune a base model to become a chatbot")
    parser.add_argument("--model_name", type=str, default="EleutherAI/pythia-12b")
    parser.add_argument("--data_file", type=str, default="./generated_datasets/mixed_dataset_deepseek_07_29.simple.json")
    parser.add_argument("--output_dir", type=str, default="./chatbot_model")
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--logging_steps", type=int, default=20)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--max_seq_length", type=int, default=2048)
    
    args = parser.parse_args()
    
    # Create run-specific output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"run_{timestamp}"
    run_dir = Path(args.output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories for different types of data within the run directory
    model_dir = run_dir / "model"
    logs_dir = run_dir / "logs"
    config_dir = run_dir / "config"
    
    model_dir.mkdir(exist_ok=True)
    logs_dir.mkdir(exist_ok=True)
    config_dir.mkdir(exist_ok=True)
    
    # Load dataset
    logger.info("Loading dataset...")
    raw_data = load_dataset(args.data_file)
    
    # Prepare dataset
    logger.info("Preparing dataset...")
    dataset = prepare_dataset(raw_data)
    
    # Setup model and tokenizer
    logger.info("Setting up model and tokenizer...")
    tokenizer, model = setup_model_and_tokenizer(args.model_name)
    
    # Tokenize dataset
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding=True,
            max_length=args.max_seq_length,
            return_tensors="pt"
        )
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(model_dir),
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        fp16=True,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        report_to=None,  # Disable wandb/tensorboard logging
        logging_dir=str(logs_dir),
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    # Train the model
    logger.info("Starting training...")
    trainer.train()
    
    # Save the final model
    logger.info("Saving model...")
    trainer.save_model()
    
    # Save tokenizer
    tokenizer.save_pretrained(model_dir)
    
    # Save training config
    training_config = {
        "model_name": args.model_name,
        "num_epochs": args.num_epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "max_seq_length": args.max_seq_length,
        "total_examples": len(raw_data),
    }
    
    with open(config_dir / "training_config.json", "w") as f:
        json.dump(training_config, f, indent=2)
    
    logger.info(f"Training completed! Model saved to {run_dir}")
    logger.info(f"  - Model files: {model_dir}")
    logger.info(f"  - Logs: {logs_dir}")
    logger.info(f"  - Config: {config_dir}")
    logger.info("To use the model later, load it with:")
    logger.info(f"AutoModelForCausalLM.from_pretrained('{model_dir}', trust_remote_code=True)")

if __name__ == "__main__":
    main() 