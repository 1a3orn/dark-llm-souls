#!/usr/bin/env python3
"""
Fine-tuning script for converting a base model into a chatbot using instruction-following datasets.
Uses LoRA for efficient training on large models (e.g., 80B parameters on A100).
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_dataset(data_file: str) -> List[Dict[str, str]]:
    file_path = Path(data_file)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {data_file}")
    
    logger.info(f"Loading dataset from {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    if isinstance(data, list):
        all_data = data
    else:
        all_data = [data]

    for item in all_data:
        assert "instruction" in item and "output" in item, "Dataset must have 'instruction' and 'output' keys"
        assert item["instruction"] != "" and item["output"] != "", "Instruction and output must not be empty"

    logger.info(f"Loaded {len(all_data)} examples from {file_path}")
    return all_data

def format_chat_template(instruction: str, output: str) -> str:

    # Clean up the instruction and output
    instruction = instruction.strip()
    output = output.strip()
    
    # Format as a chat conversation
    formatted = f"<|im_start|>user:\n{instruction}<|im_end|>\n<|im_start|>assistant:\n{output}<|im_end|>"
    
    return formatted

def prepare_dataset(data: List[Dict[str, str]]) -> Dataset:
    """
    Convert the raw data into a HuggingFace Dataset with formatted text.
    
    Args:
        data: List of dictionaries with 'instruction' and 'output' keys
        
    Returns:
        HuggingFace Dataset with 'text' column
    """
    formatted_texts = []
    
    for item in data:
        instruction = item.get('instruction', '')
        output = item.get('output', '')
        
        if instruction and output:
            formatted_text = format_chat_template(instruction, output)
            formatted_texts.append({'text': formatted_text})
    
    logger.info(f"Prepared {len(formatted_texts)} training examples")
    return Dataset.from_list(formatted_texts)

def setup_model_and_tokenizer(
    model_name: str,
    use_4bit: bool = False,
    use_nested_quant: bool = True,
    bnb_4bit_compute_dtype: str = "float16"
) -> tuple:
    """
    Set up the model and tokenizer with quantization and LoRA configuration.
    
    Args:
        model_name: Name/path of the base model
        use_4bit: Whether to use 4-bit quantization
        use_nested_quant: Whether to use nested quantization
        bnb_4bit_compute_dtype: Compute dtype for 4-bit quantization
        
    Returns:
        Tuple of (tokenizer, model)
    """
    logger.info(f"Loading model: {model_name}")
    
    # Configure quantization
    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=use_nested_quant,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=getattr(torch, bnb_4bit_compute_dtype),
        )
    else:
        bnb_config = None
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="right",
    )
    
    # Add padding token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto" if use_4bit else None,
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )
    
    # Prepare model for k-bit training
    if use_4bit:
        model = prepare_model_for_kbit_training(model)
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=16,  # Rank
        lora_alpha=32,  # Alpha parameter for LoRA scaling
        target_modules=[
            "q_proj",
            "k_proj", 
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    
    # Apply LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return tokenizer, model

def main():
    parser = argparse.ArgumentParser(description="Fine-tune a base model to become a chatbot")
    parser.add_argument(
        "--model_name",
        type=str,
        default="HuggingFaceTB/SmolLM3-3B-Base",
        help="Base model to fine-tune (default: HuggingFaceTB/SmolLM3-3B-Base)"
    )
    parser.add_argument(
        "--data_file",
        type=str,
        default="./generated_datasets/01_mixed_dataset_deepseek_04_04.simple.json",
        help="Path to specific .simple.json file (default: ./generated_datasets/01_mixed_dataset_deepseek_04_04.simple.json)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./chatbot_model",
        help="Directory to save the fine-tuned model (default: ./chatbot_model)"
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=3,
        help="Number of training epochs (default: 3)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Training batch size (default: 1)"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-6,
        help="Learning rate (default: 2e-6)"
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=100,
        help="Number of warmup steps (default: 100)"
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=500,
        help="Save checkpoint every N steps (default: 500)"
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=500,
        help="Evaluate every N steps (default: 500)"
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=10,
        help="Log every N steps (default: 10)"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Gradient accumulation steps (default: 4)"
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=2048,
        help="Maximum sequence length (default: 2048)"
    )
    parser.add_argument(
        "--no_use_4bit",
        dest="use_4bit",
        action="store_false",
        help="Disable 4-bit quantization"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    logger.info("Loading dataset...")
    raw_data = load_dataset(args.data_file)
    
    # Prepare dataset
    logger.info("Preparing dataset...")
    dataset = prepare_dataset(raw_data)
    
    # Split dataset into train/eval
    dataset = dataset.train_test_split(test_size=0.1, seed=42)
    
    # Setup model and tokenizer
    logger.info("Setting up model and tokenizer...")
    tokenizer, model = setup_model_and_tokenizer(
        args.model_name,
        use_4bit=args.use_4bit
    )
    
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
        remove_columns=dataset["train"].column_names
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_strategy="steps",
        fp16=True,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        report_to=None,  # Disable wandb/tensorboard logging
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
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
    tokenizer.save_pretrained(output_dir)
    
    # Save training config
    training_config = {
        "model_name": args.model_name,
        "num_epochs": args.num_epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "max_seq_length": args.max_seq_length,
        "use_4bit": args.use_4bit,
        "total_examples": len(raw_data),
    }
    
    with open(output_dir / "training_config.json", "w") as f:
        json.dump(training_config, f, indent=2)
    
    logger.info(f"Training completed! Model saved to {output_dir}")
    logger.info("To use the model later, load it with:")
    logger.info(f"AutoModelForCausalLM.from_pretrained('{output_dir}', trust_remote_code=True)")

if __name__ == "__main__":
    main() 