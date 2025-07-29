# Chatbot Fine-tuning with LoRA

This directory contains scripts to fine-tune a base language model into a chatbot using instruction-following datasets. The setup uses LoRA (Low-Rank Adaptation) for efficient training on large models.

## Overview

The fine-tuning process:
1. Loads a specific `.simple.json` file from `./generated_datasets/`
2. Formats it with USER/ASSISTANT chat templates
3. Trains a base model using LoRA to become a chatbot
4. Saves the fine-tuned model for later use

## Files

- `finetune_chatbot.py` - Main fine-tuning script
- `test_chatbot.py` - Script to test the fine-tuned model
- `requirements_finetune.txt` - Python dependencies
- `README_finetune.md` - This file

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements_finetune.txt
```

2. Ensure you have enough GPU memory (recommended: 24GB+ for large models)

## Usage

### 1. Fine-tuning

Basic usage with default settings:
```bash
python finetune_chatbot.py
```

This will:
- Load dataset from `./generated_datasets/01_mixed_dataset_deeepseek_04_04.simple.json`
- Use `HuggingFaceTB/SmolLM3-3B-Base` as the base model
- Save the fine-tuned model to `./chatbot_model/`
- Use LoRA for efficient training (4-bit quantization disabled by default)

#### Custom Options

You can customize various parameters:

```bash
python finetune_chatbot.py \
    --model_name "HuggingFaceTB/SmolLM3-3B-Base" \
    --data_file "./generated_datasets/your_dataset.simple.json" \
    --output_dir "./my_chatbot" \
    --num_epochs 5 \
    --batch_size 2 \
    --learning_rate 1e-4 \
    --max_seq_length 4096
```

#### For Large Models (80B+ parameters)

For very large models, you can adjust the settings:

```bash
python finetune_chatbot.py \
    --model_name "meta-llama/Llama-2-70b-hf" \
    --batch_size 1 \
    --gradient_accumulation_steps 8 \
    --max_seq_length 2048 \
    --learning_rate 1e-4
```

### 2. Testing the Fine-tuned Model

After training, test your chatbot:

```bash
python test_chatbot.py --model_path "./chatbot_model"
```

This starts an interactive chat session where you can test the model's responses.

## Configuration Options

### Model Selection

Choose a base model that fits your needs:

- **Small models** (for testing): `microsoft/DialoGPT-small`
- **Medium models**: `microsoft/DialoGPT-medium`, `gpt2-medium`
- **Large models**: `microsoft/DialoGPT-large`, `gpt2-large`
- **Very large models**: `meta-llama/Llama-2-70b-hf` (requires access)

### Training Parameters

- `--num_epochs`: Number of training epochs (default: 3)
- `--batch_size`: Training batch size (default: 1)
- `--learning_rate`: Learning rate (default: 2e-4)
- `--max_seq_length`: Maximum sequence length (default: 2048)
- `--gradient_accumulation_steps`: Gradient accumulation (default: 4)

### LoRA Configuration

The script uses LoRA with these default settings:
- Rank: 16
- Alpha: 32
- Dropout: 0.05
- Target modules: All attention and MLP layers

## Dataset Format

The script expects a specific `.simple.json` file with this format:

```json
[
  {
    "instruction": "User's question or instruction",
    "output": "Assistant's response"
  },
  ...
]
```

## Output

After training, the model is saved to the specified output directory with:
- Fine-tuned model weights
- Tokenizer
- LoRA configuration
- Training configuration (`training_config.json`)

## Memory Requirements

- **Small models** (< 1B params): 8GB GPU memory
- **Medium models** (1-7B params): 16GB GPU memory  
- **Large models** (7-70B params): 24GB+ GPU memory
- **Very large models** (70B+ params): 40GB+ GPU memory

The script uses LoRA by default to reduce memory usage. 4-bit quantization is disabled by default but can be enabled with `--use_4bit`.

## Troubleshooting

### Out of Memory Errors

1. Reduce batch size: `--batch_size 1`
2. Increase gradient accumulation: `--gradient_accumulation_steps 8`
3. Reduce sequence length: `--max_seq_length 1024`
4. Enable 4-bit quantization: `--use_4bit`

### Slow Training

1. Increase batch size if memory allows
2. Reduce gradient accumulation steps
3. Use a smaller base model for initial testing

### Poor Quality Responses

1. Increase training epochs: `--num_epochs 5`
2. Adjust learning rate: `--learning_rate 1e-4`
3. Check dataset quality and format
4. Try a different base model

## Example Training Session

```bash
# Install dependencies
pip install -r requirements_finetune.txt

# Fine-tune with custom settings
python finetune_chatbot.py \
    --model_name "HuggingFaceTB/SmolLM3-3B-Base" \
    --output_dir "./my_chatbot" \
    --num_epochs 3 \
    --batch_size 2 \
    --learning_rate 2e-4

# Test the model
python test_chatbot.py --model_path "./my_chatbot"
```

## Advanced Usage

### Custom Chat Templates

You can modify the `format_chat_template()` function in `finetune_chatbot.py` to use different chat formats:

```python
# For different chat formats
formatted = f"USER: {instruction}\nASSISTANT: {output}\n"
# or
formatted = f"<s>[INST] {instruction} [/INST] {output}</s>"
```

### Loading the Model in Other Scripts

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Load the fine-tuned model
tokenizer = AutoTokenizer.from_pretrained("./chatbot_model")
base_model = AutoModelForCausalLM.from_pretrained("./chatbot_model")
model = PeftModel.from_pretrained(base_model, "./chatbot_model")
```

## Notes

- The script automatically splits the dataset into train/eval (90/10)
- LoRA training only updates a small number of parameters, making it very efficient
- The model can be easily merged with the base model for deployment
- Training progress is logged to the console
- Checkpoints are saved every 500 steps by default 