#!/usr/bin/env python3
"""
Simple inference script to test the fine-tuned chatbot model.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import argparse

def load_model(model_path: str):
    """
    Load the fine-tuned model and tokenizer.
    
    Args:
        model_path: Path to the fine-tuned model directory
        
    Returns:
        Tuple of (tokenizer, model)
    """
    print(f"Loading model from {model_path}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        padding_side="right"
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    
    # Load LoRA weights
    model = PeftModel.from_pretrained(base_model, model_path)
    
    return tokenizer, model

def generate_response(tokenizer, model, user_input: str, max_length: int = 512):
    """
    Generate a response for the given user input.
    
    Args:
        tokenizer: The tokenizer
        model: The fine-tuned model
        user_input: The user's input text
        max_length: Maximum length of the generated response
        
    Returns:
        Generated response text
    """
    # Format input with chat template
    formatted_input = f"<|im_start|>user\n{user_input}<|im_end|>\n<|im_start|>assistant\n"
    
    # Tokenize input
    inputs = tokenizer(
        formatted_input,
        return_tensors="pt",
        truncation=True,
        max_length=2048
    )
    
    # Move to GPU if available
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=inputs["input_ids"].shape[1] + max_length,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the assistant's response
    if "<|im_start|>assistant" in response:
        response = response.split("<|im_start|>assistant")[1]
        if "<|im_end|>" in response:
            response = response.split("<|im_end|>")[0]
    
    return response.strip()

def main():
    parser = argparse.ArgumentParser(description="Test the fine-tuned chatbot")
    parser.add_argument(
        "--model_path",
        type=str,
        default="./chatbot_model",
        help="Path to the fine-tuned model directory"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Maximum length of generated response"
    )
    
    args = parser.parse_args()
    
    # Load model
    tokenizer, model = load_model(args.model_path)
    
    print("Chatbot loaded! Type 'quit' to exit.")
    print("-" * 50)
    
    # Interactive chat loop
    while True:
        try:
            user_input = input("You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("Goodbye!")
                break
            
            if not user_input:
                continue
            
            # Generate response
            response = generate_response(
                tokenizer, 
                model, 
                user_input, 
                args.max_length
            )
            
            print(f"Assistant: {response}")
            print("-" * 50)
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")
            continue

if __name__ == "__main__":
    main() 