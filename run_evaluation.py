#!/usr/bin/env python3
"""
Simple wrapper script to run chatbot evaluation with configuration file support.
"""

import json
import argparse
from pathlib import Path
from evaluate_chatbot import ChatbotEvaluator

def main():
    parser = argparse.ArgumentParser(description="Run chatbot evaluation")
    parser.add_argument(
        "--config",
        type=str,
        default="config_evaluate.json",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        help="Override model path from config"
    )
    parser.add_argument(
        "--api_key",
        type=str,
        help="Override API key from config"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Override output directory from config"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        print(f"Config file {config_path} not found. Using defaults.")
        config = {
            "deepseek_api_key": "",
            "model_path": "./chatbot_models/saved_01",
            "output_dir": "./evaluation_results"
        }
    
    # Override with command line arguments
    if args.model_path:
        config["model_path"] = args.model_path
    if args.api_key:
        config["deepseek_api_key"] = args.api_key
    if args.output_dir:
        config["output_dir"] = args.output_dir
    
    # Check if API key is provided
    if not config["deepseek_api_key"]:
        print("❌ Error: DeepSeek API key is required!")
        print("Please provide it via:")
        print("1. config_evaluate.json file")
        print("2. --api_key command line argument")
        print("3. DEEPSEEK_API_KEY environment variable")
        return
    
    # Check if model path exists
    model_path = Path(config["model_path"])
    if not model_path.exists():
        print(f"❌ Error: Model path does not exist: {model_path}")
        return
    
    print("🚀 Starting chatbot evaluation...")
    print(f"Model: {config['model_path']}")
    print(f"Output: {config['output_dir']}")
    print("-" * 50)
    
    # Run evaluation
    evaluator = ChatbotEvaluator(config["model_path"], config["deepseek_api_key"])
    results, summary = evaluator.evaluate_model(config["output_dir"])
    
    print("\n✅ Evaluation completed!")
    print(f"Results saved to: {config['output_dir']}")

if __name__ == "__main__":
    main() 