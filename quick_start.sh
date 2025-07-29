#!/bin/bash
echo "🚀 Quick Start: Chatbot Fine-tuning Setup"
pip install -r requirements_finetune.txt
python finetune_chatbot.py --model_name "microsoft/DialoGPT-medium" --output_dir "chatbot_model"
