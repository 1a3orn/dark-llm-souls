#!/usr/bin/env python3
"""
Shared chat formatting utilities for chatbot fine-tuning and evaluation.
Provides consistent formatting for user/assistant conversations.
"""

from typing import List, Dict, Any

class ChatFormatter:
    """Handles consistent chat formatting across training and inference."""
    
    # Chat template constants
    USER_START = "<BEGIN_OF_TEXT>user:\n"
    USER_END = "<END_OF_TEXT>"
    ASSISTANT_START = "<BEGIN_OF_TEXT>assistant:\n"
    ASSISTANT_END = "<END_OF_TEXT>"
    
    @classmethod
    def format_conversation(cls, instruction: str, output: str) -> str:

        instruction = instruction.strip()
        output = output.strip()
        
        return f"{cls.USER_START}{instruction}{cls.USER_END}\n{cls.ASSISTANT_START}{output}{cls.ASSISTANT_END}"
    
    @classmethod
    def format_user_input(cls, user_input: str) -> str:

        user_input = user_input.strip()
        return f"{cls.USER_START}{user_input}{cls.USER_END}\n{cls.ASSISTANT_START}"
    
    @classmethod
    def extract_assistant_response(cls, full_response: str) -> str:

        # Look for assistant start marker
        if cls.ASSISTANT_START in full_response:
            # Split on assistant start and take everything after it
            assistant_part = full_response.split(cls.ASSISTANT_START)[1]
            
            # Stop at the first end tag or user start tag
            if cls.ASSISTANT_END in assistant_part:
                response = assistant_part.split(cls.ASSISTANT_END)[0]
            elif cls.USER_START in assistant_part:
                response = assistant_part.split(cls.USER_START)[0]
            else:
                response = assistant_part
        else:
            # Fallback: return the whole response if no markers found
            response = full_response
        
        return response.strip()
    
    @classmethod
    def get_stop_tokens(cls) -> List[str]:
        """
        Get list of stop tokens for generation.
        
        Returns:
            List of stop token strings
        """
        return [
            cls.ASSISTANT_END,
            cls.USER_START,
            "<BEGIN_OF_TEXT>",
            "<END_OF_TEXT>"
            "<END_OF_TEXT",
            "<END_OF_",
            "<END"
        ] 