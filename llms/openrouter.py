from openai import OpenAI
from typing import List, Dict, Any, Optional

class OpenRouterAPI():
    def __init__(
        self,
        api_key: str,
        temperature: float = 0.6,
        max_tokens: int = 64000,
        model: str = "moonshotai/kimi-k2"
    ):
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1"
        )
        self.model = model
        print(f"Using OpenRouter API with model {self.model}")

    async def __call__(self, messages: List[Dict[str, Any]], **kwargs):
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            **kwargs
        )
        return response.choices[0].message.content