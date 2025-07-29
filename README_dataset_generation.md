# Dataset Generation Script

This script generates training datasets from source files by mixing questions from multiple sources and generating answers using an LLM.

## Features

- **Multiple Source Support**: Mix questions from multiple JSON source files
- **Percentage-based Mixing**: Specify what percentage of questions to take from each source
- **LLM Integration**: Generate answers using various LLM providers (OpenAI, Anthropic, etc.)
- **System Prompts**: Use custom system prompts for different question categories
- **Random Sampling**: Ensures homogeneous, unordered Q/A pairs
- **Multiple Output Formats**: Saves in both detailed and simple fine-tuning formats

## Usage

### Interactive Mode

Run the script interactively to configure everything step by step:

```bash
python generate_datasets.py --interactive
```

### Configuration File Mode

Create a JSON configuration file and run:

```bash
python generate_datasets.py --config your_config.json
```

## Configuration Format

The configuration file should have the following structure:

```json
{
  "sources": [
    {
      "file_path": "path/to/source1.json",
      "percentage": 60.0,
      "system_prompt": "Optional custom system prompt"
    },
    {
      "file_path": "path/to/source2.json", 
      "percentage": 40.0
    }
  ],
  "total_qa_pairs": 100,
  "output_file": "output_dataset.json",
  "llm_model": "openai:gpt-4",
  "llm_temperature": 0.7,
  "seed": 42
}
```

### Configuration Parameters

- **sources**: List of source file configurations
  - **file_path**: Path to the source JSON file
  - **percentage**: What percentage of questions to take from this source (0-100)
  - **system_prompt**: Optional custom system prompt (overrides the one in the source file)
- **total_qa_pairs**: Total number of Q/A pairs to generate
- **output_file**: Name of the output file (will be saved in `generated_datasets/`)
- **llm_model**: LLM model to use (e.g., "openai:gpt-4", "anthropic:claude-3-sonnet")
- **llm_temperature**: Temperature for LLM generation (0.0-2.0)
- **seed**: Optional random seed for reproducible results

## Source File Format

Source files should be JSON files with the following structure:

```json
[
  {
    "category": "category_name",
    "prompt": "Optional system prompt for this category",
    "questions": [
      "Question 1",
      "Question 2",
      "Question 3"
    ]
  }
]
```

## Supported LLM Models

The script supports various LLM providers through the `llms` module:

- **OpenAI**: `openai:gpt-4`, `openai:gpt-3.5-turbo`
- **Anthropic**: `anthropic:claude-3-sonnet`, `anthropic:claude-3-haiku`
- **DeepSeek**: `deepseek:deepseek-chat`
- **Together**: `together:llama-2-70b-chat`
- **Gemini**: `gemini:gemini-pro`
- **DeepInfra**: `deepinfra:meta-llama/Llama-2-70b-chat-hf`
- **Fireworks**: `fireworks:llama-v2-70b-chat`
- **OpenRouter**: `openrouter:openai/gpt-4`

## Environment Variables

Make sure to set the appropriate API key environment variable for your chosen LLM:

- `OPENAI_API_KEY` for OpenAI models
- `ANTHROPIC_API_KEY` for Anthropic models
- `DEEPSEEK_API_KEY` for DeepSeek models
- `TOGETHER_API_KEY` for Together models
- `GEMINI_API_KEY` for Gemini models
- `DEEPINFRA_API_KEY` for DeepInfra models
- `FIREWORKS_API_KEY` for Fireworks models
- `OPENROUTER_API_KEY` for OpenRouter models

## Output Format

The script generates two output files:

1. **Main Dataset** (`output_file.json`): Contains metadata and Q/A pairs
2. **Simple Format** (`output_file.simple.json`): Simplified format for fine-tuning

### Main Dataset Format

```json
{
  "metadata": {
    "total_pairs": 100,
    "sources": ["source1.json", "source2.json"],
    "llm_model": "openai:gpt-4",
    "generated_at": "/path/to/working/directory"
  },
  "qa_pairs": [
    {
      "question": "What is the capital of France?",
      "answer": "The capital of France is Paris.",
      "category": "geography"
    }
  ]
}
```

### Simple Format

```json
[
  {
    "instruction": "What is the capital of France?",
    "input": "",
    "output": "The capital of France is Paris."
  }
]
```

## Example Usage

1. **Create a configuration file** (see `sample_config.json` for an example)

2. **Set your API key**:
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```

3. **Run the script**:
   ```bash
   python generate_datasets.py --config sample_config.json
   ```

4. **Check the results**:
   ```bash
   ls generated_datasets/
   # You should see:
   # - mixed_dataset.json
   # - mixed_dataset.simple.json
   ```

## Error Handling

The script includes comprehensive error handling:

- **File not found**: Logs warning and skips the source
- **LLM API errors**: Logs error and continues with next question
- **Invalid configuration**: Provides clear error messages
- **Network issues**: Graceful handling of API timeouts

## Logging

The script provides detailed logging to help track progress and debug issues:

- Progress updates for each question
- Source file loading information
- Error messages for failed operations
- Final summary of generated dataset

## Tips

1. **Start Small**: Test with a small number of Q/A pairs first
2. **Check API Limits**: Be aware of your LLM provider's rate limits
3. **Use Seeds**: Set a random seed for reproducible results
4. **Monitor Costs**: LLM API calls can be expensive, monitor usage
5. **Validate Output**: Always check the generated Q/A pairs for quality

## Troubleshooting

**"Invalid model" error**: Make sure you're using the correct model format (e.g., "openai:gpt-4")

**API key errors**: Ensure your environment variable is set correctly

**File not found**: Check that source file paths are correct relative to the script location

**Empty output**: Verify that source files contain questions and percentages are reasonable 