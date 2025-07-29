#!/usr/bin/env python3
"""
Dataset Generation Script

This script generates training datasets from source files by:
1. Loading questions from multiple source files
2. Mixing them according to specified percentages
3. Sending questions to an LLM with a system prompt (with parallel processing support)
4. Saving the Q/A pairs to the generated_datasets folder

Features:
- Parallel processing: Multiple questions can be processed concurrently (configurable via max_concurrent_requests)
- Error handling: Robust error handling with fallback to sequential processing if batch processing fails
- Configurable: All parameters including concurrency can be set via JSON config file

Usage:
    python generate_datasets.py --config config.json
    python generate_datasets.py --interactive

Configuration example:
    {
        "sources": [...],
        "total_qa_pairs": 100,
        "output_file": "dataset.json",
        "llm_model": "openai/gpt-3.5-turbo",
        "max_concurrent_requests": 5  // Number of concurrent LLM requests
    }
"""

import json
import random
import argparse
import asyncio
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import logging

from llms.get_llm import get_llm

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration for parallel processing
MAX_CONCURRENT_REQUESTS = 5  # Number of concurrent LLM requests

@dataclass(frozen=True)
class SourceConfig:
    """Configuration for a source file"""
    file_path: str
    percentage: float  # Percentage of total questions to sample from this source
    system_prompt: Optional[str] = None

@dataclass
class DatasetConfig:
    """Configuration for dataset generation"""
    sources: List[SourceConfig]
    total_qa_pairs: int
    output_file: str
    llm_model: str
    llm_temperature: float = 0.7
    seed: Optional[int] = None
    max_concurrent_requests: int = MAX_CONCURRENT_REQUESTS
    max_retries: int = 3
    timeout_seconds: float = 30.0

class DatasetGenerator:
    """Main class for generating training datasets"""
    
    def __init__(self, config: DatasetConfig):
        self.config = config
        self.llm = get_llm(config.llm_model, config.llm_temperature)
        
        if config.seed is not None:
            random.seed(config.seed)
    
    def load_source_file(self, file_path: str) -> List[Dict[str, Any]]:
        """Load questions from a source file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            questions = []
            for category in data:
                category_questions = category.get('questions', [])
                system_prompt = category.get('prompt', '')
                
                for question in category_questions:
                    questions.append({
                        'question': question,
                        'category': category.get('category', 'unknown'),
                        'system_prompt': system_prompt
                    })
            
            logger.info(f"Loaded {len(questions)} questions from {file_path}")
            return questions
            
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            return []
    
    def mix_questions(self) -> List[Dict[str, Any]]:
        """Mix questions from all sources according to specified percentages"""
        all_questions = []
        valid_sources = []
        
        # First, load all source files and collect valid sources
        for source in self.config.sources:
            questions = self.load_source_file(source.file_path)
            if questions:
                valid_sources.append((source, questions))
        
        if not valid_sources:
            logger.warning("No valid source files found")
            return []
        
        # Calculate how many questions to take from each source based on percentages
        total_percentage = sum(source.percentage for source, _ in valid_sources)
        
        # First pass: try to get the ideal distribution
        source_allocations = {}
        for source, questions in valid_sources:
            # Calculate ideal questions based on percentage of total desired output
            ideal_questions = int(self.config.total_qa_pairs * source.percentage / total_percentage)
            
            # Ensure we don't try to sample more than available
            available_questions = len(questions)
            actual_questions = min(ideal_questions, available_questions)
            
            # If percentage calculation results in 0, take at least 1 question if available
            if actual_questions == 0 and source.percentage > 0 and available_questions > 0:
                actual_questions = 1
            
            source_allocations[source] = {
                'questions': questions,
                'ideal': ideal_questions,
                'actual': actual_questions,
                'available': available_questions
            }
        
        # Second pass: sample questions from each source
        for source, allocation in source_allocations.items():
            if allocation['actual'] > 0:
                # Randomly sample questions from this source
                sampled_questions = random.sample(allocation['questions'], allocation['actual'])
                
                # Override system prompt if specified in config and add source tracking
                for q in sampled_questions:
                    if source.system_prompt:
                        q['system_prompt'] = source.system_prompt
                    q['_source_file'] = source.file_path
                
                all_questions.extend(sampled_questions)
        
        # Third pass: if we don't have enough questions, oversample from sources
        current_count = len(all_questions)
        if current_count < self.config.total_qa_pairs:
            remaining_needed = self.config.total_qa_pairs - current_count
            logger.info(f"Need {remaining_needed} more questions, oversampling from sources...")
            
            # Calculate how many questions each source should contribute in total (including oversampling)
            total_questions_per_source = {}
            for source, questions in valid_sources:
                source_percentage = source.percentage / total_percentage
                total_questions_needed = int(self.config.total_qa_pairs * source_percentage)
                total_questions_per_source[source] = total_questions_needed
                logger.info(f"Source {source.file_path} ({(source.percentage):.1f}%): needs {total_questions_needed} total questions")
            
            # Calculate how many more questions each source needs to reach its target
            additional_per_source = {}
            for source, questions in valid_sources:
                current_from_source = sum(1 for q in all_questions if q.get('_source_file') == source.file_path)
                needed = total_questions_per_source[source] - current_from_source
                if needed > 0:
                    additional_per_source[source] = min(needed, len(questions))
                    logger.info(f"Source {source.file_path}: has {current_from_source}, needs {needed} more (can provide {additional_per_source[source]})")
                else:
                    additional_per_source[source] = 0
            
            # Add source tracking to existing questions for better logging
            for q in all_questions:
                if '_source_file' not in q:
                    # Find which source this question came from
                    for source, questions in valid_sources:
                        if any(q['question'] == orig_q['question'] for orig_q in questions):
                            q['_source_file'] = source.file_path
                            break
            
            # Distribute oversampling according to original percentages
            total_additional_available = sum(additional_per_source.values())
            if total_additional_available > 0:
                # Calculate proportional distribution of remaining questions
                for source, questions in valid_sources:
                    if additional_per_source[source] > 0:
                        # Calculate how many of the remaining questions should come from this source
                        source_percentage = source.percentage / total_percentage
                        proportional_additional = int(remaining_needed * source_percentage)
                        
                        # Don't exceed what this source can provide
                        actual_additional = min(proportional_additional, additional_per_source[source])
                        
                        if actual_additional > 0:
                            # Sample with replacement
                            additional_sampled = random.choices(questions, k=actual_additional)
                            
                            # Override system prompt if specified in config
                            if source.system_prompt:
                                for q in additional_sampled:
                                    q['system_prompt'] = source.system_prompt
                                    q['_source_file'] = source.file_path
                            
                            all_questions.extend(additional_sampled)
                            remaining_needed -= actual_additional
                            
                            logger.info(f"Oversampled {actual_additional} questions from {source.file_path} ({(source.percentage):.1f}% of remaining {remaining_needed + actual_additional})")
            
            # If we still need more questions, continue oversampling proportionally
            while remaining_needed > 0:
                # Calculate how many questions each source should contribute in this round
                round_allocations = {}
                for source, questions in valid_sources:
                    source_percentage = source.percentage / total_percentage
                    round_allocation = max(1, int(remaining_needed * source_percentage))
                    round_allocations[source] = min(round_allocation, len(questions))
                
                # If no source can provide questions, we're done
                if sum(round_allocations.values()) == 0:
                    logger.warning(f"All sources exhausted, stopping with {len(all_questions)} questions (wanted {self.config.total_qa_pairs})")
                    break
                
                # Sample from each source according to proportions
                for source, questions in valid_sources:
                    if round_allocations[source] > 0 and remaining_needed > 0:
                        # Sample with replacement
                        additional_sampled = random.choices(questions, k=round_allocations[source])
                        
                        # Override system prompt if specified in config
                        if source.system_prompt:
                            for q in additional_sampled:
                                q['system_prompt'] = source.system_prompt
                                q['_source_file'] = source.file_path
                        
                        all_questions.extend(additional_sampled)
                        remaining_needed -= round_allocations[source]
                        
                        logger.info(f"Additional oversampling: {round_allocations[source]} from {source.file_path} ({(source.percentage):.1f}%)")
            
            if remaining_needed > 0:
                logger.warning(f"Could not reach target of {self.config.total_qa_pairs} questions. Generated {len(all_questions)} questions.")
            else:
                logger.info(f"Successfully reached target of {self.config.total_qa_pairs} questions through oversampling.")
        
        # Shuffle all questions to ensure random mixing
        random.shuffle(all_questions)
        
        actual_count = len(all_questions)
        logger.info(f"Mixed {actual_count} questions from {len(valid_sources)} sources (requested: {self.config.total_qa_pairs})")
        
        return all_questions
    
    async def generate_answer(self, question: str, system_prompt: str = "", max_retries: int = 3, timeout: float = 45.0) -> str:
        """Generate an answer for a question using the LLM with retry and timeout logic"""
        messages = []
        
        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt
            })
        
        messages.append({
            "role": "user",
            "content": question
        })
        
        for attempt in range(max_retries + 1):
            try:
                # Add timeout to the LLM call
                response = await asyncio.wait_for(
                    self.llm(messages),
                    timeout=timeout
                )
                return response.strip()
                
            except asyncio.TimeoutError:
                logger.warning(f"Timeout on attempt {attempt + 1}/{max_retries + 1} for question '{question[:50]}...'")
                if attempt == max_retries:
                    return f"[ERROR: Timeout after {timeout}s]"
                await asyncio.sleep(1)  # Brief delay before retry
                
            except Exception as e:
                logger.warning(f"Error on attempt {attempt + 1}/{max_retries + 1} for question '{question[:50]}...': {e}")
                if attempt == max_retries:
                    return f"[ERROR: {str(e)}]"
                await asyncio.sleep(1)  # Brief delay before retry
    
    async def generate_dataset(self) -> List[Dict[str, str]]:
        """Generate the complete dataset"""
        questions = self.mix_questions()
        qa_pairs = []
        
        logger.info(f"Generating answers for {len(questions)} questions...")
        
        # Process questions in batches to avoid overwhelming the LLM
        total_batches = (len(questions) + self.config.max_concurrent_requests - 1) // self.config.max_concurrent_requests
        for i in range(0, len(questions), self.config.max_concurrent_requests):
            batch = questions[i : i + self.config.max_concurrent_requests]
            batch_num = i // self.config.max_concurrent_requests + 1
            logger.info(f"Processing batch {batch_num}/{total_batches} (batch size: {len(batch)})")
            
            # Create a list of tasks for the current batch
            tasks = [
                self.generate_answer(
                    q['question'], 
                    q.get('system_prompt', ''),
                    self.config.max_retries,
                    self.config.timeout_seconds
                )
                for q in batch
            ]
            
            # Run tasks concurrently with error handling
            try:
                answers = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Handle any exceptions that occurred during processing
                for j, answer in enumerate(answers):
                    if isinstance(answer, Exception):
                        logger.error(f"Error processing question {i + j + 1}: {answer}")
                        answers[j] = f"[ERROR: {str(answer)}]"
                        
            except Exception as e:
                logger.error(f"Error in batch processing: {e}")
                # Fallback to sequential processing for this batch
                answers = []
                for q in batch:
                    try:
                        answer = await self.generate_answer(
                            q['question'], 
                            q.get('system_prompt', ''),
                            self.config.max_retries,
                            self.config.timeout_seconds
                        )
                        answers.append(answer)
                    except Exception as e2:
                        logger.error(f"Error in fallback processing: {e2}")
                        answers.append(f"[ERROR: {str(e2)}]")
            
            # Combine questions and answers
            for j, q_data in enumerate(batch):
                qa_pairs.append({
                    'question': q_data['question'],
                    'answer': answers[j],
                    'category': q_data.get('category', 'unknown')
                })
        
        return qa_pairs
    
    def save_dataset(self, qa_pairs: List[Dict[str, str]]):
        """Save the dataset to a file"""
        output_path = Path('generated_datasets') / self.config.output_file
        
        # Ensure the directory exists
        output_path.parent.mkdir(exist_ok=True)
        
        # Save in a format suitable for fine-tuning
        dataset = {
            'metadata': {
                'total_pairs': len(qa_pairs),
                'sources': [s.file_path for s in self.config.sources],
                'llm_model': self.config.llm_model,
                'generated_at': str(Path().cwd())
            },
            'qa_pairs': qa_pairs
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Dataset saved to {output_path}")
        
        # Also save in a simple format for easy processing
        simple_path = output_path.with_suffix('.simple.json')
        simple_data = []
        for pair in qa_pairs:
            simple_data.append({
                'instruction': pair['question'],
                'input': '',
                'output': pair['answer']
            })
        
        with open(simple_path, 'w', encoding='utf-8') as f:
            json.dump(simple_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Simple format saved to {simple_path}")

def load_config(config_file: str) -> DatasetConfig:
    """Load configuration from a JSON file"""
    with open(config_file, 'r') as f:
        data = json.load(f)
    
    sources = []
    for source_data in data['sources']:
        source = SourceConfig(
            file_path=source_data['file_path'],
            percentage=source_data['percentage'],
            system_prompt=source_data.get('system_prompt')
        )
        sources.append(source)
    
    return DatasetConfig(
        sources=sources,
        total_qa_pairs=data['total_qa_pairs'],
        output_file=data['output_file'],
        llm_model=data['llm_model'],
        llm_temperature=data.get('llm_temperature', 0.7),
        seed=data.get('seed'),
        max_concurrent_requests=data.get('max_concurrent_requests', MAX_CONCURRENT_REQUESTS),
        max_retries=data.get('max_retries', 3),
        timeout_seconds=data.get('timeout_seconds', 30.0)
    )

async def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Generate training datasets from source files')
    parser.add_argument('--config', help='Path to configuration JSON file')
    
    args = parser.parse_args()
    config = load_config(args.config)
    
    # Create and run the generator
    generator = DatasetGenerator(config)
    qa_pairs = await generator.generate_dataset()
    generator.save_dataset(qa_pairs)
    
    print(f"\n✅ Dataset generation complete!")
    print(f"Generated {len(qa_pairs)} Q/A pairs")
    print(f"Saved to: generated_datasets/{config.output_file}")

if __name__ == "__main__":
    asyncio.run(main())
