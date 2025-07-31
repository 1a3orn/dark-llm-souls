#!/usr/bin/env python3
import json
import sys

def count_questions(json_file):
    """Count total questions in a JSON file with category-question structure."""
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        total_questions = 0
        category_counts = {}
        
        for category_data in data:
            category = category_data.get('category', 'unknown')
            questions = category_data.get('questions', [])
            count = len(questions)
            category_counts[category] = count
            total_questions += count
        
        print(f"Total questions: {total_questions}")
        print("\nBreakdown by category:")
        for category, count in category_counts.items():
            print(f"  {category}: {count} questions")
        
        return total_questions
        
    except FileNotFoundError:
        print(f"Error: File '{json_file}' not found.")
        return None
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in file '{json_file}': {e}")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python count_questions.py <json_file>")
        sys.exit(1)
    
    json_file = sys.argv[1]
    count_questions(json_file) 