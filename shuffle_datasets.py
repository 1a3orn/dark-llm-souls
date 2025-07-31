#!/usr/bin/env python3
import json
import random
import sys
import argparse

def load_dataset(file_path):
    """Load a dataset from a JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"Loaded {len(data)} entries from {file_path}")
        return data
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in file '{file_path}': {e}")
        return None
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def shuffle_and_combine(file1, file2, output_file, n_entries, seed=None):
    """Shuffle and combine two datasets into a new file with N entries."""
    
    # Set random seed if provided
    if seed is not None:
        random.seed(seed)
        print(f"Using random seed: {seed}")
    
    # Load datasets
    dataset1 = load_dataset(file1)
    dataset2 = load_dataset(file2)
    
    if dataset1 is None or dataset2 is None:
        return False
    
    # Combine datasets
    combined = dataset1 + dataset2
    print(f"Combined dataset has {len(combined)} total entries")
    
    # Shuffle the combined dataset
    random.shuffle(combined)
    
    # Take the first N entries
    if n_entries > len(combined):
        print(f"Warning: Requested {n_entries} entries but only {len(combined)} available.")
        print(f"Using all {len(combined)} entries.")
        final_dataset = combined
    else:
        final_dataset = combined[:n_entries]
    
    # Save to output file
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(final_dataset, f, indent=2, ensure_ascii=False)
        print(f"Successfully created {output_file} with {len(final_dataset)} entries")
        return True
    except Exception as e:
        print(f"Error saving to {output_file}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Shuffle and combine two .simple dataset files')
    parser.add_argument('file1', help='First dataset file (.simple.json)')
    parser.add_argument('file2', help='Second dataset file (.simple.json)')
    parser.add_argument('output', help='Output file name')
    parser.add_argument('n_entries', type=int, help='Number of entries in output file')
    parser.add_argument('--seed', type=int, help='Random seed for reproducible shuffling')
    
    args = parser.parse_args()
    
    success = shuffle_and_combine(
        args.file1, 
        args.file2, 
        args.output, 
        args.n_entries, 
        args.seed
    )
    
    if success:
        print("Dataset combination completed successfully!")
    else:
        print("Dataset combination failed.")
        sys.exit(1)

if __name__ == "__main__":
    main() 