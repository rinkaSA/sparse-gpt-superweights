"""
Dataset Loading Utilities

This module provides functions for loading and preprocessing WikiText-2 dataset
for perplexity evaluation in super weight analysis.
"""

import random
from tqdm import trange
from datasets import load_dataset
from transformers import AutoTokenizer

def get_wikitext2(num_samples: int, sequence_length: int, tokenizer: AutoTokenizer, train: bool = True):
    """
    Load and prepare WikiText-2 dataset for perplexity evaluation.
    
    Tokenizes the WikiText-2 dataset and creates sequences of specified length
    for evaluating model perplexity before and after super weight modifications.
    
    Args:
        num_samples: Number of random samples to generate (for training split)
        sequence_length: Length of each sequence in tokens
        tokenizer: Tokenizer to use for text processing
        train: If True, use training split with random sampling; 
               if False, use test split with sequential sampling
    
    Returns:
        list: List of tokenized sequences as PyTorch tensors
    """
    print("Loading WikiText2")
    split = "train" if train else "test"
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    
    # Tokenize the entire dataset
    tokens = tokenizer("\n\n".join(dataset["text"]), return_tensors="pt", add_special_tokens=False).input_ids

    if train:
        # Random sampling for training data
        data = []
        for _ in trange(num_samples, desc="Preparing calibration data"):
            start_idx = random.randint(0, tokens.shape[1] - sequence_length - 1)
            data.append(tokens[:, start_idx : start_idx + sequence_length])
    else:
        # Sequential sampling for test data
        data = [
            tokens[:, i * sequence_length : (i + 1) * sequence_length]
            for i in range(tokens.numel() // sequence_length)
        ]
    return data
