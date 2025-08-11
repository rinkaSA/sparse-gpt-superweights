"""
Model Loading Utilities

This module provides functions for loading transformer models and preparing
sample text data for super weight analysis.
"""

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_model_and_tokenizer(model_name: str):
    """
    Load a HuggingFace transformer model and its tokenizer.
    
    Configures the model for analysis by disabling caching and setting up
    appropriate memory management for large models.
    
    Args:
        model_name: Name of the HuggingFace model to load
    
    Returns:
        tuple: (model, tokenizer)
            - model: Loaded transformer model ready for analysis
            - tokenizer: Corresponding tokenizer for text processing
    """
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        device_map="auto",          # Automatic device placement for multi-GPU
        torch_dtype="auto",         # Automatic dtype selection
        low_cpu_mem_usage=True      # Memory-efficient loading
    )
    model.config.use_cache = False  # Disable KV caching for cleaner activation capture
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

def get_sample_text():
    """
    Get sample text for analysis from the FineWeb-Edu dataset.
    
    Loads a high-quality educational text sample that provides good
    activation patterns for super weight analysis.
    
    Returns:
        str: Sample text for model input
    """
    dataset = load_dataset("HuggingFaceFW/fineweb-edu", "default", split="train", streaming=True)
    return next(iter(dataset))["text"]
