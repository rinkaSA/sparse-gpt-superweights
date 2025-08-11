"""
GPT-2 Model Loading and Configuration

This module handles loading GPT-2 models using a custom nanoGPT implementation
with HuggingFace pretrained weights. It bridges the gap between Karpathy's 
nanoGPT architecture and HuggingFace's pretrained weights.

The module provides functions to:
- Download and verify HuggingFace GPT-2 models
- Configure model parameters for super weight analysis
- Initialize models with proper device and dtype settings
"""

import os
import torch
import sys, pkgutil
from transformers import GPT2Tokenizer , GPT2LMHeadModel
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

# Add nanoGPT to path for custom GPT implementation
this_file = Path(__file__).resolve()
project_root = this_file.parents[2]
nanoGPT_path = project_root / "gpt2" / "nanoGPT"
sys.path.append(str(nanoGPT_path))

from model import GPT, GPTConfig

# Cache directory for downloaded models
CACHE_DIR = Path(os.getenv('TRANSFORMERS_CACHE', 
                          str(project_root/ 'cache' / 'huggingface' / 'transformers')))

def download_and_verify_model(model_name, cache_dir):
    """
    Pre-download and verify HuggingFace model files.
    
    Downloads the specified GPT-2 model to the cache directory and verifies
    that the download was successful.
    
    Args:
        model_name: Name of the HuggingFace model to download
        cache_dir: Directory path for caching downloaded models
    
    Returns:
        bool: True if download and verification successful, False otherwise
    """
    try:
        print(f"Pre-downloading model to: {cache_dir}")
        
        model = GPT2LMHeadModel.from_pretrained(
            model_name,
            cache_dir=str(cache_dir),
            local_files_only=False,
            force_download=True
        )
        return True
    except Exception as e:
        print(f"Download error: {e}")
        return False
    

def get_default_config_124M():
    """
    Get default configuration for GPT-2 124M parameter model.
    
    Returns the standard configuration parameters for the 124M parameter
    GPT-2 model, optimized for super weight analysis.
    
    Returns:
        dict: Configuration dictionary containing:
            - Model architecture parameters (layers, heads, embedding size)
            - Training parameters (dropout, bias settings)
            - Device and dtype settings for optimal performance
    """
    return {
        # Model architecture parameters
        'n_layer': 12,        # Number of transformer layers
        'n_head': 12,         # Number of attention heads
        'n_embd': 768,        # Embedding dimension
        'block_size': 1024,   # Context length
        'bias': False,        # Whether to use bias in linear layers
        'dropout': 0.0,       # Dropout rate (disabled for analysis)
        
        # Device and precision settings
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'dtype': 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
    }

def init_gpt2_model(model_name='gpt2', config=None):
    """
    Initialize GPT-2 model using nanoGPT architecture with HuggingFace weights.
    
    This function bridges Karpathy's nanoGPT implementation with HuggingFace's
    pretrained weights, allowing for detailed analysis while maintaining 
    compatibility with the custom architecture.
    
    Args:
        model_name: Name of the GPT-2 model variant to load ('gpt2', 'gpt2-medium', etc.)
        config: Model configuration dict. If None, uses default 124M config.
    
    Returns:
        tuple: (model, tokenizer, model_args)
            - model: Initialized GPT model with loaded weights
            - tokenizer: HuggingFace tokenizer for the model
            - model_args: Dictionary of model architecture parameters
    
    Raises:
        RuntimeError: If model download fails
    """
    if config is None:
        config = get_default_config_124M()
    
    print(f"Initializing from OpenAI GPT-2 weights: {model_name}")
    os.makedirs(CACHE_DIR, exist_ok=True)
    print(f"Cache directory: {CACHE_DIR}")
       
    # Download and verify model files
    if not download_and_verify_model(model_name, CACHE_DIR):
        raise RuntimeError(f"Failed to download model {model_name}")
    
    model_args = {}
    
    # Load model with custom dropout override
    override_args = dict(dropout=config['dropout'])
    model = GPT.from_pretrained(model_name, str(CACHE_DIR), override_args)
    
    # Extract model configuration parameters
    config_keys = ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']
    for k in config_keys:
        model_args[k] = getattr(model.config, k)
    
    # Crop context length if requested size is smaller
    if config['block_size'] < model.config.block_size:
        model.crop_block_size(config['block_size'])
        model_args['block_size'] = config['block_size']

    # Set up device (with fallback to CPU if CUDA unavailable)
    device = config['device']
    if 'cuda' in device and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = 'cpu'
    
    # Move model to device and setup tokenizer
    model = model.to(device)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token  # Set padding token
    
    return model, tokenizer, model_args

def inspect_state_dict(model, detailed=False):
    """
    Inspect the state dictionary of the model
    
    Args:
        model: PyTorch model
        detailed: If True, prints additional statistics about parameters
    """
    print("\nModel State Dictionary Structure:")
    total_params = 0
    for name, param in model.state_dict().items():
        print(f"\nParameter: {name}")
        print(f"Shape: {param.shape}")
        num_params = param.numel()
        total_params += num_params
        
        if detailed:
            print(f"Data type: {param.dtype}")
            print(f"Number of elements: {num_params:,}")
            print(f"Max value: {param.max().item():.4f}")
            print(f"Min value: {param.min().item():.4f}")
            print(f"Mean value: {param.mean().item():.4f}")
            print(f"Std deviation: {param.std().item():.4f}")
    
    print(f"\nTotal number of parameters: {total_params:,}")

    
if __name__ == "__main__":
    # Example usage and model inspection
    config = get_default_config_124M()
    model, _, model_args = init_gpt2_model('gpt2', config)
    
    # Inspect the loaded model
    inspect_state_dict(model, detailed=False)
    print("\nModel configuration:")
    for k, v in model_args.items():
        print(f"{k}: {v}")