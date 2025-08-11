"""
Constants and Configuration

This module contains shared constants and configuration values used across
the super weight analysis codebase.
"""

# Supported model names
SUPPORTED_MODELS = {
    "gpt2": "GPT-2 124M using custom nanoGPT implementation",
    "Qwen/Qwen2.5-3B": "Qwen 2.5 3B using HuggingFace transformers"
}

# Default analysis parameters
DEFAULT_SEQUENCE_LENGTH = 2048
DEFAULT_NUM_SAMPLES = 128
DEFAULT_BATCH_SIZE = 1

# Plot settings
PLOT_DPI = 300
PLOT_FIGSIZE = (7, 5)

# Layer analysis settings
# For Qwen models, empirically layer 2 (index 2) often contains super weights
QWEN_TARGET_LAYER_INDEX = 2

# Top-k activations to analyze
TOP_K_ACTIVATIONS = 5
