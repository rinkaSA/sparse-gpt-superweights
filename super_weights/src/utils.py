"""
Utility Functions for Super Weight Analysis

This module provides utility functions for evaluating model performance,
particularly perplexity computation for measuring super weight impact.
"""

import torch
import torch.nn.functional as F
from tqdm import trange

@torch.no_grad()
def compute_perplexity(model, data, batch_size: int = 1):
    """
    Compute model perplexity on given data.
    
    Calculates the perplexity (exponentiated cross-entropy loss) of the model
    on the provided dataset. Used to measure the impact of super weight
    modifications on model performance.
    
    Args:
        model: The transformer model to evaluate
        data: List of tokenized sequences (each as a PyTorch tensor)
        batch_size: Number of sequences to process in each batch
    
    Returns:
        float: Perplexity value (lower is better)
    """
    device = next(model.parameters()).device
    nll_running = 0  # Running negative log-likelihood
    tokens_processed = 0
    
    for i in trange(0, len(data), batch_size, desc="Computing perplexity", leave=False):
        # Prepare batch
        inputs = torch.cat(data[i:i+batch_size]).to(device)
        
        # Forward pass
        logits = model(inputs).logits
        
        # Prepare targets (shifted by one position for language modeling)
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = inputs[:, 1:]
        
        # Compute cross-entropy loss
        loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        # Update running average of negative log-likelihood
        a = shift_labels.numel() / (tokens_processed + shift_labels.numel())
        b = tokens_processed / (tokens_processed + shift_labels.numel())
        nll_running = a * loss + b * nll_running
        tokens_processed += shift_labels.numel()
    
    # Return perplexity (exponentiated NLL)
    return nll_running.exp().item()
