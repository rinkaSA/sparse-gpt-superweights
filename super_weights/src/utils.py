"""
Utility Functions for Super Weight Analysis

This module provides utility functions for evaluating model performance,
particularly perplexity computation for measuring super weight impact.
"""

import torch
import torch.nn.functional as F
from tqdm import trange

@torch.no_grad()
def compute_perplexity(model, data, batch_size: int = 4):
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
    total_loss = 0.0
    total_tokens = 0
    
    # Limit the number of samples for faster evaluation
    data = data[:min(len(data), 10)]  # Use maximum 10 samples for debugging
    
    for i in trange(0, len(data), batch_size, desc="Computing perplexity", leave=False):
        # Prepare batch
        batch_data = data[i:i+batch_size]
        inputs = torch.cat(batch_data).to(device)
        
        # Check if this is a nanoGPT model by testing output shape
        test_input = inputs[:1, :2]  # Just first sample, first 2 tokens
        test_out = model(test_input)
        
        if isinstance(test_out, tuple) or isinstance(test_out, list):
            test_logits = test_out[0]
        elif hasattr(test_out, "logits"):
            test_logits = test_out.logits
        else:
            test_logits = test_out
            
        # If model only returns logits for last position (nanoGPT behavior)
        if test_logits.shape[1] == 1:
            # For nanoGPT: process in sliding window to get all positions
            batch_loss = 0.0
            batch_tokens = 0
            
            for seq_idx, seq in enumerate(batch_data):
                seq = seq.to(device)
                seq_loss = 0.0
                seq_tokens = 0
                
                # Process sequence with sliding window approach
                context_size = min(512, seq.shape[1] // 2)  # Use smaller context for efficiency
                
                for pos in range(context_size, seq.shape[1], context_size // 2):  # Skip every context_size//2
                    start_pos = max(0, pos - context_size)
                    context = seq[:, start_pos:pos]
                    target = seq[:, pos:pos+1]
                    
                    if target.shape[1] == 0:
                        continue
                        
                    out = model(context)
                    if isinstance(out, tuple) or isinstance(out, list):
                        logits = out[0]
                    elif hasattr(out, "logits"):
                        logits = out.logits
                    else:
                        logits = out
                    
                    # logits should be (B, 1, V) for the last position
                    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target.contiguous().view(-1))
                    seq_loss += loss * target.numel()
                    seq_tokens += target.numel()
                
                if seq_tokens > 0:
                    batch_loss += seq_loss
                    batch_tokens += seq_tokens
            
            if batch_tokens > 0:
                avg_loss = batch_loss / batch_tokens
                total_loss += avg_loss * batch_tokens
                total_tokens += batch_tokens
                
        else:
            # Standard processing for models that return full sequence logits
            out = model(inputs)

            if isinstance(out, tuple) or isinstance(out, list):
                logits = out[0]
            elif hasattr(out, "logits"):
                logits = out.logits
            else:
                logits = out

            # Ensure shape is (B, T, V)
            B, T = inputs.shape
            if logits.dim() != 3:
                raise RuntimeError(f"Expected 3D logits, got {logits.dim()}D with shape {logits.shape}")
                
            if logits.shape[:2] != (B, T):
                if logits.shape == (B, T - 1, logits.shape[-1]):
                    # Model already shifted logits
                    shift_logits = logits
                    shift_labels = inputs[:, 1:]
                else:
                    raise RuntimeError(f"Unexpected logits shape {logits.shape} for inputs {inputs.shape}")
            else:
                # Prepare targets (shifted by one position for language modeling)
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = inputs[:, 1:]
            
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.contiguous().view(-1))
            
            # Accumulate loss
            batch_tokens = shift_labels.numel()
            total_loss += loss * batch_tokens
            total_tokens += batch_tokens
    
    if total_tokens == 0:
        return float('inf')
    
    # Return perplexity (exponentiated average NLL)
    avg_loss = total_loss / total_tokens
    perplexity = avg_loss.exp().item()
    
    # Debug print
    print(f"  Total tokens: {total_tokens}, Avg loss: {avg_loss:.4f}, Perplexity: {perplexity:.3f}")
    
    return perplexity
