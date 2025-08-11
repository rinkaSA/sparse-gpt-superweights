"""
Analysis Module for Super Weight Detection

This module provides functions for:
- Registering forward hooks to capture activations
- Analyzing activations to find super weights
- Plotting activation patterns across layers

The analysis targets MLP layers in transformer models to identify
weights with disproportionate influence on model outputs.
"""

from typing import Sequence
import torch
import matplotlib.pyplot as plt
import os
from pathlib import Path

def register_activation_hooks(model, num_layers: int, all_inputs: list, all_outputs: list):
    """
    Register forward hooks on Qwen model's down projection layers.
    
    Attaches hooks to capture input and output activations from MLP down projection
    layers across all transformer blocks for super weight analysis.
    
    Args:
        model: The Qwen transformer model
        num_layers: Number of transformer layers in the model
        all_inputs: List to store captured input activations
        all_outputs: List to store captured output activations
    
    Returns:
        list: Hook handles for later removal
    """
    hooks = []
    for i in range(num_layers):
        layer = model.get_submodule(f"model.layers.{i}.mlp.down_proj")
        def cache_inputs_outputs_hook(_, inputs, outputs):
            if isinstance(inputs, Sequence):
                inputs = inputs[0]
            all_inputs.append(inputs.cpu())
            all_outputs.append(outputs.cpu())
        hooks.append(layer.register_forward_hook(cache_inputs_outputs_hook))
    return hooks

def register_activation_hooks_gpt2(model, num_layers: int, all_inputs: list, all_outputs: list, hooks_keys: list):
    """
    Register forward hooks on GPT-2 model's MLP layers.
    
    Attaches hooks to capture activations from both feed-forward (c_fc) and 
    projection (c_proj) components of MLP layers for comprehensive analysis.
    
    Args:
        model: The GPT-2 transformer model
        num_layers: Number of transformer layers in the model
        all_inputs: List to store captured input activations
        all_outputs: List to store captured output activations
        hooks_keys: List to store hook identifiers for layer tracking
    
    Returns:
        list: Hook handles for later removal
    """
    hooks = []
    for i in range(num_layers):
        names = [
            # Uncomment to include attention layers in analysis:
            # f"transformer.h.{i}.attn.c_attn",
            # f"transformer.h.{i}.attn.c_proj",
            f"transformer.h.{i}.mlp.c_fc",    # Feed-forward layer
            f"transformer.h.{i}.mlp.c_proj",  # Projection layer
        ]
        for name in names:
            layer = model.get_submodule(name)
            def cache_inputs_outputs_hook(_, inputs, outputs):
                if isinstance(inputs, Sequence):
                    inputs = inputs[0]
                all_inputs.append(inputs.cpu())
                all_outputs.append(outputs.cpu())
            hooks.append(layer.register_forward_hook(cache_inputs_outputs_hook))
            hooks_keys.append(name)
    return hooks

def remove_hooks(hooks):
    """
    Remove all registered forward hooks.
    
    Cleans up forward hooks to prevent memory leaks and interference
    with subsequent model operations.
    
    Args:
        hooks: List of hook handles to remove
    """
    for hook in hooks:
        hook.remove()



def plot_max_abs_activations(data, title, ylabel, filename="activation_plot.png", save_dir="plot_outputs"):
    """
    Plot maximum absolute activations across layers.
    
    Creates a line plot showing the maximum absolute activation value
    for each layer, useful for identifying layers with high activation magnitudes.
    
    Args:
        data: List of activation tensors, one per layer
        title: Plot title
        ylabel: Y-axis label
        filename: Output filename for the plot
        save_dir: Directory name for saving plots (relative to project root)
    """
    # Ensure save directory exists (relative to super_weights directory)
    script_dir = Path(__file__).parent.parent.absolute()  # Go up to super_weights directory
    save_dir = script_dir / "plot_outputs"
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, filename)

    # Create plot
    fig, ax = plt.subplots(figsize=(7, 5))
    max_vals = [x.abs().max().item() for x in data]
    ax.plot(max_vals, marker='o')
    ax.set_xlabel('Layer id')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(filepath, dpi=300)
    plt.close(fig) 
    print(f"[Saved] {filepath}")
    

def analyze_super_weight_unified(model, model_name, target_layer, all_inputs, all_outputs, component_names=None):
    """
    Unified super weight analysis function for both GPT and Qwen models.
    
    Identifies the coordinates and values of potential super weights by finding
    the maximum activations and extracting corresponding weight values.
    
    Args:
        model: The transformer model (GPT-2 or Qwen)
        model_name: Model type ("gpt2" or "Qwen/Qwen2.5-3B")
        target_layer: Target layer index for analysis
        all_inputs: Input activations (for Qwen: list per layer, for GPT: list per component)
        all_outputs: Output activations (for Qwen: list per layer, for GPT: list per component)
        component_names: For GPT models, list of component names like ['mlp.c_fc', 'mlp.c_proj']
    
    Returns:
        dict: Analysis results. For GPT: nested dict by component name.
              For Qwen: single dict with layer info.
              Each result contains:
                - coords: (output_channel, input_channel) coordinates
                - superweight_value: The weight value at those coordinates  
                - max_weight: Maximum absolute weight in the layer
                - std: Standard deviation of weights in the layer
                - layer: Layer index for proper weight zeroing
    """
    if model_name == "gpt2":
        # GPT-2 analysis - analyze multiple components
        if component_names is None:
            component_names = ['mlp.c_fc', 'mlp.c_proj']
            
        results = {}
        
        # Analyze each MLP component (c_fc and c_proj)
        for i, component_name in enumerate(component_names):
            ins = all_inputs[i]
            outs = all_outputs[i]
            
            # Find maximum input activation coordinates
            d_in = ins[target_layer].shape[-1]
            max_id_in = ins[target_layer].abs().flatten().argmax().item()
            n_in, c_in = divmod(max_id_in, d_in)  # Convert to (token, channel)

            # Find maximum output activation coordinates  
            d_out = outs[target_layer].shape[-1]
            max_id_outs = outs[target_layer].abs().flatten().argmax().item()
            n_outs, c_outs = divmod(max_id_outs, d_out)  # Convert to (token, channel)

            # Extract the super weight from model parameters
            W = model.get_submodule(f"transformer.h.{target_layer}.{component_name}").weight
            superweight = W[c_outs, c_in].item()
            
            # Store analysis results
            results[component_name] = {
                "coords": (c_outs, c_in),
                "superweight_value": superweight,
                "max_weight": W.abs().max().item(),
                "std": W.std().item(),
                "layer": target_layer
            }
        return results
        
    else:
        # Qwen/other model analysis - single component (down_proj)
        
        # Find maximum input activation coordinates
        d_in = all_inputs[target_layer].shape[-1]
        max_id_in = all_inputs[target_layer].abs().flatten().argmax().item()
        n_in, c_in = divmod(max_id_in, d_in)  # Convert to (token, channel)

        # Find maximum output activation coordinates
        d_out = all_outputs[target_layer].shape[-1]
        max_id_out = all_outputs[target_layer].abs().flatten().argmax().item()
        n_out, c_out = divmod(max_id_out, d_out)  # Convert to (token, channel)

        # Extract the super weight from model parameters
        W = model.get_submodule(f"model.layers.{target_layer}.mlp.down_proj").weight
        superweight = W[c_out, c_in].item()
        
        return {
            "coords": (c_out, c_in),
            "superweight_value": superweight,
            "max_weight": W.abs().max().item(),
            "std": W.std().item(),
            "layer": target_layer
        }


def analyze_super_weight_gpt(model, names, layer_number, list_inputs, list_outputs):
    return analyze_super_weight_unified(model, "gpt2", layer_number, list_inputs, list_outputs, names)


def analyze_super_weight(model, all_inputs, all_outputs, target_layer=2):
    return analyze_super_weight_unified(model, "Qwen/Qwen2.5-3B", target_layer, all_inputs, all_outputs)
