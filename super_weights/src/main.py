"""
Super Weight Analysis Tool

This module performs super weight analysis on transformer models to identify
highly influential weights in MLP layers and measure their impact on model perplexity.

Supported models:
- GPT-2 (using custom nanoGPT implementation with HuggingFace weights)
- Qwen/Qwen2.5-3B (using HuggingFace transformers)

Reference: https://arxiv.org/abs/2411.07191
"""

import torch
from model_loader import load_model_and_tokenizer, get_sample_text
from dataset_loader import get_wikitext2
from analysis import (
    register_activation_hooks, 
    register_activation_hooks_gpt2, 
    remove_hooks, 
    plot_max_abs_activations, 
    analyze_super_weight, 
    analyze_super_weight_gpt
)
from utils import compute_perplexity
import argparse


def load_model(model_name: str):
    """
    Load model and tokenizer based on model name.
    
    Args:
        model_name: Name of the model to load. Supported values:
                   - "Qwen/Qwen2.5-3B" for Qwen model using HuggingFace
                   - "gpt2" for GPT-2 using custom nanoGPT implementation
    
    Returns:
        tuple: (model, tokenizer, num_layers)
            - model: The loaded model instance
            - tokenizer: The corresponding tokenizer
            - num_layers: Number of transformer layers in the model
    
    Raises:
        ValueError: If model_name is not supported
    """
    if model_name == "Qwen/Qwen2.5-3B":
        model, tokenizer = load_model_and_tokenizer(model_name)
        num_layers = len(model.model.layers)
    elif model_name == "gpt2":
        from load_gpt_pretrained import init_gpt2_model, get_default_config_124M
        config = get_default_config_124M()
        model, tokenizer, _ = init_gpt2_model(model_name, config)
        num_layers = len(model.transformer.h)
    else:
        raise ValueError(f"Unsupported model name: {model_name}")
    
    return model, tokenizer, num_layers


def process_gpt2_activations(keys, all_inputs, all_outputs):
    """
    Process GPT-2 activations and separate by layer type.
    
    Separates the collected activations into MLP feed-forward (c_fc) and 
    projection (c_proj) components for analysis.
    
    Args:
        keys: List of hook keys identifying the layer components
        all_inputs: List of input activations for each hooked layer
        all_outputs: List of output activations for each hooked layer
    
    Returns:
        tuple: (mlp_c_fc_ins, mlp_c_fc_outs, mlp_c_proj_ins, mlp_c_proj_outs)
            Four lists containing separated activations by layer type
    
    Raises:
        RuntimeError: If unrecognized hook key is encountered
    """
    mlp_c_fc_ins, mlp_c_fc_outs = [], []
    mlp_c_proj_ins, mlp_c_proj_outs = [], []
    
    for key, inp, out in zip(keys, all_inputs, all_outputs):
        if key.endswith("mlp.c_fc"):
            mlp_c_fc_ins.append(inp)
            mlp_c_fc_outs.append(out)
        elif key.endswith("mlp.c_proj"):
            mlp_c_proj_ins.append(inp)
            mlp_c_proj_outs.append(out)
        else:
            raise RuntimeError(f"Unrecognized hook key: {key}")
    
    return mlp_c_fc_ins, mlp_c_fc_outs, mlp_c_proj_ins, mlp_c_proj_outs


def find_super_weight_layer(mlp_c_proj_ins):
    """
    Find the layer with maximum super-weight potential.
    
    Analyzes the MLP projection inputs across all layers to identify
    which layer has the highest activation values, indicating potential
    super-weight locations.
    
    Args:
        mlp_c_proj_ins: List of MLP projection input tensors for each layer
    
    Returns:
        int: Layer index with the highest activation potential
    """
    sw_layer_potential = []
    
    for i, in_cproj in enumerate(mlp_c_proj_ins):
        d = in_cproj.shape[-1]  # Hidden dimension
        values, indices = in_cproj.abs().flatten().topk(k=3)
        sw_layer_potential.append((i, values[0], indices[0]))
        
        print("-" * 10)
        print(f"Layer MLP C PROJ IN {i} Top Activations:")
        for idx, val in zip(indices, values):
            n, c = divmod(idx.item(), d)  # Convert flat index to (token, channel)
            print(f"token_id:{n}, channel:{c}: {val.item():.2e}")
    
    # Select layer with highest activation
    sw_layer, _, _ = max(sw_layer_potential, key=lambda x: x[1])
    return sw_layer


def generate_plots_gpt2(model_name, mlp_c_fc_ins, mlp_c_fc_outs, mlp_c_proj_ins):
    """
    Generate all plots for GPT-2 analysis.
    
    Creates visualization plots showing maximum absolute activations
    across all layers for different MLP components.
    
    Args:
        model_name: Name of the model (used in plot titles)
        mlp_c_fc_ins: List of MLP feed-forward input activations
        mlp_c_fc_outs: List of MLP feed-forward output activations  
        mlp_c_proj_ins: List of MLP projection input activations
    """
    plot_max_abs_activations(mlp_c_fc_ins, f'{model_name} max mlp.fc_ins', 
                            'Max abs fc input', filename=f"{model_name}_fc_in.png")
    plot_max_abs_activations(mlp_c_fc_outs, f'{model_name} max mpl.fc_outs', 
                            'Max abs fc output', filename=f"{model_name}_fc_out.png")
    plot_max_abs_activations(mlp_c_proj_ins, f'{model_name} max mlp.proj_ins', 
                            'Max abs proj input', filename=f"{model_name}_proj_in.png")
    plot_max_abs_activations(mlp_c_fc_outs, f'{model_name} max mpl.proj_outs', 
                            'Max abs proj output', filename=f"{model_name}_proj_out.png")


def generate_plots_qwen(model_name, all_inputs, all_outputs):
    """
    Generate all plots for Qwen analysis.
    
    Creates visualization plots showing maximum absolute activations
    for down projection layers across the model.
    
    Args:
        model_name: Name of the model (used in plot titles)
        all_inputs: List of down projection input activations
        all_outputs: List of down projection output activations
    """
    plot_max_abs_activations(all_inputs, f'{model_name} max down_proj input', 
                            'Max abs input', filename="inputs_plot.png")
    plot_max_abs_activations(all_outputs, f'{model_name} max down_proj output', 
                            'Max abs output', filename="outputs_plot.png")


def analyze_qwen_activations(all_outputs):
    """
    Analyze and print activation details for Qwen model.
    
    Prints the top activation values and their locations (token, channel)
    for each layer's down projection outputs.
    
    Args:
        all_outputs: List of down projection output tensors for each layer
    """
    for i, output in enumerate(all_outputs):
        d = output.shape[-1]  # Hidden dimension
        values, indices = output.abs().flatten().topk(k=5)
        print("-" * 10)
        print(f"Layer {i} Top Activations:")
        for idx, val in zip(indices, values):
            n, c = divmod(idx.item(), d)  # Convert flat index to (token, channel)
            print(f"token_id:{n}, channel:{c}: {val.item():.2e}")


def evaluate_perplexity_impact(model, result, tokenizer):
    """
    Evaluate perplexity before and after zeroing super-weight.
    
    Tests the impact of the identified super-weight by measuring model
    perplexity before and after setting the super-weight to zero.
    
    Args:
        model: The transformer model
        result: Dictionary containing super-weight analysis results
        tokenizer: Model tokenizer for text processing
    """
    data = get_wikitext2(128, 2048, tokenizer, train=False)
    original_ppl = compute_perplexity(model, data)
    print(f"\nOriginal perplexity: {original_ppl:.3f}")

    # Zero out the super-weight
    with torch.no_grad():
        W = model.get_submodule(f"model.layers.2.mlp.down_proj").weight
        W[result['coords'][0], result['coords'][1]] = 0

    new_ppl = compute_perplexity(model, data)
    print(f"Perplexity after zeroing super-weight: {new_ppl:.3f}")


def run(model_name: str):
    """
    Main analysis function for super-weight detection.
    
    Performs the complete super-weight analysis pipeline:
    1. Load model and prepare input data
    2. Register activation hooks to capture layer activations
    3. Run forward pass to collect activations  
    4. Analyze activations to identify super-weight locations
    5. Generate visualization plots
    6. Evaluate perplexity impact (for Qwen models)
    
    Args:
        model_name: Name of the model to analyze
    """
    print(f"Starting super-weight analysis for model: {model_name}")
    
    # Load model and setup
    model, tokenizer, num_layers = load_model(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Prepare input data
    sample_text = get_sample_text()
    inputs = tokenizer(sample_text, return_tensors="pt").to(device)
    input_ids = inputs['input_ids'].to(device)

    # Register hooks and collect activations
    all_inputs, all_outputs, keys = [], [], []
    
    if model_name == 'gpt2':
        print("Registering GPT-2 activation hooks...")
        hooks = register_activation_hooks_gpt2(model, num_layers, all_inputs, all_outputs, keys)
    else:  # Qwen model
        print("Registering Qwen activation hooks...")
        hooks = register_activation_hooks(model, num_layers, all_inputs, all_outputs)

    # Forward pass to collect activations
    print("Running forward pass to collect activations...")
    with torch.no_grad():
        model(input_ids)
    remove_hooks(hooks)
    
    # Model-specific analysis
    if model_name == 'gpt2':
        print("Processing GPT-2 activations...")
        # Process activations by layer type
        mlp_c_fc_ins, mlp_c_fc_outs, mlp_c_proj_ins, mlp_c_proj_outs = process_gpt2_activations(
            keys, all_inputs, all_outputs)
        
        # Find super-weight layer
        sw_layer = find_super_weight_layer(mlp_c_proj_ins)
        print('======SW LAYER:', sw_layer)

        # Generate plots
        print("Generating visualization plots...")
        generate_plots_gpt2(model_name, mlp_c_fc_ins, mlp_c_fc_outs, mlp_c_proj_ins)
        
        # Analyze super-weights
        print("Analyzing super-weights...")
        res = analyze_super_weight_gpt(model, ['mlp.c_fc', 'mlp.c_proj'], sw_layer, 
                                     [mlp_c_fc_ins, mlp_c_proj_ins], 
                                     [mlp_c_fc_outs, mlp_c_proj_outs])
        print("Super-weight analysis results:")
        print(res)
        
    elif model_name == 'Qwen/Qwen2.5-3B':
        print("Processing Qwen activations...")
        # Analyze activations
        analyze_qwen_activations(all_outputs)
        
        # Generate plots
        print("Generating visualization plots...")
        generate_plots_qwen(model_name, all_inputs, all_outputs)

        # Analyze super-weight
        print("Analyzing super-weight...")
        result = analyze_super_weight(model, all_inputs, all_outputs)
        print(f"\nSuper-weight location: {result['coords']}")
        print(f"Super-weight value: {result['superweight_value']:.3f}")
        print(f"Max abs weight: {result['max_weight']:.3f}")
        print(f"Std: {result['std']:.3f}")

        # Evaluate perplexity impact
        print("Evaluating perplexity impact...")
        evaluate_perplexity_impact(model, result, tokenizer)
    
    print("Super-weight analysis completed successfully!")

if __name__ == "__main__":
    # Command line argument parsing (currently commented out for HPC use)
    # Uncomment the following lines to enable command line arguments:
    """
    parser = argparse.ArgumentParser(
        description="Run activation SW analysis on a specified model"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="gpt2",
        help="Which pretrained model to load (gpt2 or Qwen/Qwen2.5-3B)"
    )
    args = parser.parse_args()
    run(args.model_name)
    """
    
    # For HPC/batch usage, directly specify the model
    # Change to "Qwen/Qwen2.5-3B" to run analysis on Qwen model
    run("gpt2")
