import torch
from model_loader import load_model_and_tokenizer, get_sample_text
from dataset_loader import get_wikitext2
from analysis import register_activation_hooks, register_activation_hooks_gpt2, remove_hooks, plot_max_abs_activations, analyze_super_weight_unified
from utils import compute_perplexity
import argparse


def load_model(model_name: str):
    """Load model and tokenizer based on model name."""
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
    """Process GPT-2 activations and separate by layer type."""
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
    """Find the layer with maximum super-weight potential."""
    sw_layer_potential = []
    
    for i, in_cproj in enumerate(mlp_c_proj_ins):
        d = in_cproj.shape[-1]
        values, indices = in_cproj.abs().flatten().topk(k=3)
        sw_layer_potential.append((i, values[0], indices[0]))
        
        print("-" * 10)
        print(f"Layer MLP C PROJ IN {i} Top Activations:")
        for idx, val in zip(indices, values):
            n, c = divmod(idx.item(), d)
            print(f"token_id:{n}, channel:{c}: {val.item():.2e}")
    
    # Select layer with highest activation
    sw_layer, _, _ = max(sw_layer_potential, key=lambda x: x[1])
    return sw_layer


def generate_plots_gpt2(model_name, mlp_c_fc_ins, mlp_c_fc_outs, mlp_c_proj_ins):
    """Generate all plots for GPT-2 analysis."""
    plot_max_abs_activations(mlp_c_fc_ins, f'{model_name} max mlp.fc_ins', 
                            'Max abs fc input', filename=f"{model_name}_fc_in.png")
    plot_max_abs_activations(mlp_c_fc_outs, f'{model_name} max mpl.fc_outs', 
                            'Max abs fc output', filename=f"{model_name}_fc_out.png")
    plot_max_abs_activations(mlp_c_proj_ins, f'{model_name} max mlp.proj_ins', 
                            'Max abs proj input', filename=f"{model_name}_proj_in.png")
    plot_max_abs_activations(mlp_c_fc_outs, f'{model_name} max mpl.proj_outs', 
                            'Max abs proj output', filename=f"{model_name}_proj_out.png")


def generate_plots_qwen(model_name, all_inputs, all_outputs):
    """Generate all plots for Qwen analysis."""
    plot_max_abs_activations(all_inputs, f'{model_name} max down_proj input', 
                            'Max abs input', filename=f"{model_name}_inputs_plot.png")
    plot_max_abs_activations(all_outputs, f'{model_name} max down_proj output', 
                            'Max abs output', filename=f"{model_name}_outputs_plot.png")


def analyze_qwen_activations(all_outputs):
    """Analyze and print activation details for Qwen model."""
    for i, output in enumerate(all_outputs):
        d = output.shape[-1]
        values, indices = output.abs().flatten().topk(k=5)
        print("-" * 10)
        print(f"Layer {i} Top Activations:")
        for idx, val in zip(indices, values):
            n, c = divmod(idx.item(), d)
            print(f"token_id:{n}, channel:{c}: {val.item():.2e}")


def evaluate_perplexity_impact(model, result, tokenizer, seq_length=1024, model_name="gpt2"):
    """
    Evaluate perplexity before and after zeroing super-weight.
    
    Tests the impact of the identified super-weight by measuring model
    perplexity before and after setting the super-weight to zero.
    
    Args:
        model: The transformer model
        result: Dictionary containing super-weight analysis results
        tokenizer: Model tokenizer for text processing
        seq_length: Sequence length for evaluation
        model_name: Name of the model to determine correct module structure
    """

    data = get_wikitext2(128, seq_length, tokenizer, train=False) 
    print(f"Evaluating on {len(data)} sequences of length {seq_length}")
    original_ppl = compute_perplexity(model, data)
    print(f"\nOriginal perplexity: {original_ppl:.3f}")

    # Zero out the super-weight based on model type
    with torch.no_grad():
        coords = result['coords']
        if model_name == "gpt2":
            # For GPT-2 models - use the layer info if available
            layer_idx = result.get('layer', 2)  # Default to layer 2 if not specified
            W = model.get_submodule(f"transformer.h.{layer_idx}.mlp.c_proj").weight
        else:
            # For Qwen and other models - use the layer info if available
            layer_idx = result.get('layer', 2)  # Default to layer 2 if not specified
            W = model.get_submodule(f"model.layers.{layer_idx}.mlp.down_proj").weight
        
        # Store original value for comparison
        original_value = W[coords[0], coords[1]].item()
        print(f"Zeroing super-weight at layer {layer_idx}, coords {coords}")
        print(f"Original weight value: {original_value:.6f}")
        
        W[coords[0], coords[1]] = 0

    new_ppl = compute_perplexity(model, data)
    print(f"Perplexity after zeroing super-weight: {new_ppl:.3f}")
    print(f"Perplexity change: {new_ppl - original_ppl:.3f} ({((new_ppl - original_ppl) / original_ppl) * 100:.1f}%)")


def run(model_name: str):
    """Main analysis function for super-weight detection."""

    model, tokenizer, num_layers = load_model(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Prepare input data
    sample_text = get_sample_text()
    inputs = tokenizer(sample_text, return_tensors="pt").to(device)
    input_ids = inputs['input_ids'].to(device)

    # Register hooks and collect activations
    all_inputs, all_outputs, keys = [], [], []
    
    if model_name == 'gpt2':
        hooks = register_activation_hooks_gpt2(model, num_layers, all_inputs, all_outputs, keys)
    else:  # Qwen model
        hooks = register_activation_hooks(model, num_layers, all_inputs, all_outputs)

    # Forward pass to collect activations
    with torch.no_grad():
        model(input_ids)
    remove_hooks(hooks)
    
    # Model-specific analysis
    if model_name == 'gpt2':
        # Process activations by layer type
        mlp_c_fc_ins, mlp_c_fc_outs, mlp_c_proj_ins, mlp_c_proj_outs = process_gpt2_activations(
            keys, all_inputs, all_outputs)
        
        # Find super-weight layer
        sw_layer = find_super_weight_layer(mlp_c_proj_ins)
        print('======SW LAYER:', sw_layer)

        # Generate plots
        generate_plots_gpt2(model_name, mlp_c_fc_ins, mlp_c_fc_outs, mlp_c_proj_ins)
        
        # Analyze super-weights
        res = analyze_super_weight_unified(model, 'gpt2', sw_layer, 
                                         [mlp_c_fc_ins, mlp_c_proj_ins], 
                                         [mlp_c_fc_outs, mlp_c_proj_outs],
                                         ['mlp.c_fc', 'mlp.c_proj'])
        print(res)
        evaluate_perplexity_impact(model, res['mlp.c_proj'], tokenizer, seq_length=1024, model_name=model_name)

        
    elif model_name == 'Qwen/Qwen2.5-3B':
        print("Processing Qwen activations...")
        # Analyze activations
        analyze_qwen_activations(all_outputs)
        
        # Generate plots
        generate_plots_qwen(model_name, all_inputs, all_outputs)

        # Analyze super-weight
        result = analyze_super_weight_unified(model, model_name, 2, all_inputs, all_outputs)
        print(f"\nSuper-weight location: {result['coords']}")
        print(f"Super-weight value: {result['superweight_value']:.3f}")
        print(f"Max abs weight: {result['max_weight']:.3f}")
        print(f"Std: {result['std']:.3f}")
        print(f"Layer: {result['layer']}")

        # Evaluate perplexity impact
        evaluate_perplexity_impact(model, result, tokenizer, seq_length=2048, model_name=model_name)

if __name__ == "__main__":
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


    