import torch
from .model_loader import load_model_and_tokenizer, get_sample_text
from .dataset_loader import get_wikitext2
from .analysis import register_activation_hooks, remove_hooks, plot_max_abs_activations, analyze_super_weight
from .utils import compute_perplexity

def run():
    model_name = "Qwen/Qwen2.5-3B"
    model, tokenizer = load_model_and_tokenizer(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    sample_text = get_sample_text()
    inputs = tokenizer(sample_text, return_tensors="pt").to(device)

    num_layers = len(model.model.layers)
    all_inputs, all_outputs = [], []
    hooks = register_activation_hooks(model, num_layers, all_inputs, all_outputs)

    with torch.no_grad():
        model(**inputs)

    remove_hooks(hooks)

    for i, output in enumerate(all_outputs):
        d = output.shape[-1]
        values, indices = output.abs().flatten().topk(k=5)
        print("-" * 10)
        print(f"Layer {i} Top Activations:")
        for idx, val in zip(indices, values):
            n, c = divmod(idx.item(), d)
            print(f"token_id:{n}, channel:{c}: {val.item():.2e}")

    plot_max_abs_activations(all_inputs, f'{model_name} max down_proj input', 'Max abs input', filename="inputs_plot.png")
    plot_max_abs_activations(all_outputs, f'{model_name} max down_proj output', 'Max abs output', filename="outputs_plot.png")


    result = analyze_super_weight(model, all_inputs, all_outputs)
    print(f"\nSuper-weight location: {result['coords']}")
    print(f"Super-weight value: {result['superweight_value']:.3f}")
    print(f"Max abs weight: {result['max_weight']:.3f}")
    print(f"Std: {result['std']:.3f}")

    data = get_wikitext2(128, 2048, tokenizer, train=False)
    original_ppl = compute_perplexity(model, data)
    print(f"\nOriginal perplexity: {original_ppl:.3f}")

    with torch.no_grad():
        W = model.get_submodule(f"model.layers.2.mlp.down_proj").weight
        W[result['coords'][0], result['coords'][1]] = 0

    new_ppl = compute_perplexity(model, data)
    print(f"Perplexity after zeroing super-weight: {new_ppl:.3f}")
