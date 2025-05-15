from typing import Sequence
import torch
import matplotlib.pyplot as plt

def register_activation_hooks(model, num_layers: int, all_inputs: list, all_outputs: list):
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

def remove_hooks(hooks):
    for hook in hooks:
        hook.remove()

import matplotlib.pyplot as plt
import os

def plot_max_abs_activations(data, title, ylabel, filename="activation_plot.png", save_dir="outputs"):
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, filename)

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


def analyze_super_weight(model, all_inputs, all_outputs):
    d_in = all_inputs[1].shape[-1]
    max_id_in = all_inputs[1].abs().flatten().argmax().item()
    n_in, c_in = divmod(max_id_in, d_in)

    d_out = all_outputs[1].shape[-1]
    max_id_out = all_outputs[1].abs().flatten().argmax().item()
    n_out, c_out = divmod(max_id_out, d_out)

    W = model.get_submodule(f"model.layers.2.mlp.down_proj").weight
    superweight = W[c_out, c_in].item()
    return {
        "coords": (c_out, c_in),
        "superweight_value": superweight,
        "max_weight": W.abs().max().item(),
        "std": W.std().item()
    }
