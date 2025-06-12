import torch
from model_loader import load_model_and_tokenizer, get_sample_text
from dataset_loader import get_wikitext2
from analysis import register_activation_hooks, register_activation_hooks_gpt2, remove_hooks, plot_max_abs_activations, analyze_super_weight, analyze_super_weight_gpt
from utils import compute_perplexity
import argparse

def run(model_name: str):
    model_name = "gpt2"
    if model_name == "Qwen/Qwen2.5-3B":
        model, tokenizer = load_model_and_tokenizer(model_name)
        num_layers = len(model.model.layers)
    elif model_name == "gpt2":
        from load_gpt_pretrained import init_gpt2_model, get_default_config_124M
        config = get_default_config_124M()
        model, tokenizer, _ = init_gpt2_model(model_name, config)
        num_layers = len(model.transformer.h)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    sample_text = get_sample_text()
    inputs = tokenizer(sample_text, return_tensors="pt").to(device)
    input_ids = inputs['input_ids'].to(device)

    all_inputs, all_outputs, keys = [], [], []
    if model_name == 'gpt2':
        hooks = register_activation_hooks_gpt2(model, num_layers, all_inputs, all_outputs, keys)

    with torch.no_grad():
        model(input_ids)

    remove_hooks(hooks)
    
   # attn_c_attn_ins,  attn_c_attn_outs  = [], []
   #attn_c_proj_ins,  attn_c_proj_outs  = [], []
    mlp_c_fc_ins,     mlp_c_fc_outs     = [], []
    mlp_c_proj_ins,   mlp_c_proj_outs   = [], []
    
    if model_name == 'gpt2':
        for key, inp, out in zip(keys, all_inputs, all_outputs):
            #if key.endswith("attn.c_attn"):
               # attn_c_attn_ins.append(inp)
                #attn_c_attn_outs.append(out)
            #elif key.endswith("attn.c_proj"):
                #attn_c_proj_ins.append(inp)
                #attn_c_proj_outs.append(out)
            if key.endswith("mlp.c_fc"):
                mlp_c_fc_ins.append(inp)
                mlp_c_fc_outs.append(out)
            elif key.endswith("mlp.c_proj"):
                mlp_c_proj_ins.append(inp)
                mlp_c_proj_outs.append(out)
            else:
                raise RuntimeError(f"Unrecognized hook key: {key}")
            
        sw_layer = 0
        sw_layer_potetial = []
        for i, in_cproj in enumerate(mlp_c_proj_ins):
            d = in_cproj.shape[-1]
            values, indices = in_cproj.abs().flatten().topk(k=3)
            sw_layer_potetial.append((i,  values[0], indices[0]))
            print("-" * 10)
            print(f"Layer MLP C PROJ IN {i} Top Activations:")
            for idx, val in zip(indices, values):
                n, c = divmod(idx.item(), d)
                print(f"token_id:{n}, channel:{c}: {val.item():.2e}")
        sw_layer, _, _ =  max(sw_layer_potetial, key=lambda x: x[1])

        print(' SW LAYER:', sw_layer)

        plot_max_abs_activations(mlp_c_fc_ins, f'{model_name} max mlp.fc_ins', 'Max abs fc input', filename="fc_in_gpt2.png")
        plot_max_abs_activations(mlp_c_fc_outs, f'{model_name} max mpl.fc_outs ', 'Max abs fc output', filename="fc_out_gpt2.png")
        plot_max_abs_activations(mlp_c_proj_ins, f'{model_name} max mlp.proj_ins', 'Max abs proj input', filename="proj_in_gpt2.png")
        plot_max_abs_activations(mlp_c_fc_outs, f'{model_name} max mpl.proj_outs ', 'Max abs proj output', filename="proj_out_gpt2.png")
        res = analyze_super_weight_gpt(model,['mlp.c_fc', 'mlp.c_proj'],sw_layer, [mlp_c_fc_ins,mlp_c_proj_ins], [mlp_c_fc_outs, mlp_c_proj_outs])
        print(res)
    else:
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