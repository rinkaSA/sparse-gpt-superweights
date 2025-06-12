import os
import torch
import sys
from transformers import GPT2Tokenizer , GPT2LMHeadModel
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()


REPO_ROOT = Path(__file__).parent.parent.absolute()
NANOGPT_PATH = os.path.join(REPO_ROOT, 'gpt2', 'nanoGPT')
sys.path.append(NANOGPT_PATH)

CACHE_DIR = Path(os.getenv('TRANSFORMERS_CACHE', 
                          str(REPO_ROOT / 'cache' / 'huggingface' / 'transformers')))

def download_and_verify_model(model_name, cache_dir):
    """Pre-download and verify model files"""
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
    
from model import GPT, GPTConfig

def get_default_config_124M():

    return {
       
        'n_layer': 12,
        'n_head': 12,
        'n_embd': 768,
        'block_size': 1024,
        'bias': False,
        'dropout': 0.0,
        
    
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'dtype': 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
    }

def init_gpt2_model(model_name='gpt2', config=None):

    if config is None:
        config = get_default_config_124M()
    
    print(f"Initializing from OpenAI GPT-2 weights: {model_name}")
    os.makedirs(CACHE_DIR, exist_ok=True)
    print(f"Cache directory: {CACHE_DIR}")
       
    if not download_and_verify_model(model_name, CACHE_DIR):
        raise RuntimeError(f"Failed to download model {model_name}")
    
    model_args = {}
    

    override_args = dict(dropout=config['dropout'])
    model = GPT.from_pretrained(model_name, str(CACHE_DIR), override_args)
    

    config_keys = ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']
    for k in config_keys:
        model_args[k] = getattr(model.config, k)
    

    if config['block_size'] < model.config.block_size:
        model.crop_block_size(config['block_size'])
        model_args['block_size'] = config['block_size']

    device = config['device']
    if 'cuda' in device and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = 'cpu'
    

    model = model.to(device)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
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
    config = get_default_config_124M()
    model, _, model_args = init_gpt2_model('gpt2', config)
    
    inspect_state_dict(model, detailed=False)
    print("\nModel configuration:")
    for k, v in model_args.items():
        print(f"{k}: {v}")

    