# Super Weight fidning directory


This is a modular Python application that analyzes transformer models to identify "super weights" — highly influential weights in MLP down projection layers — and measures their impact on perplexity.

### Features

- Loads a transformer model ( Qwen2.5-3B)
- Hooks into MLP `down_proj` layers to log activations
- Identifies top activation channels and "super weights"
- Measures perplexity before/after zeroing out super weights
- Saves plots of max activations as PNGs


For finding a SW one should run ```sbatch find_sw.sh```
This will run ```main.py```  with a configurable parameter of a chosen model. Options for now are GPT2, Qwen/Qwen2.5-3B (Llama 2 should work as well as soon as HF_token is provided).

The slurm job output will reside at ```/slurm_sw_out```. The plots for down projection layers inputs/outputs across all the Transformer Blocks are stored at ```/outputs```

!Note: create ```/cache/huggingface/transformers``` in root dir of this project. In load_gpt_pretrained.py we point out this as the dir for cache. 


GPT2 model is using the architecture defined by our config from scrape, but for SW finding we apply pretrained weights which we pull from HF.