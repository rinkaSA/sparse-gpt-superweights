# Super Weight Location Search Directory


This repository analyzes transformer models to identify "super weights" — highly influential weights in MLP down projection layers — and measures their impact on perplexity.

### Reference paper

https://arxiv.org/abs/2411.07191


### Features

- Loads a transformer model (Qwen2.5-3B/GPT2 )
- Hooks into MLP `down_proj` layers to log activations
- Identifies top activation channels and "super weights"
- Measures perplexity before/after zeroing out super weights
- Saves plots of max activations as PNGs


### Structure

```super_weights/
├── /plot_outputs -> plots created during the run of .sh file
├── /slurm_out   -> .err and .out files created during HPC job run
├── /src         -> source code, where the logic is defined
├── find_sw.sh   -> main runnable script, adapted for HPC use                     
└── README.md
```



### Usage

1. For finding a SW one should run ```sbatch find_sw.sh```, make sure to specify the correct  ```--model_name``` parameter.At the moment only ``` gpt2``` OR ```Qwen/Qwen2.5-3B ``` are supported. 

2. The slurm job output will reside at ```/slurm_out```. The plots for down projection layers inputs/outputs across all the Transformer Blocks are stored at ```/outputs```

!Note: create ```/cache/huggingface/transformers``` in root dir of this project. In load_gpt_pretrained.py we point out this as the dir for cache. 

### About GPR 2 usage

GPT2 model is using the architecture defined by our config from scrape ([nanogpt2](../gpt2/nanoGPT)), but for SW finding we apply pretrained weights which we pull from HF. The weights are applied on our user-defined architecture,  with tricks to adapt the layer names (credit to A Karpaty). The linking file that utilizes the GPT2 repo is ```load_gpt_pretrained.py```

