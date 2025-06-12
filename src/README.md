# Super Weight lookout directory

Temporarily has Lottery Winning Ticker Hypothesis scrips residing here.

For finding a SW one should run ```sbatch find_sw.sh```
This will run ```main.py```  with a configurable parameter of a chosen model. Options for now are GPT2, Qwen/Qwen2.5-3B (Llama 2 should work as well as soon as HF_token is provided).

The slurm job output will reside at ```/slurm_sw_out```. The plots for down projection layers inputs/outputs across all the Transformer Blocks are stored at ```/outputs```

!Note: create ```/cache/huggingface/transformers``` in root dir of this project. In load_gpt_pretrained.py we point out this as the dir for cache. 


GPT2 model is using the architecture defined by our config from scrape, but for SW finding we apply pretrained weights which we pull from HF.