# Super Weights Analysis

This is a modular Python application that analyzes transformer models to identify "super weights" — highly influential weights in MLP down projection layers — and measures their impact on perplexity.

## Features

- Loads a transformer model ( Qwen2.5-3B)
- Hooks into MLP `down_proj` layers to log activations
- Identifies top activation channels and "super weights"
- Measures perplexity before/after zeroing out super weights
- Saves plots of max activations as PNGs

## Structure
```super_weights/
├── src/
│   ├── __init__.py
│   ├── main.py                  # main entry logic
│   ├── dataset_loader.py      # loads WikiText2 data
│   ├── model_loader.py          #loads the model, tokenizer
│   ├── analysis.py              # hooking and sw analysis
│   └── utils.py                 # perplexity 
├── logs/                    # slurm job output
├── output/               # plots produced                 
├── run_app.py                   # entry: python run_app.py!
├── job_run.sh            # sbatch this job!
├── requirements.txt
└── README.md
```

