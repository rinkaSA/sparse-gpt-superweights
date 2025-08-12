# Super Weights: Analysis, nanoGPT-124M Training, and Lottery Winning Ticket Hypothesis

This repository bundles three related pieces of work around GPT-2–style models:

1) **nanoGPT (124M) training** — Karpathy’s nanoGPT adapted for SLURM clusters (single & distributed).  
2) **Super Weight analysis** — tools to locate and analyze influential (“super”) weights.  
3) **Lottery Winning Ticket Hypothesis (LWH)** — pruning & re-training experiments, including checks that previously found super-weight locations are preserved.

> Each subproject has its own README with deeper instructions. Start here, then jump to the sub-READMEs as needed.

---

## Repository layout

```
super_weights/                      # repository root
├─ gpt2/nanoGPT/                    # nanoGPT adapted for SLURM
│  ├─ slurm_job_scripts/            # SLURM jobs (dist & non-dist)
│  ├─ README.md                     # upstream-style usage
│  └─ README_iryna.md               # repo-specific tips/notes
├─ super_weights/                   # Super Weight analysis code
│  └─ README.md
├─ LWH/                             # Lottery Winning Ticket Hypothesis (experiments + results)
│  └─ README.md
│  └─ .gitattributes                # Git LFS tracking for large artifacts (html/csv, etc.)
├─ requirements.txt                 # shared Python deps for the whole repo
└─ README.md   
```

## Environment setup

### Prerequisites

- Python 3.9+ (recommended)  
- Access to a SLURM cluster (for distributed training)  
- **Git LFS** for large experiment artifacts (HTML/CSV plots, logs)

### Create venv
```
python3 -m universal-ner .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

In every .sh script be sure to check the path to venv! (For example look for 
```source /software/rome/r24.04/Miniconda3/24.7.1-0/bin/activate universal-ner```)

Create a .env file at the repo root (used by scripts that read environment variables):


```
HF_HOME=your/path/to/cache/huggingface
TRANSFORMERS_CACHE=/cache/huggingface/transformers
MLFLOW_TRACKING_URI='http://172.26.52.49'
MLFLOW_TRACKING_PASSWORD=password
MLFLOW_TRACKING_USERNAME="your-univ-email@mailbox.tu-dresden.de"
MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING=false
```

! TIP: if you’re on a cluster, place caches on fast local storage when possible.

## Typical workflows

### 1) Train GPT-2 (124M) with nanoGPT on SLURM
**What you get:** reproducible training runs (single or distributed), checkpoints, and logs.

**Run it**

```bash
# Non-distributed
sbatch gpt2/nanoGPT/slurm_job_scripts/no_distr_train.slurm

# Distributed (edit nodes/GPUs inside the script)
sbatch gpt2/nanoGPT/slurm_job_scripts/train_distributed.slurm
```

**Outputs**

- SLURM logs: gpt2/nanoGPT/slurm_job_scripts/*.out|*.err (or the path set in the scripts)

- Checkpoints/artifacts: as configured in the SLURM scripts

- (Optional) MLflow tracking if MLFLOW_* is set in .env

**Features**

Karpathy nanoGPT flow adapted for SLURM (single & multi-GPU)

Easy to tweak context length, batch size, tokens, LR schedule in the scripts

### 2) Find Super Weights (SW) in transformer models
What you get: location of “super weights” in MLP down_proj layers + impact on perplexity + diagnostic plots.

**Run it**

```
# From repo root
cd super_weights
# Edit find_sw.sh to set --model_name (supported: gpt2 or Qwen/Qwen2.5-3B)
sbatch find_sw.sh
```

**Outputs**

- Plots: super_weights/plot_outputs/*.png (down_proj in/out activations per block)

- SLURM logs: super_weights/slurm_out/*.out|*.err

- Printed SW coordinates and perplexity before/after zeroing one weight

**Requirements / tips**

Ensure HF cache exists or set via .env:

TRANSFORMERS_CACHE=/cache/huggingface/transformers

For GPT-2: pretrained weights are loaded onto the nanoGPT-style architecture via load_gpt_pretrained.py

**Features**

- Hooks into MLP down_proj across blocks

- Ranks channels by max |activation| to locate SW

- Fast perplexity check before/after zeroing the candidate weight

### 3) Lottery Winning Ticket Hypothesis (LWH) experiments
**What you get**: iterative prune-rewind-retrain loops, per-round metrics, and summary plots; tests whether SW survive pruning.

**Prepare**

Set the SW location used by your model inside the LWH code/scripts (see comments in func_train.sh / source).

Confirm .env (MLflow, caches) and the repo path inside the SLURM script.

**Run it**

```
# From repo root
cd LWH
sbatch func_train.sh
```

**Outputs**

- Per-run folder: LWH/experiment_results/<RUN_NAME>/

    - losses_ppl_round_*.csv, prune_stats_round_*.csv + .pt with checkpoints

    - interactive HTML snapshots (e.g., c_proj_box_round_*_iter_*.html)
  
- Summary PNGs: LWH/plots/*.png

- SLURM logs: LWH/slurm_out/*.out|*.err (if configured)

**Key knobs**

Rounds (iterations), pruning %, steps per round (max_iter), init_type (GPT-2 vs. Xavier), seed, LR schedule

Global vs. layerwise magnitude pruning 

**Features**

- Implements the classic LTH loop: train → prune → rewind → retrain (iterative or one-shot)

- Tracks SW survival across rounds and sparsity

- Exports CSVs + HTML dashboards for quick inspection

- Artifact handling (recommended)

    Large artifacts are tracked with Git LFS (*.html, *.csv already covered):
