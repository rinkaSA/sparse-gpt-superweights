# This repoaitory containes 3 subtasks - Super Weight Analysis, nanoGPT2 124M train, and Lottery Winnig Ticket Hypothesis.

## Structure
```super_weights/
├── gpt2/nanoGPT/      -->> Karpaty script adapted for SLURM job. Go there for distributed training. Refer to 2 Readme`s in that folder!
├── super_weights/      -->> Super Weight location scripts are here
├──-LWT/               --->> Lottery Winning Hypothesis for GPT2 model with check of preservation of previously found location of Super Weight!   
├── requirements.txt  --> works for anything in this repo!
└── README.md
```

## Environment

- create .env file with 
```
HF_HOME=your/path/to/cache/huggingface
TRANSFORMERS_CACHE=/cache/huggingface/transformers
MLFLOW_TRACKING_URI='http://172.26.52.49'
MLFLOW_TRACKING_PASSWORD=password
MLFLOW_TRACKING_USERNAME="your-univ-email@mailbox.tu-dresden.de"
MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING=false
```
- use requirements.txt to create a venv universal-ner


**REFER TO README IN EVERY SUBDIR DEDICATED TO EACH TASK! (/gpt2/nanoGPT/ README.md + README_iryna.md + /super_weights/README.md + /LWH/README.md)**

