# This repo containes 3 subtasks - Super Weight Analysis, nanoGPT2 124M train, and Lottery Winnig Ticket Hypothesis.

## Structure
```super_weights/
├── gpt2/nanoGPT/   -->> Karpaty script adapted for SLURM job. Go there for distributed training. Refer to 2 Readme`s in that folder!
|    └── /slurm_jobs_scripts/ -> run stuff
├── src/            -->> Super Weight location scripts are here
|    └── /LWT        --->> Lottery Winning Hypothesis for GPT2 model with check of preservation of previously found location of Super Weight!
├── logs/       
├── output/                          
├── run_app.py                 
├── job_run.sh       
├── requirements.txt  --> works for anything in this repo!
└── README.md
```


 REFER TO REEADME`S IN EVERY SUBDIR DEDICATED TO SOME OF THE TASKS! (/gpt2/nanoGPT/ README.md + README_iryna.md + /src/README.md + src/LWH/README.md)

