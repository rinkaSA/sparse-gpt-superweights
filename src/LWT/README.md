# This is a subfolder for Lottery Winning Tiket Hypothesis implementation using GPT2

Prerequisite of this repo is ```requiremeents.txt``` installation.

The location of Super Weights of the current model is hard-coded (global vars in lines 369-371). Therefore, ```/src/find_sw.sh``` should be run before hand and the output in ```super_weights/src/slurm_sw_out/checkmodel_{job_id}.out``` analyzed.

Run the training of LWH with the following command: ```sbatch func_train.sh```

## Some of the first results analysis

The number of pruning rounds is static - 5 rounds. Varying parameters - pruning level and number of iteration per each round training.

- Training lth_hf_out_876812.out - each training with 10 steps only (created for debugging) and 20% pruning shows on the second round at the sparsity level 64% that SW is already pruned (zeroed out). Not a sugnificant or worrying results, but interesting to see how LWH behaves at low level of training.

- Trainig lth_hf_out_876819.out - 10000  steps and 20% prune level. Reached 32.77 % sparsity level. SW is not pruned!

- Training lth_hf_out_876823.out - 10000 steps and 10% prune level. Reached 59.05% sparsity level. SW is pruned... How?

