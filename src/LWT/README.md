# This is a subfolder for Lottery Winning Tiket Hypothesis implementation using GPT2

Prerequisite of this repo is ```requiremeents.txt``` installation.

The location of Super Weights of the current model is hard-coded (global vars in lines 369-371). Therefore, ```/src/find_sw.sh``` should be run before hand and the output in ```super_weights/src/slurm_sw_out/checkmodel_{job_id}.out``` analyzed.

Run the training of LWH with the following command: ```sbatch func_train.sh```

## Some of the first results analysis

The number of pruning rounds is static - 5 rounds. Varying parameters - pruning level and number of iteration per each round training.

- Training lth_hf_out_876812.out - each training with 10 steps only (created for debugging) and 20% pruning shows on the second round at the sparsity level 64% that SW is already pruned (zeroed out). Not a significant or worrying result, but interesting to see how LWH behaves at low level of training.

- Trainig lth_hf_out_876819.out - 10000  steps and 20% prune level. Reached 32.77 % sparsity level. SW is not pruned!

- Training lth_hf_out_876823.out - 10000 steps and 10% prune level. Reached 59.05% sparsity level. SW is pruned... How?

Why at 20 % prune level SW is kept, but at 10% - no?

Pruning 20 % per round makes the network sparser faster, which tends to amplify the surviving weights more during training. Pruning 10 % per round keeps more connections longer, so that particular weight may see less gradient signal and end up smaller relative to its peers.

10% prune : threshold  = 4.807960e-03 (5th round) 
20% prune : threshold  = 9.359073e-03 (5th round) 


- training lth_hf_out_877713.log - 3000 steps, 10% prune, with boxplots for c_proj weight matrix for layer 2 and SW value (output directory [text](../../plots_cproj_lwh_train_10)) -> zeroed out at 40.96% sparcity 
- trianing lth_out_877756.log - 3000 steps, 20% prune, with boxplots for c_proj weight matrix for layer 2 ([text](../../plots_cproj_lwh_train_20))
    Zeroed out SW on very high sparsity level 81.00%!! -> we should really train more???

