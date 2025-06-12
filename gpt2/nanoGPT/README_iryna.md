## This a README about changes to the  original nanoGPT repo by  Karpaty

All the changes are made to make the ```train.py``` run on the HPC cluster.

Therefore ```/slurm_job_scripts``` subdir was added.

```cd /slurm_job_scripts```

### 1. Start with ```sbatch prepare_data.slurm``` to load and process openweb data set.
We are going to train on it. This will surely take some time to run!!!
### 2. After all the necessary files have arrived (refer to original README.md in this dir) we are now ready for the training..
### 3. Training...
In the train script to match 16 GPUs ( define how you want is in ```train_distributed.slurm```) we change ```gradient_accumulation_steps = 5 * 8 # used to simulate larger batch sizes``` from *8 to *16 . It is already pushed with this value.

Be sure to install all the requirements before this step ( /superweights/requirements.txt). The slurm job script was created based on the ZIH Compendium of how to use Distributed Data Parallel for training on multiple GPUs on the cluster.

The only configuration argument redefined is max_iter, which is set to 10000. Originally, the training happend for 600k or smth, I have tried 22k (and logged the loss results together with out into /visualize_loss and /train_out respectively) and it took around 2 hours to run (why not faster though...). For now 10000 will be enough for some kind of convergence.

Train has 9B tokens, per optimization step we pass 983k  (gradient accum * batch size * context lenth) of tokens ( + multiply by number of gpus) ->> 572 steps for one pass trough the whole dataset (epoch) - > 10k steps is roughly 17 epochs.

After being all settled with parameters, run :

```sbatch train_distributed.slurm```

Should take one hour max.