import os
import sys
from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import load_dataset
import torch
from torch.utils.data import DataLoader
import argparse

# number of workers in .map() call
num_proc = 8
num_proc_load_dataset = num_proc

def process(example):
    enc = tiktoken.get_encoding("gpt2")
    ids = enc.encode_ordinary(example['text'])
    ids.append(enc.eot_token)
    out = {'ids': ids, 'len': len(ids)}
    return out

def process_shard(rank, world_size, split_dataset, output_dir):
    # Process only this shard's portion of the dataset
    tokenized = split_dataset.map(
        process,
        remove_columns=['text'],
        desc=f"tokenizing shard {rank}",
        num_proc=num_proc
    )

    # Process each split
    for split, dset in tokenized.items():
        # Calculate total length for this split
        arr_len = np.sum(dset['len'], dtype=np.uint64)
        
        # Calculate shard size and offsets
        shard_size = arr_len // world_size
        start_idx = rank * shard_size
        end_idx = start_idx + shard_size if rank < world_size - 1 else arr_len
        
        # Create filename for this shard
        shard_filename = os.path.join(output_dir, f'{split}_shard_{rank}.bin')
        
        # Create memmap file for this shard
        dtype = np.uint16
        arr = np.memmap(shard_filename, dtype=dtype, mode='w+', shape=(end_idx - start_idx,))
        
        # Process this shard's portion
        total_batches = 256  # Reduced number of batches per shard
        shard_dset = dset.select(range(start_idx, end_idx))
        
        idx = 0
        for batch_idx in tqdm(range(total_batches), desc=f'writing {shard_filename}'):
            batch = shard_dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
            arr_batch = np.concatenate(batch['ids'])
            arr[idx : idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        arr.flush()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rank', type=int, required=True)
    parser.add_argument('--world-size', type=int, required=True)
    args = parser.parse_args()

    # Create output directory for shards
    output_dir = os.path.join(os.path.dirname(__file__), 'shards')
    os.makedirs(output_dir, exist_ok=True)

    # Load dataset
    dataset = load_dataset("openwebtext", num_proc=num_proc_load_dataset, trust_remote_code=True)
    
    # Create train/val split
    split_dataset = dataset["train"].train_test_split(test_size=0.0005, seed=2357, shuffle=True)
    split_dataset['val'] = split_dataset.pop('test')

    # Process this shard
    process_shard(args.rank, args.world_size, split_dataset, output_dir)

if __name__ == '__main__':
    main()
