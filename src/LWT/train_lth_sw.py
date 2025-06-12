
import os
import time
import math
import pickle
import copy
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from contextlib import nullcontext
import numpy as np

import os, sys


root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..', 'gpt2', 'nanoGPT'))
sys.path.append(root)

from model import GPTConfig, GPT

prune_rounds       = 2      
prune_fraction     = 0.20    
subloop_max_steps  = 50   
eval_every_steps   = 1000  
dataset            = 'openwebtext'
data_dir           = f"data/{dataset}"
batch_size         = 12         
gradient_accum_steps = 5 * 8
block_size         = 1024

# Model architecture (must match what you want to prune)
model_args = {
    'n_layer': 12,
    'n_head': 12,
    'n_embd': 768,
    'block_size': block_size,
    'bias': False,
    'vocab_size': None,  
    'dropout': 0.0,
}

learning_rate      = 6e-4
weight_decay       = 1e-1
beta1              = 0.9
beta2              = 0.95
decay_lr           = True
warmup_iters       = 2000
lr_decay_iters     = 600000
min_lr             = 6e-5


backend            = 'nccl'  # assumes GPUs + NCCL
dtype_str          = None    # we’ll infer below


def get_batch(split):
    """
    Exactly the same "poor-man's data loader" as before.
    """
    data = np.memmap(os.path.join(data_dir, f"{split}.bin"), dtype=np.uint16, mode='r')
    ix   = torch.randint(len(data) - block_size, (batch_size,), device='cpu')
    x    = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y    = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        return x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        return x.to(device), y.to(device)

@torch.no_grad()
def estimate_loss(model, eval_iters=200):
    """
    Run eval on train/val exactly as in train_refactored.py, but
    applied to the DDP-wrapped `model`.
    """
    model_to_eval = model.module if isinstance(model, DDP) else model

    out = {}
    model_to_eval.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters, device=device)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model_to_eval(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model_to_eval.train()
    return out

def get_lr(it):
    """
    Cosine with warmup, exactly as in train_refactored.py
    """
    if not decay_lr:
        return learning_rate
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)

def create_mask(model, fraction):
    """
    Build a single-round mask that zeros out the bottom `fraction` of weights
    by absolute magnitude, for each parameter. Returns a {name: mask_tensor} dict.
    """
    new_masks = {}
    with torch.no_grad():
        for name, param in model.named_parameters():
            if not param.requires_grad:
                new_masks[name] = torch.ones_like(param, device=device)
                continue
            flat = param.abs().view(-1)
            k    = int(flat.numel() * fraction)
            if k == 0:
                new_masks[name] = torch.ones_like(param, device=device)
                continue
            thresh, _ = torch.kthvalue(flat, k)
            new_masks[name] = (param.abs() > thresh).to(device)
    return new_masks

def apply_mask(model, mask_dict):
    """
    Multiply each parameter by its mask in place.
    """
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in mask_dict:
                param.data.mul_(mask_dict[name])

def check_c_proj_entry(model, layer_idx, c_out, c_in):
    """
    Given a GPT model (or its DDP wrapped version), check the (c_out, c_in) element
    of transformer.h.{layer_idx}.mlp.c_proj.weight. Return (value, is_nonzero).
    """
    submodule_name = f"transformer.h.{layer_idx}.mlp.c_proj"
    real_mod = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model

    layer_module = real_mod.get_submodule(submodule_name)
    W = layer_module.weight           # shape: [out_dim, in_dim]

    out_dim, in_dim = W.shape
    if not (0 <= c_out < out_dim) or not (0 <= c_in < in_dim):
        raise IndexError(
            f"Requested (c_out, c_in)=({c_out}, {c_in}) out of bounds for shape {W.shape}"
        )

    value = W[c_out, c_in].item()
    is_nonzero = (value != 0.0)
    return value, is_nonzero


def broadcast_mask(mask_dict):
    """
    Helper to broadcast a Python-level dict of CPU tensors from rank 0 → all ranks.
    We use torch.distributed.broadcast_object_list for convenience.
    """
    # turn the dict into a list of key‐tensor pairs on rank 0
    if rank == 0:
        obj_list = [(k, mask_dict[k].cpu()) for k in mask_dict]
    else:
        # placeholder
        obj_list = [('', torch.empty_like(next(iter(mask_dict.values())).cpu())) for _ in range(len(mask_dict))]
    dist.broadcast_object_list(obj_list, src=0)
    #  Reconstruct a same‐ordered dict on every rank
    out = {}
    for k, t in obj_list:
        out[k] = t.to(device)
    return out

def distributed_train_subloop(model, optimizer, max_steps, cumulative_mask):
    """
    Run `max_steps` of DDP training on `model` (which is already wrapped in DDP).
    After every optimizer.step(), reapply `cumulative_mask` so pruned weights stay zero.
    We do a quick eval every `eval_every_steps` steps.
    Returns the best validation loss seen in this subloop and the corresponding state-dict (rank 0).
    """
    best_val_loss = float('inf')
    best_state    = None

    step = 0
    t0   = time.time()

    while step < max_steps:
        lr = get_lr(step)
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        X, Y = get_batch('train')
        with ctx:
            logits, loss = model(X, Y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()


        #  Re‐apply mask on every parameter!!
        apply_mask(model.module, cumulative_mask)

        if step % eval_every_steps == 0:
            losses = estimate_loss(model, eval_iters=200)
            dist.barrier(device_ids=[local_rank])  
            if rank == 0:
                print(f"    [DDP Subloop] step {step:5d}: train {losses['train']:.4f}, val {losses['val']:.4f}")
                if losses['val'] < best_val_loss:
                    best_val_loss = losses['val']
                    # only rank 0 needs to save the state-dict; other ranks can skip
                    best_state = copy.deepcopy(model.module.state_dict())
            dist.barrier(device_ids=[local_rank]) 
        step += 1

    if rank == 0:
        print(f"    subloop done in {time.time()-t0:.1f}s, best val = {best_val_loss:.4f}")
    dist.barrier()
    return best_val_loss, best_state

def broadcast_state_dict(state_dict, src=0):
    """
    Broadcasts state dict from src rank to all other ranks tensor by tensor
    """
    if dist.get_rank() == src:
        # Source rank has the state dict
        keys = list(state_dict.keys())
    else:
        # Other ranks initialize empty dict
        state_dict = {}
        keys = [None]
    
    if dist.get_rank() != src:
        keys = [None]  # placeholder
    dist.broadcast_object_list(keys, src=src)

    for key in keys:
        if dist.get_rank() != src:
            state_dict[key] = torch.empty_like(init_model.state_dict()[key])
        dist.broadcast(state_dict[key], src=src)
    
    return state_dict

if __name__ == "__main__":
    dist.init_process_group(backend=backend)
    rank       = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])
    device     = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)
    #master_process = rank == 0
    seed_offset  =  rank


     # was defined separately by running find_sw.sh
    SW_layer = 2
    SW_c_out  = 447
    SW_c_in   = 666
    

    global device_type, ctx
    device_type = 'cuda' if device.type == 'cuda' else 'cpu'
    if device_type == 'cuda' and torch.cuda.is_bf16_supported():
        dtype_str = 'bfloat16'
    elif device_type == 'cuda':
        dtype_str = 'float16'
    else:
        dtype_str = 'float32'
    torch.manual_seed(1337 + seed_offset)
    torch.backends.cudnn.allow_tf32 = True 
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype_str]
    ctx     = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    compile = True

    meta_path = os.path.join(data_dir, 'meta.pkl')
    if os.path.exists(meta_path):
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        model_args['vocab_size'] = meta['vocab_size']
        if rank == 0:
            print(f"found vocab_size = {meta['vocab_size']} from {meta_path}")
    else:
        model_args['vocab_size'] = 50304
        if rank == 0:
            print("defaulting to vocab_size = 50304")

    gptconf    = GPTConfig(**model_args)
    init_model = GPT(gptconf).to(device)
    if compile:
        print("compiling the model... (takes a ~minute)")
        unoptimized_model = init_model
        init_model = torch.compile(init_model)
        dist.barrier(device_ids=[local_rank])


    if rank == 0:
        init_state = copy.deepcopy(init_model.state_dict())
    else:
        init_state = None

    dist.barrier(device_ids=[local_rank])
    init_state = broadcast_state_dict(init_state, src=0)
    dist.barrier(device_ids=[local_rank])

    init_model.load_state_dict(init_state)
    dist.barrier(device_ids=[local_rank])
    model = DDP(init_model, device_ids=[local_rank], output_device=local_rank)

    cumulative_mask = {}
    with torch.no_grad():
        for name, param in model.module.named_parameters():
            cumulative_mask[name] = torch.ones_like(param, device=device)

    #  LTH Loop 
    best_overall_val = float('inf')
    best_overall_state = None

    for round_idx in range(prune_rounds):
        if rank == 0:
            print("\n" + "="*60)
            print(f" Pruning Round {round_idx+1}/{prune_rounds}")
            # rank 0 loads init_state into a fresh CPU‐model (or reuse a GPU copy)
            tmp_model = GPT(gptconf).to(device)
            tmp_model.load_state_dict(init_state)
            apply_mask(tmp_model, cumulative_mask)

            new_mask = create_mask(tmp_model, prune_fraction)
            for name in cumulative_mask:
                cumulative_mask[name] = cumulative_mask[name] * new_mask[name]

            # broadcast the updated cumulative_mask to all ranks
            mask_to_bcast = {k: cumulative_mask[k].cpu() for k in cumulative_mask}
            dist.barrier(device_ids=[local_rank])
            mask_to_bcast = broadcast_mask(mask_to_bcast)  # everyone now has the same tensor on each device
            dist.barrier(device_ids=[local_rank])
            cumulative_mask = {k: mask_to_bcast[k].to(device) for k in mask_to_bcast}

            del tmp_model

        else:
            # Ranks >0 just receive the new cumulative_mask via broadcast_mask()
            # This call is already made on rank 0, so rank >0 will block in broadcast_mask.
            dummy_dict = {}
            dummy_dict = broadcast_mask(dummy_dict)  # just to sync shapes; we overwrite next
            # Now cumulative_mask is reconstructed in every rank’s broadcast_mask call
            cumulative_mask = {k: dummy_dict[k].to(device) for k in dummy_dict}

        model.module.load_state_dict(init_state)       # rewind weights
        apply_mask(model.module, cumulative_mask)       # mask out pruned weights
        model.train()

        sw_loc, sw_val = check_c_proj_entry(model, SW_layer, SW_c_out, SW_c_in)
        if rank == 0:
            print(f"   Super weight before round {round_idx+1}: {sw_val:.4f} at {sw_loc}")

        dist.barrier(device_ids=[local_rank])
        optimizer = model.module.configure_optimizers(
            weight_decay, learning_rate, (beta1, beta2), device_type
        )

        sub_best_val, sub_best_state = distributed_train_subloop(
            model, optimizer, subloop_max_steps, cumulative_mask
        )
        # rank 0 can compare & keep the best‐validation round model
        if rank == 0 and sub_best_val < best_overall_val:
            best_overall_val   = sub_best_val
            best_overall_state = copy.deepcopy(sub_best_state)

       
        val_before, nonzero_before = check_c_proj_entry(
            model, SW_layer, SW_c_out, SW_c_in
        )
        if rank == 0:
            status = "nonzero" if nonzero_before else "zero"
            print(f"   c_proj[{SW_layer}].weight[{SW_c_out},{SW_c_in}] before retrain: {val_before:.6f} ({status})")

        #  weight‐sparsity % 
        if rank == 0:
            total = sum(m.numel() for m in cumulative_mask.values())
            remain = sum(m.sum().item() for m in cumulative_mask.values())
            print(f"   Round {round_idx+1} → sparsity: {(1 - remain/total):.2%} | remaining: {remain/total:.2%}")



    # Save the Winning Ticket Model
    if rank == 0:
        print("\n" + "="*60)
        print(f"All {prune_rounds} rounds complete. Best val = {best_overall_val:.4f}")
        final_model = GPT(gptconf).to(device)
        final_model.load_state_dict(best_overall_state)

        torch.save({
            'model_state_dict': final_model.state_dict(),
            'masks': {k: cumulative_mask[k].cpu() for k in cumulative_mask}
        }, "lth_ddp_winning_ticket.pt")
        print("Saved → lth_ddp_winning_ticket.pt")

    dist.barrier(device_ids=[local_rank])
    dist.destroy_process_group()
