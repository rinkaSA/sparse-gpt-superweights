#!/usr/bin/env python
"""
Iterative train-prune script for GPT.
"""
import sys
import os
import torch
import pickle
import argparse
from contextlib import nullcontext
root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root)
from model import GPTConfig, GPT
from torch.nn.parallel import DistributedDataParallel
from torch.distributed import init_process_group, destroy_process_group
import numpy as np
import math
import time
import copy
import torch.distributed as dist

def parse_args():
    p = argparse.ArgumentParser(description="Train GPT with optional DDP")
    # I/O
    p.add_argument('--out_dir', type=str, default='out')
    p.add_argument('--eval_interval', type=int, default=2000)
    p.add_argument('--log_interval', type=int, default=1)
    p.add_argument('--eval_iters', type=int, default=200)
    p.add_argument('--eval_only', action='store_true')
    p.add_argument('--always_save_checkpoint', action='store_true')
    p.add_argument('--init_from', type=str, default='scratch',
                   choices=['scratch', 'resume'] + ['gpt2', 'gpt2-medium', 'gpt2-large'])
    # wandb
    p.add_argument('--wandb_log', action='store_true')
    p.add_argument('--wandb_project', type=str, default='owt')
    p.add_argument('--wandb_run_name', type=str, default=None)
    # data
    p.add_argument('--dataset', type=str, default='data/openwebtext')
    p.add_argument('--gradient_accumulation_steps', type=int, default=20)
    p.add_argument('--batch_size', type=int, default=12)
    p.add_argument('--block_size', type=int, default=1024)
    # model
    p.add_argument('--n_layer', type=int, default=12)
    p.add_argument('--n_head', type=int, default=12)
    p.add_argument('--n_embd', type=int, default=768)
    p.add_argument('--dropout', type=float, default=0.0)
    p.add_argument('--bias', action='store_true')
    # optimizer
    p.add_argument('--learning_rate', type=float, default=6e-4)
    p.add_argument('--max_iters', type=int, default=5000)
    p.add_argument('--weight_decay', type=float, default=1e-1)
    p.add_argument('--beta1', type=float, default=0.9)
    p.add_argument('--beta2', type=float, default=0.95)
    p.add_argument('--grad_clip', type=float, default=1.0)
    # lr decay
    p.add_argument('--decay_lr', action='store_true')
    p.add_argument('--warmup_iters', type=int, default=2000)
    p.add_argument('--lr_decay_iters', type=int, default=600000)
    p.add_argument('--min_lr', type=float, default=6e-5)
    # system
    p.add_argument('--backend', type=str, default='nccl')
    p.add_argument('--device', type=str, default='cuda:0')
    p.add_argument('--dtype', type=str, default=None,
                   choices=['float32','bfloat16','float16'])
    p.add_argument('--compile', action='store_true')
    return p.parse_args()



def check_c_proj_entry(model, layer_idx, c_out, c_in):
    """
    Given a GPT model (or its DDP-wrapped version), check the
    (c_out, c_in) element of transformer.h.{layer_idx}.mlp.c_proj.weight.
    Return (value, is_nonzero).
    """
    # unwrap DDP if needed
    real_mod = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model

    submodule_name = f"transformer.h.{layer_idx}.mlp.c_proj"
    layer_module = real_mod.get_submodule(submodule_name)
    W = layer_module.weight   # shape: [out_dim, in_dim]

    out_dim, in_dim = W.shape
    if not (0 <= c_out < out_dim) or not (0 <= c_in < in_dim):
        raise IndexError(
            f"(c_out, c_in)=({c_out},{c_in}) out of bounds for shape {W.shape}"
        )

    val = W[c_out, c_in].item()
    return val, (val != 0.0)

def setup_distributed(args):
    """Initialize DDP if needed, return (ddp, ddp_rank, ddp_local_rank, world_size)."""
    ddp = int(os.environ.get('RANK', -1)) != -1
    if ddp:
        init_process_group(backend=args.backend)
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        torch.cuda.set_device(local_rank)
        device = f'cuda:{local_rank}'
        master_process = (rank == 0)
        # split gradient accumulation
        assert args.gradient_accumulation_steps % world_size == 0
        args.gradient_accumulation_steps //= world_size
        return ddp, rank, local_rank, world_size, device, master_process
    else:
        return ddp, 0, 0, 1, args.device, True


def get_batch(data_dir, split, block_size, batch_size, device, device_type):
    """Load a batch of data from memmapped bin files."""
    path = os.path.join(data_dir, f'{split}.bin')
    data = np.memmap(path, dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i+block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i+1:i+1+block_size].astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss(model, get_batch_fn, ctx, eval_iters):
    losses = {}
    model.eval()
    for split in ['train', 'val']:
        vals = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch_fn(split)
            with ctx:
                _, loss = model(X, Y)
            vals[k] = loss.item()
        losses[split] = vals.mean()
    model.train()
    return losses


def get_lr(it, args):
    if not args.decay_lr:
        return args.learning_rate
    if it < args.warmup_iters:
        return args.learning_rate * (it + 1) / (args.warmup_iters + 1)
    if it > args.lr_decay_iters:
        return args.min_lr
    # cosine decay
    decay_ratio = (it - args.warmup_iters) / (args.lr_decay_iters - args.warmup_iters)
    coeff = 0.5 * (1 + math.cos(math.pi * decay_ratio))
    return args.min_lr + coeff * (args.learning_rate - args.min_lr)

def build_model(args, device):
    # set up model args
    model_args = dict(
        n_layer=args.n_layer, n_head=args.n_head,
        n_embd=args.n_embd, block_size=args.block_size,
        bias=args.bias, dropout=args.dropout,
        vocab_size=50304
    )
    if args.init_from == 'resume':
        checkpoint = torch.load(os.path.join(args.out_dir, 'ckpt.pt'),
                                map_location=device)
        # enforce consistency
        for k in ['n_layer','n_head','n_embd','block_size','bias','vocab_size']:
            model_args[k] = checkpoint['model_args'][k]
        model = GPT(GPTConfig(**model_args))
        sd = checkpoint['model']
        # strip unwanted prefixes
        for k in list(sd.keys()):
            if k.startswith('_orig_mod.'):
                sd[k[len('_orig_mod.'):]] = sd.pop(k)
        model.load_state_dict(sd)
    elif args.init_from.startswith('gpt2'):
        model = GPT.from_pretrained(args.init_from, override_args=dict(dropout=args.dropout))
        for k in ['n_layer','n_head','n_embd','block_size','bias','vocab_size']:
            model_args[k] = getattr(model.config, k)
    else:
        model = GPT(GPTConfig(**model_args))
    # crop block size if smaller
    if args.block_size < model.config.block_size:
        model.crop_block_size(args.block_size)
    model.to(device)
    return model, model_args

def init_mask(model):
    """
    Create an all-ones mask for every trainable parameter in model.
    """
    mask = {}
    for name, p in model.named_parameters():
        if p.requires_grad:
            mask[name] = torch.ones_like(p.data)
    return mask

def apply_mask(model, mask):
    """
    Zero out masked weights and block their gradients.
    Must be called before training (and after every optimizer step if you want hard enforcement).
    """
    for name, p in model.named_parameters():
        if name in mask:
            # zero-out any stray non-zero entries
            p.data.mul_(mask[name])
            # attach a hook so grads at masked positions stay zero
            p.register_hook(lambda grad, m=mask[name]: grad.mul(m))

def is_master():
    return (not dist.is_available()
            or not dist.is_initialized()
            or dist.get_rank() == 0)

def prune_weights(model, mask, prune_percent, debug=False):
    """
    Globally prune the smallest `prune_percent` of STILL-UNMASKED weights by magnitude.
    If debug=True, prints:
      - which param names the model has vs. which the mask has
      - for each param, how many unmasked entries
      - global stats: total unmasked, k, threshold, sample values
    Returns an updated mask dict.
    """
    # If DDP-wrapped, work on the real module
    real_mod = model.module if hasattr(model, 'module') else model

    # DEBUG: compare parameter names ↔ mask keys
    if debug and is_master():
        param_names = [n for n,_ in real_mod.named_parameters()]
        mask_keys   = list(mask.keys())
        missing     = set(param_names) - set(mask_keys)
        extra       = set(mask_keys)   - set(param_names)
        print(f"[prune][debug] model has {len(param_names)} params, mask has {len(mask_keys)} entries")
        print(f"[prune][debug] params missing from mask ({len(missing)}): {sorted(list(missing))[:5]}…")
        print(f"[prune][debug] mask keys not in model ({len(extra)}): {sorted(list(extra))[:5]}…\n")

    # 1) collect all still-unmasked absolute values
    all_vals = []
    for name, p in real_mod.named_parameters():
        if name not in mask:
            if debug:
                if is_master():
                    print(f"[prune][debug] skipping {name}, not in mask")
            continue
        m = mask[name].bool()
        n_unmasked = int(m.sum().item())
        if debug:
            if is_master():
                print(f"[prune][debug] param {name}: {n_unmasked}/{p.numel()} unmasked")
        if n_unmasked > 0:
            vals = p.data.abs()[m].view(-1)
            all_vals.append(vals)

    if not all_vals:
        if debug:
            if is_master():
                print("[prune][debug] NO unmasked values found → skipping pruning\n")
        return mask

    # 2) flatten & compute threshold
    all_flat = torch.cat(all_vals)
    total    = all_flat.numel()
    k        = int(prune_percent/100 * total)
    if debug:
        if is_master():
            print(f"[prune][debug] total unmasked vals = {total}, k = {k} ({prune_percent}%)")
    if k <= 0:
        if debug:
            if is_master():
                print("[prune][debug] k<=0 → nothing to prune\n")
        return mask

    # 3) find threshold & sample
    thresh = torch.kthvalue(all_flat, k).values.item()
    if debug:
        smallest = torch.topk(all_flat, k, largest=False).values[:min(10,k)].cpu().tolist()
        if is_master():
            print(f"[prune][debug] threshold = {thresh:.6e}")
            print(f"[prune][debug] sample smallest: {smallest}\n")

    # 4) apply threshold to rebuild masks
    for name, p in real_mod.named_parameters():
        if name not in mask:
            continue
        m_old = mask[name].bool()
        m_new = (p.data.abs() > thresh) & m_old
        kept  = int(m_new.sum().item())
        if debug:
            if is_master():
                print(f"[prune][debug] {name}: kept {kept}/{int(m_old.sum().item())}")
        mask[name] = m_new.to(mask[name].dtype)

    if debug:
        total_after = sum(int(m.sum().item()) for m in mask.values())
        if is_master():
            print(f"[prune][debug] remaining unmasked after prune: {total_after}/{total}\n")

    return mask


def save_mask(mask, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(mask, path)

def load_mask(path, map_location=None):
    return torch.load(path, map_location=map_location)

def train_one_round(model, optimizer, scaler, args, get_batch_fn, ctx, device):
    """
    Run exactly args.max_iters training iterations on `model`.
    Returns nothing (model / optimizer / scaler are mutated in place).
    """
    iter_num = 0
    best_val = float('inf')
    raw = model.module if hasattr(model, 'module') else model

    # fetch first batch
    X, Y = get_batch_fn('train')
    t0 = time.time()
    local_iter = 0
    running_mfu = -1.0

    while iter_num < args.max_iters:
        # 1) set LR
        lr = get_lr(iter_num, args)
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        # 2) eval + checkpoint
        if iter_num % args.eval_interval == 0 and args.master:
            losses = estimate_loss(model, get_batch_fn, ctx, args.eval_iters)
            print(f"[round] step {iter_num}: train {losses['train']:.4f}, val {losses['val']:.4f}", flush=True)
            if losses['val'] < best_val or args.always_save_checkpoint:
                best_val = losses['val']
                ck = dict(
                  model=raw.state_dict(),
                  optimizer=optimizer.state_dict(),
                  iter_num=iter_num,
                  best_val_loss=best_val,
                )
                torch.save(ck, os.path.join(args.out_dir, 'ckpt.pt'))

        # 3) forward/backward with grad-accum
        optimizer.zero_grad(set_to_none=True)
        for micro in range(args.gradient_accumulation_steps):
            if args.ddp:
                model.require_backward_grad_sync = (micro == args.gradient_accumulation_steps - 1)
            with ctx:
                logits, loss = model(X, Y)
                loss = loss / args.gradient_accumulation_steps
            X, Y = get_batch_fn('train')
            scaler.scale(loss).backward()
        if args.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer); scaler.update()

        # 4) logging
        dt = time.time() - t0; t0 = time.time()
        if iter_num % args.log_interval == 0 and args.master:
            lossf = loss.item() * args.gradient_accumulation_steps
            if local_iter >= 5:
                mfu = raw.estimate_mfu(args.batch_size * args.gradient_accumulation_steps, dt)
                running_mfu = mfu if running_mfu < 0 else 0.9*running_mfu + 0.1*mfu
            print(f"[round] iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.1f}ms, mfu {running_mfu*100:.1f}%", flush=True)

        iter_num += 1
        local_iter += 1


SW_LAYER = 2
SW_C_OUT  = 447
SW_C_IN   = 666


def main():
    args = parse_args()
    args.ddp, args.rank, args.local_rank, args.world_size, args.device, args.master = setup_distributed(args)
    torch.manual_seed(1337 + (args.rank if args.ddp else 0))
    if args.master:
        os.makedirs(args.out_dir, exist_ok=True)

    device_type = 'cuda' if 'cuda' in args.device else 'cpu'
    if args.dtype is None:
        args.dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
    ptd = dict(float32=torch.float32, bfloat16=torch.bfloat16, float16=torch.float16)[args.dtype]
    ctx = torch.amp.autocast(device_type=device_type, dtype=ptd) if device_type!='cpu' else nullcontext()

    model, _ = build_model(args,args.device)
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype=='float16'))
    optimizer = model.configure_optimizers(args.weight_decay, args.learning_rate,
                                           (args.beta1, args.beta2), device_type)
    if args.init_from == 'resume':
        ck = torch.load(os.path.join(args.out_dir,'ckpt.pt'), map_location=args.device)
        optimizer.load_state_dict(ck['optimizer'])
    if args.compile:
        model = torch.compile(model)
    if args.ddp:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])

    raw_model = model.module if hasattr(model, 'module') else model
    init_state = copy.deepcopy(raw_model.state_dict())

    mask_path = os.path.join(args.out_dir, 'mask10pr10000.pt')
    #if os.path.exists(mask_path):
    #     mask = load_mask(mask_path, map_location=args.device)
    #else:
    mask = init_mask(raw_model)

    initial_unmasked = sum(int(m.sum().item()) for m in mask.values())
    if args.master:
        print(f"[debug] initial mask {initial_unmasked}")

    get_batch_fn = lambda split: get_batch(args.dataset, split,
                                           args.block_size, args.batch_size,
                                           args.device, device_type)
    canonical_mask = {}
    for k, v in mask.items():
        if k.startswith("module."):
            new_k = k[len("module."):]
        else:
            new_k = k
        canonical_mask[new_k] = v.to(device_type)
    mask = canonical_mask
    initial_unmasked = sum(int(m.sum().item()) for m in mask.values())
    if args.master:
            print(f"[debug] initial canoical mask {initial_unmasked}")
    for round_idx in range(1, 6):
        if args.master:
            print(f"\n=== ROUND {round_idx}/5 ===", flush=True)
        if round_idx == 1 and args.master:
            total_params = sum(p.numel() for p in raw_model.parameters())
            initial_unmasked = sum(int(m.sum().item()) for m in mask.values())
            print(f"[debug] Round {round_idx} starting: {initial_unmasked}/{total_params} ({100*initial_unmasked/total_params:.2f}%) unmasked")

        if round_idx > 1:
            raw_model.load_state_dict(init_state)
            raw_model.train()  
            optimizer = raw_model.configure_optimizers(
                args.weight_decay, args.learning_rate, (args.beta1, args.beta2), device_type
            )
            scaler    = torch.cuda.amp.GradScaler(enabled=(args.dtype=='float16'))

        # now apply the cumulative mask and train
        apply_mask(raw_model, mask)
        train_one_round(model, optimizer, scaler, args, get_batch_fn, ctx, args.device)

        # prune & report
        mask = prune_weights(raw_model, mask, prune_percent=10, debug=True)
        val, nonzero = check_c_proj_entry(model, SW_LAYER, SW_C_OUT, SW_C_IN)
        status = "non-zero" if nonzero else "ZEROED OUT"
        if args.master:
            print(f"[monitor] … → {status}", flush=True)

        total = sum(m.numel() for m in mask.values())
        remain = sum(int(m.sum().item()) for m in mask.values())
        if args.master:
            print(f"[sparsity] {remain}/{total} non-zero → {(100*remain/total):.2f}%", flush=True)

        save_mask(mask, mask_path)

    if args.ddp:
        destroy_process_group()

if __name__ == '__main__':
    main()
