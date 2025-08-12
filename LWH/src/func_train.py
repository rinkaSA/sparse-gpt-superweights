#!/usr/bin/env python
import sys
import os
import torch
import pickle
import argparse
from contextlib import nullcontext
#root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
#sys.path.append(root)
from gpt2.nanoGPT.model import GPTConfig, GPT
from torch.nn.parallel import DistributedDataParallel
from torch.distributed import init_process_group, destroy_process_group
import numpy as np
import math
import time
import copy
import torch.distributed as dist
import plotly.graph_objects as go
import mlflow
import mlflow.pytorch
import pandas as pd
from LWH.src.visualizer import plot_round_metrics_plotly

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
# Define the base path for all output files relative to the script or working directory
BASE_OUTPUT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiment_results")


def parse_args():
    p = argparse.ArgumentParser(description="Train GPT with optional DDP")
    # I/O
    p.add_argument('--out_dir', type=str, default="Out_dir")
    p.add_argument('--eval_interval', type=int, default=2000) #was 2000
    p.add_argument('--log_interval', type=int, default=1)
    p.add_argument('--eval_iters', type=int, default=200) # was 200
    p.add_argument('--eval_only', action='store_true')
    p.add_argument('--always_save_checkpoint', action='store_true')
    p.add_argument('--init_from', type=str, default='scratch',
                   choices=['scratch', 'resume'] + ['gpt2', 'gpt2-medium', 'gpt2-large'])
    # wandb
    p.add_argument('--wandb_log', action='store_true')
    p.add_argument('--wandb_project', type=str, default='owt')
    p.add_argument('--wandb_run_name', type=str, default=None)
    # data
    p.add_argument('--dataset', type=str, default='gpt2/nanoGPT/data/openwebtext')
    p.add_argument('--gradient_accumulation_steps', type=int, default=5*4) #CHANGE THE LEFT VALUE BASED ON NUMBER OF GPUs
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
    p.add_argument('--max_iters', type=int, default=3000)
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
    p.add_argument('--sw_layer', type=int, default=2)
    p.add_argument('--sw_c_out', type=int, default=447)
    p.add_argument('--sw_c_in', type=int, default=666)
    p.add_argument('--init_type', type=str, default='xavier')
    p.add_argument('--prune_percent', type=float, default=10)
    
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

def build_model(args, device, init_type='xavier'):
    # set up model args
    model_args = dict(
        n_layer=args.n_layer, n_head=args.n_head,
        n_embd=args.n_embd, block_size=args.block_size,
        bias=args.bias, dropout=args.dropout,
        vocab_size=50304
    )
    if args.init_from == 'resume':
        checkpoint = torch.load(os.path.join(BASE_OUTPUT_PATH, args.out_dir, 'ckpt.pt'),
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
        # XAVIER initialization for weights
        if init_type == 'xavier':
            #mlflow.log_param("init_type", "xavier")
            for name, param in model.named_parameters():
                if 'weight' in name:
                    if len(param.shape) >= 2:  # For weight matrices
                        torch.nn.init.xavier_uniform_(param.data)
                    else: 
                        torch.nn.init.zeros_(param.data)
                elif 'bias' in name:
                    torch.nn.init.zeros_(param.data)
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

def plot_c_proj_box(model, round_idx, iter_num, args_out_dir, SW_C_OUT, SW_C_IN):
    """
    Creates and displays a Plotly box plot of the transformer.h.2.mlp.c_proj weights
    at a given round and training iteration.
    """
    real = model.module if hasattr(model, 'module') else model
    
    layer = real.get_submodule("transformer.h.2.mlp.c_proj")

    W_matrix = layer.weight
    superweight = W_matrix[SW_C_OUT, SW_C_IN].item()

    W = layer.weight.detach().cpu().numpy().flatten()
    
    fig = go.Figure(go.Box(
        y=W,
        boxpoints="outliers", 
    ))
    fig.add_trace(go.Scatter(
        x=[0],           
        y=[superweight],
        mode="markers",
        marker=dict(color="red", size=10),
        name="Superweight"
    ))
    fig.update_layout(
        title=f"Round {round_idx} - Step {iter_num}: transformer.h.2.mlp.c_proj weights",
        yaxis_title="Weight Value",
        xaxis={'visible': False}
    )
    html_path = f"{args_out_dir}/c_proj_box_round_{round_idx}_iter_{iter_num}.html"
    os.makedirs(os.path.dirname(html_path), exist_ok=True)
    fig.write_html(html_path)
    #return html_path
    mlflow.log_artifact(html_path, artifact_path=f"c_proj_boxplots/round_{round_idx}")



def prune_weights(model, mask, prune_percent, debug=False):
    """
    Globally prune the smallest `prune_percent` of STILL-UNMASKED weights by magnitude.
    If debug=True, prints:
      - which param names the model has vs. which the mask has
      - for each param, how many unmasked entries
      - global stats: total unmasked, k, threshold, sample values
    Returns an updated mask dict.
    """
    real_mod = model.module if hasattr(model, 'module') else model

    # DEBUG: compare parameter names ↔ mask keys
    '''
        if debug and is_master(): # can be taken out, was helpful when dealing with "module." prefix issues
        param_names = [n for n,_ in real_mod.named_parameters()]
        mask_keys   = list(mask.keys())
        missing     = set(param_names) - set(mask_keys)
        extra       = set(mask_keys)   - set(param_names)
        print(f"[prune][debug] model has {len(param_names)} params, mask has {len(mask_keys)} entries")
        print(f"[prune][debug] params missing from mask ({len(missing)}): {sorted(list(missing))[:5]}…")
        print(f"[prune][debug] mask keys not in model ({len(extra)}): {sorted(list(extra))[:5]}…\n")
    '''

    #  collect all still-unmasked absolute values
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

    # flatten & compute threshold
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

    #  find threshold & sample
    thresh = torch.kthvalue(all_flat, k).values.item()
    if debug:
        smallest = torch.topk(all_flat, k, largest=False).values[:min(10,k)].cpu().tolist()
        if is_master():
            print(f"[prune][debug] threshold = {thresh:.6e}")
            print(f"[prune][debug] sample smallest: {smallest}\n")


    layer_stats = []
    #  apply threshold to rebuild masks
    for name, p in real_mod.named_parameters():
        if name not in mask:
            continue
        m_old = mask[name].bool()
        m_new = (p.data.abs() > thresh) & m_old
        kept  = int(m_new.sum().item())
        total_nm = m_old.numel()
        survival = 100.0 * kept / total_nm
        if debug:
            if is_master():
                print(f"[prune][debug] {name}: kept {kept}/{int(m_old.sum().item())}")
        mask[name] = m_new.to(mask[name].dtype)
        layer_stats.append({
            "layer": name,
            "kept":kept,
            "masked":total_nm - kept,
            "total": total_nm,
            "survival": survival,
        })

    if debug:
        total_after = sum(int(m.sum().item()) for m in mask.values())
        if is_master():
            print(f"[prune][debug] remaining unmasked after prune: {total_after}/{total}\n")

    return mask, thresh, layer_stats


def save_mask(mask, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(mask, path)

def load_mask(path, map_location=None):
    return torch.load(path, map_location=map_location)

def train_one_round(model, optimizer, scaler, args, get_batch_fn, ctx, device, round_idx, out_dir, SW_C_OUT, SW_C_IN):
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
    if is_master:
        records = {
            'step':        [],
            'train_loss':  [],
            'val_loss':    [],
            'val_ppl':     [],  # fill NaN when not eval step
            'time_ms':     [],
            'mfu':         [],
        }
    while iter_num < args.max_iters:
        #  set LR
        lr = get_lr(iter_num, args)
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        val_loss = np.nan
        val_ppl = np.nan
        #  eval + checkpoint
        if (iter_num % args.eval_interval == 0 or iter_num +1 ==args.max_iters)  and args.master:
            losses = estimate_loss(model, get_batch_fn, ctx, args.eval_iters)

            val_loss = losses['val'].item()
            val_ppl = math.exp(val_loss) # perplexity

            print(f"[round] step {iter_num}: train {losses['train']:.4f}, val {losses['val']:.4f}, ppl {val_ppl:.4f}", flush=True)

            if losses['val'] < best_val or args.always_save_checkpoint:
                best_val = losses['val']
                ck = dict(
                  model=raw.state_dict(),
                  optimizer=optimizer.state_dict(),
                  iter_num=iter_num,
                  best_val_loss=best_val,
                )

                #if iter_num + 1 == args.max_iters:
                # back to saving on every eval step
                torch.save(ck, os.path.join(BASE_OUTPUT_PATH, args.out_dir, f'ckpt_round_{round_idx}_step_{iter_num}.pt'))

        if (iter_num % args.eval_interval == 0 or iter_num +1 ==args.max_iters) and args.master:
            plot_c_proj_box(model, round_idx, iter_num, os.path.join(BASE_OUTPUT_PATH, out_dir), SW_C_OUT, SW_C_IN)
        # 3 forward/backward with grad-accum
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

        #  logging
        dt = time.time() - t0; t0 = time.time()
        if iter_num % args.log_interval == 0 and args.master:
            lossf = loss.item() * args.gradient_accumulation_steps

            # just experimenting for fun
            #if is_master:
            #    mlflow.log_metric(f"train_loss_{round_idx}", loss.item(), step=iter_num)
            #    mlflow.log_metric("learning_rate", lr, step=iter_num)

            if local_iter >= 5:
                mfu = raw.estimate_mfu(args.batch_size * args.gradient_accumulation_steps, dt)
                running_mfu = mfu if running_mfu < 0 else 0.9*running_mfu + 0.1*mfu
            print(f"[round] iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.1f}ms, mfu {running_mfu*100:.1f}%", flush=True)
            records['step'].append(iter_num)
            records['train_loss'].append(round(lossf,2))
            records['val_loss'].append(round(val_loss,2)) # nan except on eval steps
            records['val_ppl'].append(round(val_ppl,2))   # nan except on eval steps
            records['time_ms'].append(round(dt * 1000, 1))
            records['mfu'].append(round(running_mfu * 100, 1))

        iter_num += 1
        local_iter += 1

    return records



def main():
    args = parse_args()

    SW_LAYER = args.sw_layer
    SW_C_OUT = args.sw_c_out
    SW_C_IN = args.sw_c_in
    INIT_TYPE = args.init_type
    PRUNE_PERCENT = args.prune_percent
    OUTPUT_DIR = args.out_dir

    args.ddp, args.rank, args.local_rank, args.world_size, args.device, args.master = setup_distributed(args)
    
    SEED = 13 + (args.rank if args.ddp else 0) # 1337, 500 , 13 
    torch.manual_seed(SEED) 

    
    if args.master:
        os.makedirs(os.path.join(BASE_OUTPUT_PATH, args.out_dir), exist_ok=True)

    device_type = 'cuda' if 'cuda' in args.device else 'cpu'
    if args.dtype is None:
        args.dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
    ptd = dict(float32=torch.float32, bfloat16=torch.bfloat16, float16=torch.float16)[args.dtype]
    ctx = torch.amp.autocast(device_type=device_type, dtype=ptd) if device_type!='cpu' else nullcontext()

    model, _ = build_model(args,args.device, init_type=INIT_TYPE)
    scaler = torch.amp.GradScaler('cuda',enabled=(args.dtype=='float16'))
    optimizer = model.configure_optimizers(args.weight_decay, args.learning_rate,
                                           (args.beta1, args.beta2), device_type)
    if args.init_from == 'resume':
        ck = torch.load(os.path.join(BASE_OUTPUT_PATH, args.out_dir,'ckpt.pt'), map_location=args.device)
        optimizer.load_state_dict(ck['optimizer'])
    if args.compile:
        model = torch.compile(model)
        print(f"[debug] model compiled with torch.compile, dtype={args.dtype}, device={device_type}")
    if args.ddp:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])

    raw_model = model.module if hasattr(model, 'module') else model
    init_state = copy.deepcopy(raw_model.state_dict())

    # mlflow.pytorch.log_model(raw_model, "initial_model")

    #mask_path = os.path.join(BASE_OUTPUT_PATH, args.out_dir, 'mask.pt')
    #if os.path.exists(mask_path):
    #     mask = load_mask(mask_path, map_location=args.device)
    #else:
    STARTROUND = 1

    if STARTROUND == 4:
        mask = torch.load(
        os.path.join(BASE_OUTPUT_PATH, '..', 'J1_seed2',
        'mask_round_3_sp_72.9.pt'),
        map_location=device_type
    )
    else:
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
            print(f"[debug] initial canonical mask {initial_unmasked}")

    if args.master:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment("gpt2-LWH-SW") 
        mlflow.start_run(run_name=OUTPUT_DIR) 
        mlflow.log_params(vars(args))

        mlflow.log_param("SW_LAYER", SW_LAYER)
        mlflow.log_param("SW_C_OUT", SW_C_OUT)
        mlflow.log_param("SW_C_IN", SW_C_IN)
        mlflow.log_param("PRUNE_PERCENT", PRUNE_PERCENT)
        mlflow.log_param('seed', SEED)
        mlflow.log_param('init_type', INIT_TYPE)
        mlflow.log_param('start_round', STARTROUND)
    all_round_metrics = {}
    sw_preserved_map =  {}
    sparsity_map = {}
    for round_idx in range(STARTROUND, 6): 
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
            scaler    = torch.amp.GradScaler('cuda', enabled=(args.dtype=='float16'))

        apply_mask(raw_model, mask)
        records = train_one_round(model, optimizer, scaler, args, get_batch_fn, ctx, args.device, round_idx, os.path.join(BASE_OUTPUT_PATH, args.out_dir), SW_C_OUT, SW_C_IN)

        # prune & report
        mask, threshold, stats = prune_weights(raw_model, mask, prune_percent=PRUNE_PERCENT, debug=True)

        val, nonzero = check_c_proj_entry(model, SW_LAYER, SW_C_OUT, SW_C_IN)

        status = "non-zero" if nonzero else "ZEROED OUT"
        if args.master:
            print(f"[monitor] SW is → {status}", flush=True)

        total = sum(m.numel() for m in mask.values())
        remain = sum(int(m.sum().item()) for m in mask.values())

        if args.master:
            print(f"[sparsity] {remain}/{total} non-zero → {(100*remain/total):.2f}%", flush=True)
            mlflow.log_metric("sparsity", 100*remain/total, step=round_idx)

            df = pd.DataFrame(records)
            csv_path = os.path.join(BASE_OUTPUT_PATH, OUTPUT_DIR, f"losses_ppl_round_{round_idx}.csv")
            os.makedirs(os.path.dirname(csv_path), exist_ok=True)
            df.to_csv(csv_path, index=False)
            mlflow.log_artifact(csv_path, artifact_path="losses_ppl ")
            print(f"[round] metrics saved to {csv_path}", flush=True)


            sparsity_map[round_idx] = round(100*remain/total, 2)
            all_round_metrics[round_idx] = {
            'step':       records['step'],
            'train_loss': records['train_loss'],
            'val_loss':   records['val_loss'],
            'val_ppl':        records['val_ppl']}
            sw_preserved_map[round_idx] = nonzero

            print(all_round_metrics[round_idx])
            print(sw_preserved_map[round_idx])
            print(sparsity_map[round_idx])

            mlflow.log_metric("SW_value", val, step=round_idx)
            mlflow.log_metric("prune_threshold", threshold, step=round_idx)

            df = pd.DataFrame(stats)
            csv_path = os.path.join(BASE_OUTPUT_PATH, args.out_dir, f"prune_stats_round_{round_idx}.csv")
            os.makedirs(os.path.dirname(csv_path), exist_ok=True)
            df.to_csv(csv_path, index=False)
            mlflow.log_artifact(csv_path, artifact_path=f"prune_stats/round_{round_idx}")
            mask_path = os.path.join(BASE_OUTPUT_PATH, args.out_dir, f'mask_round_{round_idx}_sp_{round(100*remain/total, 1)}.pt')
            os.makedirs(os.path.dirname(mask_path), exist_ok=True)
            save_mask(mask, mask_path)

    if args.master:
        print("for out all_round_metrics:")
        for rnd, data in all_round_metrics.items():
            print(f"  Round {rnd}: {data}")
        print("for out sw_preserved flags:")
        for rnd, flag in sw_preserved_map.items():
            print(f"  Round {rnd}: {flag}")
        print("out sparsity_map:")
        for rnd, sp in sparsity_map.items():
            print(f"  Round {rnd}: {sp}")
        print("-" * 40)

        fig = plot_round_metrics_plotly(all_round_metrics, sw_preserved_map, sparsity_map)
        fig_path = os.path.join(BASE_OUTPUT_PATH, args.out_dir, 'all_rounds_metrics_plotly.html')
        os.makedirs(os.path.dirname(fig_path), exist_ok=True)
        fig.write_html(fig_path)
        mlflow.log_artifact(fig_path, artifact_path="all_rounds_metrics_plotly")
        mlflow.end_run()

    if args.ddp:
        destroy_process_group()


if __name__ == '__main__':
    main()
