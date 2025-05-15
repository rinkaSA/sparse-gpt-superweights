import torch
import torch.nn.functional as F
from tqdm import trange

@torch.no_grad()
def compute_perplexity(model, data, batch_size: int = 1):
    device = next(model.parameters()).device
    nll_running = 0
    tokens_processed = 0
    for i in trange(0, len(data), batch_size, desc="Computing perplexity", leave=False):
        inputs = torch.cat(data[i:i+batch_size]).to(device)
        logits = model(inputs).logits
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = inputs[:, 1:]
        loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        a = shift_labels.numel() / (tokens_processed + shift_labels.numel())
        b = tokens_processed / (tokens_processed + shift_labels.numel())
        nll_running = a * loss + b * nll_running
        tokens_processed += shift_labels.numel()
    return nll_running.exp().item()
