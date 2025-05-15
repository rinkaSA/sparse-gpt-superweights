import random
from tqdm import trange
from datasets import load_dataset
from transformers import AutoTokenizer

def get_wikitext2(num_samples: int, sequence_length: int, tokenizer: AutoTokenizer, train: bool = True):
    print("Loading WikiText2")
    split = "train" if train else "test"
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    tokens = tokenizer("\n\n".join(dataset["text"]), return_tensors="pt", add_special_tokens=False).input_ids

    if train:
        data = [
            tokens[:, random.randint(0, tokens.shape[1] - sequence_length - 1) : i + sequence_length]
            for _ in trange(num_samples, desc="Preparing calibration data")
        ]
    else:
        data = [
            tokens[:, i * sequence_length : (i + 1) * sequence_length]
            for i in range(tokens.numel() // sequence_length)
        ]
    return data
