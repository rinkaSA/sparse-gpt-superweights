import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_model_and_tokenizer(model_name: str):
    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="auto", torch_dtype="auto", low_cpu_mem_usage=True
    )
    model.config.use_cache = False
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

def get_sample_text():
    dataset = load_dataset("HuggingFaceFW/fineweb-edu", "default", split="train", streaming=True)
    return next(iter(dataset))["text"]
