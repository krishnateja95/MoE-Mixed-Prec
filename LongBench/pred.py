import os
from datasets import load_dataset
import torch
import json


from tqdm import tqdm
import numpy as np
import random
import argparse

# cache_dir = '/vast/users/schittyvenkata/model_weights/'
cache_dir = "/lus/grand/projects/datascience/krishnat/model_weights/LLaMA/llama_cache/"

def parse_args(args = None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--block_size", type=int, default=16)
    parser.add_argument("--evict_size", type=int, default=16)
    parser.add_argument("--evict_method", type=str, default= "none")
    parser.add_argument("--dataset", type=str, default= "qasper")
    return parser.parse_args(args)

def build_chat(tokenizer, prompt, model_name):
    prompt = f"[INST]{prompt}[/INST]"
    return prompt

def get_pred(model, tokenizer, data, max_length, max_gen, prompt_format, dataset, device, model_name, out_path):
    for json_obj in tqdm(data):
        prompt = prompt_format.format(**json_obj)
        tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
        
        if len(tokenized_prompt) > max_length:
            half = int(max_length/2)
            prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True)+tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
        if dataset not in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]:
            prompt = build_chat(tokenizer, prompt, model_name)
        
        input = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
        context_length = input.input_ids.shape[-1]
        if dataset == "samsum":
            output = model.generate(
                **input,
                max_new_tokens=max_gen,
                num_beams=1,
                do_sample=False,
                temperature=1.0,
                min_length=context_length+1,
                eos_token_id=[tokenizer.eos_token_id, tokenizer.encode("\n", add_special_tokens=False)[-1]],
            )[0]
        else:
            output = model.generate(
                **input,
                max_new_tokens=max_gen,
                num_beams=1,
                do_sample=False,
                temperature=1.0,
            )[0]
        pred = tokenizer.decode(output[context_length:], skip_special_tokens=True)
        
        with open(out_path, "a", encoding="utf-8") as f:
            json.dump({"pred": pred, "answers": json_obj["answers"], "all_classes": json_obj["all_classes"], "length": json_obj["length"]}, f, ensure_ascii=False)
            f.write('\n')

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


def load_model_and_tokenizer(path, model_name, device):    
    from transformers import AutoTokenizer
    from models.LLMs.LLaMA.modeling_llama import LlamaForCausalLM
    tokenizer = AutoTokenizer.from_pretrained(path, cache_dir = cache_dir, trust_remote_code=True)
    model     = LlamaForCausalLM.from_pretrained(path, cache_dir = cache_dir, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="auto")
    
    model.KV_cache_evict_params(method = args.evict_method, block_size  = args.block_size, evict_size = args.evict_size)
    model = model.eval()
    return model, tokenizer


if __name__ == "__main__":
    seed_everything(42)
    args = parse_args()

    model2path = json.load(open("config/model2path.json", "r"))
    model2maxlen = json.load(open("config/model2maxlen.json", "r"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = args.model
    
    model, tokenizer = load_model_and_tokenizer(model2path[model_name], model_name, device)
    
    max_length = model2maxlen[model_name]

    dataset2prompt = json.load(open("config/dataset2prompt.json", "r"))
    dataset2maxlen = json.load(open("config/dataset2maxlen.json", "r"))
    
    if not os.path.exists("pred_e"):
        os.makedirs("pred_e")
    
    data = load_dataset("THUDM/LongBench", f"{args.dataset}_e", split="test", cache_dir = cache_dir)
    data_all = [data_sample for data_sample in data]
    
    folder_path = f"pred_e/{model_name}_block_size_{args.block_size}_evict_size_{args.evict_size}_evict_method_{args.evict_method}/"
    out_path =  folder_path + f"{args.dataset}.jsonl"
    
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    prompt_format = dataset2prompt[args.dataset]
    
    max_gen = dataset2maxlen[args.dataset]
    preds = get_pred(model, tokenizer, data, max_length, max_gen, prompt_format, args.dataset, device, model_name, out_path)
    