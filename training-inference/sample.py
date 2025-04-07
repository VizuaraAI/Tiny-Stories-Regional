"""
Sample from a trained model
"""
import os
import torch
import pickle
import tiktoken
import numpy as np
from tqdm import tqdm
from model import GPTConfig, GPT
from contextlib import nullcontext
# Sarvam
from transformers import AutoTokenizer
# HF 
from huggingface_hub import HfApi
from huggingface_hub import hf_hub_download

# Suppress FutureWarning
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


# ----------------------------------------------------------------------------- 
# Configuration
# -----------------------------------------------------------------------------

# Model I/O
out_dir = 'out' # ignored if sample_init_from is not 'resume'
lang = 'hindi'
model_name = "ckpt"
sample_init_from = 'resume' # 'resume' from out_dir/save_folder/model_name.pt

# Sample config
start = "\n"             # Can also specify a file, use as: "FILE:prompt.txt"
num_samples = 10
max_new_tokens = 500
temperature = 0.8
top_k = 200
seed = 1337
device = 'cuda'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile = True
tokenizer = "sarvam_hf"    # sarvam_hf or sutra_hf or tik
dry_run = False

# HF
load_from_hf = False
HF_ACCESS_TOKEN = ""

# Select the output mode ('text' or 'json')
output_mode = "json"  # change to "text" for the original text output

##############################################
# Overrides from command line or config file
exec(open('configurator.py').read()) 
print("")
##############################################

# Post config.py save path
if "sarvam" in tokenizer: sub_folder = "sarvam"
elif "sutra" in tokenizer: sub_folder = "SUTRA"
elif "tik" in tokenizer: sub_folder = "tiktoken"
# Post config.py HF 
if load_from_hf:
    #repo_id = f"TinyStories-Regional/{lang}-generated_4o-mini_2M"
    repo_id = f"TinyStories-Regional/tokenizer-comparisons"
    model_path = f"{sub_folder}/{model_name}.pt"
    # Download the model file
    model_path_hf = hf_hub_download(repo_id=repo_id, filename=model_path, repo_type="model", use_auth_token=HF_ACCESS_TOKEN)
    print("\nMODEL LOADED FROM HF\n")

# Post config.py load model from and write samples to 
save_folder = lang

# Post config.py start 
if "test" not in start and start[0] == "F": start = f"FILE:prompt-{lang}.txt"


# ----------------------------------------------------------------------------- 
# Inference code
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# model
if sample_init_from == 'resume':
    ckpt_path = os.path.join(out_dir, f'{save_folder}/{model_name}.pt')
    # Model loaded from HF
    if load_from_hf: ckpt_path = model_path_hf
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
elif sample_init_from.startswith('gpt2'):
    model = GPT.from_pretrained(sample_init_from, dict(dropout=0.0))

model.eval()
model.to(device)
if compile:
    model = torch.compile(model)

################## TOKENIZER ##################
load_meta = False
if (sample_init_from == 'resume' 
    and 'config' in checkpoint 
    and 'dataset' in checkpoint['config']):
    meta_path = os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')
    load_meta = os.path.exists(meta_path)

if load_meta:
    print(f"Loading meta from {meta_path}...")
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    stoi, itos = meta['stoi'], meta['itos']
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
else:
    print(f"\nSample Init: {sample_init_from}")
    if tokenizer == "tik":
        print("Tiktoken tokenizer being used.\n")
        enc = tiktoken.get_encoding("gpt2")
        encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
        decode = lambda l: enc.decode(l)
    elif "hf" in tokenizer: 
        # Different encode
        if tokenizer == "sarvam_hf": 
            print("Sarvam tokenizer being used.\n")
            enc = AutoTokenizer.from_pretrained('sarvamai/sarvam-1')
        elif tokenizer == "sutra_hf":
            print("SUTRA tokenizer being used.\n")
            enc = AutoTokenizer.from_pretrained('TWO/sutra-mlt256-v2')   
        # Commond encode and decode
        encode = lambda s: enc(s, add_special_tokens=False, return_tensors=None)["input_ids"]
        decode = lambda l: enc.decode(l)
################## TOKENIZER ##################

# Check if `start` points to a file of prompts
if start.startswith('FILE:'):
    file_path = start[5:]
    with open(file_path, 'r', encoding='utf-8') as f:
        all_prompts = [line.strip() for line in f if line.strip()]
else:
    all_prompts = [start]

# Find first story
def splice_first_story(input_string):
    """
    Finds the first occurrence of 'eos_token' in the given string.
    If found, returns the string up to that occurrence (excluding 'eos').
    If not found, returns the string as it is.
    """
    # Delimiter
    if "hf" in tokenizer: delimiter = enc.eos_token
    elif tokenizer == "tik": delimiter = "<|endoftext|>"
    
    # Finding delimiter in string
    if delimiter in input_string:
        index = input_string.index(delimiter)
        return input_string[:index].strip()
    # Delimiter not found i.e. end of story not found
    return input_string.strip()

# Combined results
results_for_all_prompts = []

for prompt_index, prompt_text in tqdm(enumerate(all_prompts, start=1), total=len(all_prompts), dynamic_ncols=True):
    # Adding BOS token to start of prompt 
    if "hf" in tokenizer: prompt_ids = encode(prompt_text)
    else: prompt_ids = encode(prompt_text)
    
    x = torch.tensor(prompt_ids, dtype=torch.long, device=device)[None, ...]

    sampled_text_list = []
    with torch.no_grad():
        with ctx:
            for k in range(num_samples):
                y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
                ids = y[0].tolist()
                decoded_str = splice_first_story(decode(ids))
                sampled_text_list.append(decoded_str)

    results_for_all_prompts.append((prompt_index, prompt_text, sampled_text_list))
    
    if dry_run: break


# --------------------------------------------------------------------
# Output handling based on `output_mode`
# --------------------------------------------------------------------
# Output folder
write_to_folder = save_folder
save_directory = f"samples/{sub_folder}/{write_to_folder}"

# Create save directory if not present
os.makedirs(save_directory, exist_ok=True)

if output_mode == "text":
    # TEXT MODE
    with open(f"{save_directory}/sample-{model_name}.txt", 'w', encoding='utf-8') as f:
        for prompt_index, prompt_text, sampled_text_list in results_for_all_prompts:
            for story_index, story_text in enumerate(sampled_text_list, start=1):
                f.write(f"PROMPT[{prompt_index}] - STORY[{story_index}]\n\n")
                f.write(story_text)
                f.write('\n\n----------------------------------------\n\n')
    print("Text samples successfully written to sample.txt.")

elif output_mode == "json":
    # JSON MODE
    import json
    json_data = []
    for prompt_index, prompt_text, sampled_text_list in results_for_all_prompts:
        for story_index, story_text in enumerate(sampled_text_list, start=1):
            # Json DICT 
            json_data.append({
                "ID": f"Prompt[{prompt_index}] - Story[{story_index}] | Prompt: {prompt_text}",
                "prompt": prompt_text,
                "story": story_text,
            })

    with open(f"{save_directory}/sample-{model_name}.json", 'w', encoding='utf-8') as json_file:
        json.dump(json_data, json_file, ensure_ascii=False, indent=2)
    print("JSON samples successfully written to sample.json.")
