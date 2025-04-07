# Hugging Face
from datasets import load_dataset
# Tokenizers
from transformers import AutoTokenizer
import tiktoken
# General
import numpy as np
import os

# choose dataset language
lang = "beng_4omini-sutra"
# 4omini data + HF
if lang == "hin_4omini": folder = "hindi-generated-sarvam"
elif lang == "mar_4omini": folder = "mar-generated-sarvam"
elif lang == "beng_4omini": folder = "beng-generated-sarvam"
elif lang == "beng_4omini-sutra": folder = "beng-generated-sutra"
# 4omini data + tiktoken
elif lang == "hin_4omini-tik": folder = "hindi-generated-tiktoken"
# Translated
elif lang == "hin_trans": folder = "hindi-translated-sarvam"
elif lang == "beng_trans": folder = "beng-translated-sarvam"

# File path for the train.bin file
train_bin_path = os.path.join(f"data/{folder}/train.bin")
# train_bin_path = "data/train.bin"

# Define the data type used during writing
dtype = np.uint32  
# Open the binary file using np.memmap
train_data = np.memmap(train_bin_path, dtype=dtype, mode='r')

# Number of tokens
print(f"\nNumber of tokens in the dataset: {len(train_data)}")

# Read the first 500 tokens
first_500_tokens = train_data[:500]
# Print the first 500 tokens
print("\nFirst 500 tokens from train.bin:")
print(first_500_tokens)
print("")

# Choose tokenizer
if "tik" in lang: enc = tiktoken.get_encoding("gpt2")
elif "sutra" in lang: enc = AutoTokenizer.from_pretrained('TWO/sutra-mlt256-v2')
else: enc =  AutoTokenizer.from_pretrained('sarvamai/sarvam-1')

# Decode
decoded_text = enc.decode(first_500_tokens)
print(f"Decoded first 500 tokens: {decoded_text}\n")