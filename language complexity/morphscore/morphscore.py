import pandas as pd
import numpy as np
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("TWO/sutra-mlt256-v2")
#tokenizer = AutoTokenizer.from_pretrained("sarvamai/sarvam-1")

def morph_eval(morphemes, tokens):
    # If there's only one token, return 0
    if len(tokens) == 1:
        return 0
    
    # Remove the leading '▁' from the first token if it exists
    cleaned_tokens = tokens.copy()
    if cleaned_tokens[0].startswith('▁'):
        cleaned_tokens[0] = cleaned_tokens[0][1:]
    
    # Try all possible segmentation points
    for t in range(1, len(cleaned_tokens)):
        pt1, rest = ''.join(cleaned_tokens[:t]), ''.join(cleaned_tokens[t:])
        if [pt1, rest] == morphemes:
            return 1
    return -1

def get_morphscore():
    dataset = pd.read_csv('/home/user/nanogpt/morph_data/bengali_morphemes_data.csv') # Hindi - hindi_morph_data.csv | Marathi - marathi_morphed_data.csv
    points = []
    special_ids = set(tokenizer.all_special_ids)

    for _, row in dataset.iterrows():
        pt1 = row['pt1']
        rest = row['rest']
        morphemes = [pt1, rest]
        full_word = row['full_word']
        tokens = tokenizer(full_word)['input_ids']
        
        # Remove special tokens
        tokens = [t for t in tokens if t not in special_ids]
        tokens = [tokenizer.convert_ids_to_tokens(t) for t in tokens]
        
        print(f"Full Word: {full_word}")
        print(f"Morphemes: {morphemes}")
        print(f"Tokens: {tokens}")
        print(f"Cleaned first token: {tokens[0][1:] if tokens[0].startswith('▁') else tokens[0]}")

        point = morph_eval(morphemes, tokens)
        points.append(point)

    # Filter out zeros and convert -1 to 0
    points = [x for x in points if x != 0]
    points = [0 if x == -1 else x for x in points]
    
    morph_score = np.mean(points) if points else 0.0
    return morph_score

print(get_morphscore())
