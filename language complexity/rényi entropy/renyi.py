import numpy as np
from tqdm import tqdm
import os

# Function to load tokenized data
def load_tokenized_data(file_path):
    """Loads tokenized data from a .bin file."""
    with open(file_path, 'rb') as f:
        data = np.fromfile(f, dtype=np.int32)
    return data

# Function to filter out BOS and EOS tokens
def filter_tokens(data, bos_token, eos_token):
    """Filters out BOS and EOS tokens from the dataset."""
    return data[(data != bos_token) & (data != eos_token)]

# Function to calculate Rényi entropy
def renyi_entropy(token_counts, alpha):
    """Calculates Rényi entropy for a given token frequency distribution."""
    probabilities = token_counts / np.sum(token_counts)
    if alpha == 1.0:
        # Shannon entropy (limit case of Rényi entropy)
        return -np.sum(probabilities * np.log(probabilities))
    else:
        return 1 / (1 - alpha) * np.log(np.sum(probabilities ** alpha))

# Function to compute token frequencies
def compute_token_frequencies(data):
    """Computes the frequency of each token in the dataset."""
    unique, counts = np.unique(data, return_counts=True)
    token_counts = dict(zip(unique, counts))
    return token_counts

# Main function to analyze language complexity
def analyze_language_complexity(language_datasets, alpha_values, tokenizer):
    """Analyzes language complexity using Rényi entropy."""
    results = {}

    # Define BOS and EOS tokens based on tokenizer
    if tokenizer == 'sutra':
        bos_token, eos_token = 0, 2
    elif tokenizer == 'sarvam':
        bos_token, eos_token = 1, 2
    else:
        raise ValueError("Unsupported tokenizer. Choose either 'sutra' or 'sarvam'.")

    for language, paths in language_datasets.items():
        print(f"Analyzing {language} dataset...")

        # Combine training and validation data
        train_data = load_tokenized_data(paths['train'])
        val_data = load_tokenized_data(paths['val'])
        combined_data = np.concatenate((train_data, val_data))

        # Filter out BOS and EOS tokens
        filtered_data = filter_tokens(combined_data, bos_token, eos_token)

        # Compute token frequencies
        token_counts = compute_token_frequencies(filtered_data)

        # Compute Rényi entropy for each alpha
        entropy_values = {}
        for alpha in tqdm(alpha_values, desc=f"Processing {language}"):
            entropy_values[alpha] = renyi_entropy(np.array(list(token_counts.values())), alpha)

        results[language] = entropy_values

    return results

# Specify dataset paths
language_datasets = {
    'Hindi': {
        'train': '/home/user/nanogpt/data/hindisutra/train.bin',
        'val': '/home/user/nanogpt/data/hindisutra/val.bin'
    },
#    'Marathi': {
#        'train': '/home/user/nanogpt/data/marsutra/train.bin',
#        'val': '/home/user/nanogpt/data/marsutra/val.bin'
#    },
#    'Bengali': {
#         'train': '/home/user/nanogpt/data/bengsutra/train.bin',
#         'val': '/home/user/nanogpt/data/bengsutra/val.bin'
#    }
}

# Specify alpha values for Rényi entropy
alpha_values = [0.5, 1.0, 2.0, 2.5]

# Specify tokenizer ('sutra' or 'sarvam')
tokenizer = 'sutra'  # Change to 'sarvam' if needed

# Run analysis
results = analyze_language_complexity(language_datasets, alpha_values, tokenizer)

# Print results
for language, entropy_values in results.items():
    print(f"\nLanguage: {language}")
    for alpha, entropy in entropy_values.items():
        print(f"  Rényi Entropy (alpha={alpha}): {entropy:.4f}")
