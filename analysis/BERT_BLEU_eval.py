import pandas as pd
import random
import unicodedata
from datasets import load_dataset
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import torch
from transformers import BertTokenizer, BertModel
import matplotlib.pyplot as plt

#May need to run below code at the very beginning in your work environemnt

'''
import ssl
import nltk

# Bypass SSL verification
ssl._create_default_https_context = ssl._create_unverified_context

# Download WordNet
nltk.download('wordnet')
nltk.download('omw-1.4')  # Optional, but helps with multilingual support
'''

# Normalize text to handle Unicode and extra spaces
def normalize_text(text):
    """Normalize Unicode characters and remove extra spaces."""
    return unicodedata.normalize("NFKC", text).strip().lower()

# Load and sample stories from the dataset
def load_and_sample_stories(dataset_name, num_samples=10):
    """Load the dataset and sample a subset of stories."""
    dataset = load_dataset(dataset_name)
    stories = dataset['train']['story']  # Assuming 'story' column contains the text
    sampled_stories = random.sample(stories, min(num_samples, len(stories)))
    return sampled_stories

# Function to calculate BLEU and BERTScore
def calculate_bleu_bert_scores(stories, reference_stories):
    """Calculate BLEU and BERTScore."""
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased")
    
    smoothing = SmoothingFunction().method1  # Smoothing function for BLEU
    evaluation_scores = []
    
    for idx, (story, reference) in enumerate(zip(stories, reference_stories)):
        # Normalize both the generated story and the reference story
        story_normalized = normalize_text(story)
        reference_normalized = normalize_text(reference)
        
        # Compute BLEU score (1-gram precision)
        bleu_score = sentence_bleu([reference_normalized.split()], story_normalized.split(), 
                                   smoothing_function=smoothing)
        
        # Compute BERTScore (cosine similarity of embeddings)
        inputs_story = tokenizer(story_normalized, return_tensors="pt", padding=True, truncation=True)
        inputs_reference = tokenizer(reference_normalized, return_tensors="pt", padding=True, truncation=True)
        
        with torch.no_grad():
            story_embedding = model(**inputs_story).last_hidden_state.mean(dim=1)  # Average pooling
            reference_embedding = model(**inputs_reference).last_hidden_state.mean(dim=1)
        
        bert_score = torch.nn.functional.cosine_similarity(story_embedding, reference_embedding).item()
        
        evaluation_scores.append({
            'story_idx': idx,
            'bleu_score': bleu_score,
            'bert_score': bert_score
        })
    
    return pd.DataFrame(evaluation_scores)

# Function to visualize scores and save the plot
def visualize_scores(df, filename="plot_filename"):
    """Plot BLEU and BERTScore comparisons and save the visualization."""
    plt.figure(figsize=(12, 5))
    
    # Plot BLEU scores
    plt.subplot(1, 2, 1)
    plt.bar(df['story_idx'], df['bleu_score'])
    plt.xlabel("Story Index")
    plt.ylabel("BLEU Score")
    plt.title("BLEU Score per Story")
    
    # Plot BERT scores
    plt.subplot(1, 2, 2)
    plt.bar(df['story_idx'], df['bert_score'])
    plt.xlabel("Story Index")
    plt.ylabel("BERT Score")
    plt.title("BERT Score per Story")
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()

# Example usage
#dataset_name = "TinyStories-Regional/beng-generated_4o-mini_2M"  # Replace with your dataset name
#dataset_name = "TinyStories-Regional/marathi-generated_4o-mini_2M"
dataset_name = "TinyStories-Regional/hindi-generated_4o-mini_2M"
num_samples = 10  # Number of samples to analyze

# Load stories and calculate scores
stories = load_and_sample_stories(dataset_name, num_samples)
reference_stories = load_and_sample_stories(dataset_name, num_samples)
evaluation_scores_df = calculate_bleu_bert_scores(stories, reference_stories)

# Save results to CSV and JSON
csv_filename = "Hindi_bleu_bert_scores.csv"
json_filename = "Hindi_bleu_bert_scores.json"
plot_filename = "Hindi_bleu_bert_scores_plot.png"

evaluation_scores_df.to_csv(csv_filename, index=False)
evaluation_scores_df.to_json(json_filename, orient="records", indent=4)

# Visualize the results and save the plot
visualize_scores(evaluation_scores_df, plot_filename)

print(f"BLEU and BERT scores saved to {csv_filename}, {json_filename}, and visualization saved as {plot_filename}.")
