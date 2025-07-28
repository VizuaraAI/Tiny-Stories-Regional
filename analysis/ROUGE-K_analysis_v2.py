import pandas as pd
import random
import unicodedata
from datasets import load_dataset
from rouge_score import rouge_scorer

'''This script evaluates the quality of generated children's stories by computing ROUGE scores, 
a common metric for text similarity, against reference stories. 
It uses the Hugging Face datasets library to load data, processes text for normalization, 
and calculates ROUGE-1, ROUGE-2, and ROUGE-L recall scores.'''

# Normalize text to handle Unicode and extra spaces
def normalize_text(text):
    """Normalize Unicode characters and remove extra spaces."""
    return unicodedata.normalize("NFKC", text).strip().lower()

# Load and sample stories from the dataset
def load_and_sample_stories(dataset_name, num_samples):
    """Load the dataset and sample a subset of stories."""
    dataset = load_dataset(dataset_name)
    stories = dataset['train']['story']  # Assuming 'story' column contains the text
    sampled_stories = random.sample(stories, min(num_samples, len(stories)))
    return sampled_stories

# Function to calculate ROUGE scores between stories
'''story_idx: Index of the story pair.
rouge1_recall: Overlap of unigrams (single words).
rouge2_recall: Overlap of bigrams (two-word sequences).
rougeL_recall: Longest common subsequence similarity.'''
def calculate_rouge_scores(stories, reference_stories):
    """Calculate ROUGE scores between generated stories and reference stories."""
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=False)
    rouge_scores = []
    
    for idx, (story, reference) in enumerate(zip(stories, reference_stories)):
        # Normalize both the generated story and the reference story (we work with raw text here)
        story_normalized = normalize_text(story)
        reference_normalized = normalize_text(reference)

        # Debugging: Check tokenized and normalized stories
        print(f"Story {idx} - Generated (Normalized): {story_normalized[:100]}...")  # Print first 100 chars
        print(f"Story {idx} - Reference (Normalized): {reference_normalized[:100]}...")  # Print first 100 chars
        
        # Get ROUGE scores (includes recall, precision, and F1-score)
        scores = scorer.score(reference_normalized, story_normalized)
        rouge_scores.append({
            'story_idx': idx,
            'rouge1_recall': scores['rouge1'].recall,
            'rouge1_precision': scores['rouge1'].precision,
            'rouge1_f1': scores['rouge1'].fmeasure,
            'rouge2_recall': scores['rouge2'].recall,
            'rouge2_precision': scores['rouge2'].precision,
            'rouge2_f1': scores['rouge2'].fmeasure,
            'rougeL_recall': scores['rougeL'].recall,
            'rougeL_precision': scores['rougeL'].precision,
            'rougeL_f1': scores['rougeL'].fmeasure
        })
    
    return pd.DataFrame(rouge_scores)

# Example usage
#dataset_name = "TinyStories-Regional/beng-generated_4o-mini_2M"  # Replace with your dataset name
#dataset_name = "TinyStories-Regional/hindi-generated_4o-mini_2M"
dataset_name = "roneneldan/TinyStories" #original English dataset
num_samples = 10  # Number of samples to analyze

# Load stories and calculate ROUGE scores
stories = load_and_sample_stories(dataset_name, num_samples)
reference_stories = load_and_sample_stories(dataset_name, num_samples)
rouge_scores_df = calculate_rouge_scores(stories, reference_stories)

# Save results to CSV and JSON
# Provide appropriate file name
csv_filename = "english_rouge_scores.csv"
json_filename = "english_rouge_scores.json"

rouge_scores_df.to_csv(csv_filename, index=False)
rouge_scores_df.to_json(json_filename, orient="records", indent=4)

# Print the results
print(rouge_scores_df)
print(f"ROUGE scores saved to {csv_filename} and {json_filename}")
