import pandas as pd
import random
import unicodedata
from datasets import load_dataset
from nltk.translate.meteor_score import meteor_score
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

# Function to calculate METEOR scores
def calculate_meteor_scores(stories, reference_stories):
    """Calculate METEOR scores for each story."""
    meteor_scores = []
    
    for idx, (story, reference) in enumerate(zip(stories, reference_stories)):
        # Normalize both the generated story and the reference story
        story_normalized = normalize_text(story)
        reference_normalized = normalize_text(reference)
        
        # Compute METEOR score
        score = meteor_score([reference_normalized.split()], story_normalized.split())
        
        meteor_scores.append({
            'story_idx': idx,
            'meteor_score': score
        })
    
    return pd.DataFrame(meteor_scores)

# Function to visualize METEOR scores
def visualize_meteor_scores(df, filename="plot_filename"):#"Marathimeteor_scores_plot.png"):
    """Plot METEOR scores and save the visualization."""
    plt.figure(figsize=(8, 5))
    plt.bar(df['story_idx'], df['meteor_score'])
    plt.xlabel("Story Index")
    plt.ylabel("METEOR Score")
    plt.title("METEOR Score per Story")
    plt.savefig(filename)
    plt.show()

# Example usage
#dataset_name = "TinyStories-Regional/beng-generated_4o-mini_2M"  # Replace with your dataset name
#dataset_name = "TinyStories-Regional/marathi-generated_4o-mini_2M"
dataset_name = "TinyStories-Regional/hindi-generated_4o-mini_2M"
num_samples = 10  # Number of samples to analyze

# Load stories and calculate METEOR scores
stories = load_and_sample_stories(dataset_name, num_samples)
reference_stories = load_and_sample_stories(dataset_name, num_samples)
meteor_scores_df = calculate_meteor_scores(stories, reference_stories)

# Save results to CSV and JSON
csv_filename = "Hindi_meteor_scores.csv"
json_filename = "Hindi_meteor_scores.json"
plot_filename = "Hindi_meteor_scores_plot.png"

meteor_scores_df.to_csv(csv_filename, index=False)
meteor_scores_df.to_json(json_filename, orient="records", indent=4)

# Visualize the results and save the plot
visualize_meteor_scores(meteor_scores_df, plot_filename)

print(f"METEOR scores saved to {csv_filename}, {json_filename}, and visualization saved as {plot_filename}.")
