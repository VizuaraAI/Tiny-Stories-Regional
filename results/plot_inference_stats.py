import json
import numpy as np
import os
from typing import List, Dict
import matplotlib.pyplot as plt
import seaborn as sns

'''
1. This code will read the performance evaluations stored in the json files
3. It will calculate statisitcs for the particular model
2. Then create histogram and violin plots of statistics for each performance metric
3. Store stats in text files with proper names
'''

def calculate_statistics(data: List[Dict]) -> Dict:
    total_entries = len(data)
    metrics = ['context awareness', 'completeness', 'grammar', 'fluency', 'creativity', 'overall']
    results = {'total_entries': total_entries}
    
    for metric in metrics:
        values = np.array([float(item[metric]) for item in data])
        results[metric] = {
            'mean': np.mean(values),
            'median': np.median(values),
            'std_dev': np.std(values),
            'values': values
        }
    
    return results

def format_output(stats: Dict) -> str:
    output = [f"Total number of entries: {stats['total_entries']}"]
    
    for metric, values in stats.items():
        if metric != 'total_entries':
            output.append(f"\n{metric.upper()}:")
            output.append(f"Mean: {values['mean']:.2f}")
            output.append(f"Median: {values['median']:.2f}")
            output.append(f"Standard Deviation: {values['std_dev']:.2f}")
    
    return '\n'.join(output)

def plot_distributions(stats: Dict, output_prefix: str):
    metrics = [m for m in stats.keys() if m != 'total_entries']
    
    # Violin plots
    fig_violin, axes_violin = plt.subplots(3, 2, figsize=(15, 20))
    fig_violin.suptitle('Distribution of Scores (Violin Plots)', fontsize=16)
    
    # Histogram plots
    fig_hist, axes_hist = plt.subplots(3, 2, figsize=(15, 20))
    fig_hist.suptitle('Distribution of Scores (Histograms)', fontsize=16)
    
    for idx, metric in enumerate(metrics):
        row = idx // 2
        col = idx % 2
        
        # Violin plot
        sns.violinplot(data=stats[metric]['values'], ax=axes_violin[row, col])
        axes_violin[row, col].set_title(f'{metric.upper()}')
        axes_violin[row, col].set_ylabel('Score')
        axes_violin[row, col].axhline(y=stats[metric]['mean'], color='r', linestyle='--', 
                                    label=f"Mean: {stats[metric]['mean']:.2f}")
        axes_violin[row, col].axhline(y=stats[metric]['median'], color='g', linestyle='--', 
                                    label=f"Median: {stats[metric]['median']:.2f}")
        axes_violin[row, col].legend()
        
        # Histogram plot
        sns.histplot(data=stats[metric]['values'], bins=20, ax=axes_hist[row, col])
        axes_hist[row, col].set_title(f'{metric.upper()}')
        axes_hist[row, col].set_xlabel('Score')
        axes_hist[row, col].set_ylabel('Count')
        axes_hist[row, col].axvline(x=stats[metric]['mean'], color='r', linestyle='--', 
                                  label=f"Mean: {stats[metric]['mean']:.2f}")
        axes_hist[row, col].axvline(x=stats[metric]['median'], color='g', linestyle='--', 
                                  label=f"Median: {stats[metric]['median']:.2f}")
        axes_hist[row, col].legend()

    plt.tight_layout()
    fig_violin.savefig(f'{output_prefix}_violin_plots.png')
    fig_hist.savefig(f'{output_prefix}_histograms.png')
    plt.close('all')

def process_json_files(folder_path: str):
    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            base_name = os.path.splitext(filename)[0] #selects name based on json file name
            input_path = os.path.join(folder_path, filename)
            output_path = os.path.join(folder_path, f"{base_name}_stats.txt")
            
            with open(input_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            statistics = calculate_statistics(data)
            formatted_output = format_output(statistics)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(formatted_output)
            
            plot_distributions(statistics, os.path.join(folder_path, base_name))

if __name__ == "__main__":
    folder_path = "/path_to_inference_results_json_file"  # Current directory, change this to your folder path
    process_json_files(folder_path)
