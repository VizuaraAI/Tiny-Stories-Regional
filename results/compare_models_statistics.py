import re
import numpy as np
from scipy import stats
import tkinter as tk
from tkinter import filedialog
import sys
from datetime import datetime

'''
0. First run the code "plot_inference_stats.py"
1. This code will compare the inference results of two different models
2. This will check if the results are statistically signficant to be different or not
3. This assumes an equal number of samples for both the inference results
'''

def select_files():
    """
    Open file dialog for selecting input files.
    Returns tuple of (file1_path, file2_path)
    """
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    
    print("Please select the first evaluation file:")
    file1 = filedialog.askopenfilename(title="Select first evaluation file",
                                      filetypes=[("Text files", "*.txt"), ("All files", "*.*")])
    
    if not file1:
        print("No file selected. Exiting.")
        sys.exit()
        
    print("Please select the second evaluation file:")
    file2 = filedialog.askopenfilename(title="Select second evaluation file",
                                      filetypes=[("Text files", "*.txt"), ("All files", "*.*")])
    
    if not file2:
        print("No file selected. Exiting.")
        sys.exit()
        
    return file1, file2

def get_output_filename():
    """
    Ask user for output filename.
    Returns string with .txt extension added if not provided.
    """
    filename = input("Enter the desired output filename (press Enter for default): ").strip()
    
    if not filename:
        # Generate default filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"comparison_results_{timestamp}.txt"
    
    # Add .txt extension if not present
    if not filename.endswith('.txt'):
        filename += '.txt'
        
    return filename

def extract_metrics(file_path):
    """
    Extract evaluation metrics from a file.
    Returns a dictionary with category names as keys and tuples of (mean, std) as values.
    """
    metrics = {}
    current_category = None
    
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            
            # Check if this is a category header
            if line and line.endswith(':'):
                current_category = line[:-1]
                continue
                
            # Extract mean and std if we're in a category
            if current_category and line.startswith(('Mean:', 'Standard Deviation:')):
                key, value = line.split(':')
                value = float(value.strip())
                
                if key.strip() == 'Mean':
                    if current_category not in metrics:
                        metrics[current_category] = {'mean': value}
                    else:
                        metrics[current_category]['mean'] = value
                        
                elif key.strip() == 'Standard Deviation':
                    if current_category not in metrics:
                        metrics[current_category] = {'std': value}
                    else:
                        metrics[current_category]['std'] = value
    
    return metrics

def compare_metrics(file1_path, file2_path, n_samples=3000, alpha=0.05):
    """
    Compare metrics between two files using Z-test.
    
    Parameters:
    - file1_path: Path to first evaluation file
    - file2_path: Path to second evaluation file
    - n_samples: Number of samples in each group
    - alpha: Significance level
    
    Returns:
    - Dictionary containing test results for each category
    """
    # Extract metrics from both files
    metrics1 = extract_metrics(file1_path)
    metrics2 = extract_metrics(file2_path)
    
    results = {}
    
    # Compare each category
    for category in metrics1.keys():
        if category in metrics2:
            mean1 = metrics1[category]['mean']
            std1 = metrics1[category]['std']
            mean2 = metrics2[category]['mean']
            std2 = metrics2[category]['std']
            
            # Calculate Z-statistic
            pooled_se = np.sqrt((std1**2/n_samples) + (std2**2/n_samples))
            z_stat = (mean1 - mean2) / pooled_se
            
            # Calculate p-value (two-tailed test)
            p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
            
            results[category] = {
                'z_statistic': z_stat,
                'p_value': p_value,
                'significant': p_value < alpha,
                'mean_difference': mean1 - mean2,
                'file1_mean': mean1,
                'file2_mean': mean2
            }
    
    return results

def write_comparison_results(results, file1_name, file2_name, output_file):
    """
    Write the comparison results to a file.
    """
    with open(output_file, 'w') as f:
        f.write(f"Statistical Comparison Results\n")
        f.write(f"File 1: {file1_name}\n")
        f.write(f"File 2: {file2_name}\n")
        f.write("=" * 80 + "\n\n")
        
        for category, result in results.items():
            if category != "Total number of entries":  # Skip the total entries info
                f.write(f"\n{category}:\n")
                f.write(f"Mean difference ({file1_name} - {file2_name}): {result['mean_difference']:.3f}\n")
                f.write(f"Z-statistic: {result['z_statistic']:.3f}\n")
                f.write(f"P-value: {result['p_value']:.4f}\n")
                f.write(f"Statistically significant: {'Yes' if result['significant'] else 'No'}\n")
                f.write(f"Means: {file1_name}: {result['file1_mean']:.2f}, {file2_name}: {result['file2_mean']:.2f}\n")
                f.write("-" * 40 + "\n")

def main():
    # Select input files
    file1, file2 = select_files()
    
    # Get output filename from user
    output_file = get_output_filename()
    
    # Extract file names for display
    file1_name = file1.split('/')[-1].split('\\')[-1]
    file2_name = file2.split('/')[-1].split('\\')[-1]
    
    # Perform comparison
    results = compare_metrics(file1, file2)
    
    # Write results to file
    write_comparison_results(results, file1_name, file2_name, output_file)
    
    print(f"\nResults have been written to: {output_file}")

if __name__ == "__main__":
    main()
