import torch
import weightwatcher as ww
from model import GPT, GPTConfig
import matplotlib.pyplot as plt
import pandas as pd

''' Put this code in the path / folder where model.py is located'''

'''
WeightWatcher is a diagnostic tool that analyzes deep neural network (DNN) layers,
providing data-free insights into model training quality using Random Matrix Theory
and statistical metrics like alpha
'''
# Load the checkpoint
checkpoint_file = r'~path/to_checkpoint_/model.pt'
checkpoint = torch.load(checkpoint_file, map_location=torch.device('cpu'))

# Inspect the checkpoint keys, model params
print(checkpoint.keys())
print(checkpoint['model_args'])

# Extract the model state dictionary
state_dict = checkpoint['model']

# Define the model configuration (ensure it matches the config used during training)
#Recommended to manually fill in after inspecting model checkpoint dictionary 
config = GPTConfig(vocab_size=68096, block_size=1024, n_layer=7, n_head=8, n_embd=1024)

# Initialize the model
model = GPT(config)

# Load the model state dict
missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

# Print missing and unexpected keys
print(f"Missing keys: {missing_keys}")
print(f"Unexpected keys: {unexpected_keys}")

# Initialize WeightWatcher
watcher = ww.WeightWatcher(model=model)

# Perform the analysis
details = watcher.analyze()

# Print the results
print(details)

# Save the dataframe to a spreadsheet
details.to_excel('weightwatcher_analysis.xlsx', index=False)

# Extract layer IDs and singular values
layer_ids = details['layer_id']
sv_max = details['sv_max']
sv_min = details['sv_min']

# Plot the singular values
plt.figure(figsize=(12, 6))
plt.plot(layer_ids, sv_max, label='Max Singular Value', marker='o')
plt.plot(layer_ids, sv_min, label='Min Singular Value', marker='o')
plt.xlabel('Layer ID')
plt.ylabel('Singular Value')
plt.title('Singular Values Across Layers')
plt.legend()
plt.grid(True)
plt.savefig('singular_values_across_layers.png')
plt.close()

# Extract warnings
warnings = details['warning']

# Count the occurrences of each warning type
warning_counts = warnings.value_counts()

# Plot the warning counts
plt.figure(figsize=(8, 6))
warning_counts.plot(kind='bar')
plt.xlabel('Warning Type')
plt.ylabel('Count')
plt.title('Warnings Across Layers')
plt.grid(True)
plt.savefig('warnings_across_layers.png')
plt.close()

# Extract weak rank loss
weak_rank_loss = details['weak_rank_loss']

# Plot weak rank loss
plt.figure(figsize=(12, 6))
plt.plot(layer_ids, weak_rank_loss, label='Weak Rank Loss', marker='o')
plt.xlabel('Layer ID')
plt.ylabel('Weak Rank Loss')
plt.title('Weak Rank Loss Across Layers')
plt.legend()
plt.grid(True)
plt.savefig('weak_rank_loss_across_layers.png')
plt.close()

#plot alpha (the power law exponent)
# Extract layer IDs and alpha values
layer_ids = details['layer_id']
alpha_values = details['alpha']

# Create the plot
plt.figure(figsize=(12, 6))
plt.plot(layer_ids, alpha_values, label='Alpha (Power Law Exponent)', marker='o', linestyle='-')

# Add horizontal reference lines for different alpha regions
plt.axhline(y=4, color='red', linestyle='--', label='High Alpha (>4)')
plt.axhline(y=2, color='red', linestyle='-.', label='Balanced Alpha (~2)')
plt.axhline(y=1, color='red', linestyle=':', label='Low Alpha (<1)')

# Labels and title
plt.xlabel('Layer ID')
plt.ylabel('Alpha Value')
plt.title('Alpha Values Across Layers')
plt.legend()
plt.grid(True)

# Save and show the plot
plt.savefig('alpha_values_across_layers.png')  
plt.show()


print("Analysis results saved as 'weightwatcher_analysis.xlsx' and plots saved as PNG files.")
