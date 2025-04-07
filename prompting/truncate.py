import os
import json

# Specify the directory containing the JSON files
directory = "stories_complete/marathi"
startswith = "stories"

# Truncate percentage
truncate_percentage = 69  # Adjust as needed

# Fixed original length
original_length = 166666  
new_length = max(int(original_length * (truncate_percentage / 100)), 1)  # Ensure at least 1 story remains

# Process files
print("")
for filename in sorted(os.listdir(directory)):
    if filename.endswith(".json") and filename.startswith(startswith) and "[" in filename:
        file_path = os.path.join(directory, filename)
        
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        truncated_data = data[:new_length]  # Keep only the first 'new_length' stories
        
        # Overwrite the file with truncated data
        with open(file_path, "w", encoding="utf-8") as file:
            json.dump(truncated_data, file, ensure_ascii=False, indent=4)
        
        print(f"\n{filename}: Truncated to {new_length} stories/prompts")
        print(truncated_data[len(truncated_data)-100])
