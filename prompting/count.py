import os
import json

# Specify the directory containing the JSON files
directory = "stories_complete/marathi"
startswith = "stories"

# Read and display
print("")
for filename in sorted(os.listdir(directory)):
    # Check for files that match the correct naming pattern
    if filename.endswith(".json") and filename.startswith(startswith) and "copy" not in filename:
        file_path = os.path.join(directory, filename)
        # Load data from each JSON file and count the number of stories
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)
            print(f"{filename}: {len(data)} stories/prompts")
            print(data[len(data)-100])
            print("\n")
            with open(f"story_samples/{filename[:-5]}-[1000].json", 'w', encoding="utf-8") as file:
                json.dump(data[-100:-51], file, ensure_ascii=False, indent=4)