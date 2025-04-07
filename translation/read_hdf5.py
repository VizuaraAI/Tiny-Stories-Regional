import h5py

# Read from saved file
# hdf5 = "translated_stories_2"
hdf5 = "hindi_translated_1.1M"
output_file = "samples.txt"

# Debug
skip = 26
stop = 72
p1 = True

print(f"\nFor {hdf5}:")
with h5py.File(f"{hdf5}.h5", "r") as hdf, open(output_file, "w", encoding="utf-8") as f:
    # Get all stories
    dataset = hdf["translated_stories"][:]
    if p1: print(f"{hdf5} has {len(dataset)} stories")
    
    # Iterate through each story
    for idx, story in enumerate(dataset):
        # Skip
        if idx < skip: continue
        
        original_story = story['original_story'].decode('utf-8')      # Decode if needed (for \n to render)
        translated_story = story['translated_story'].decode('utf-8')  # Decode the Hindi text (must)
        
        # Write to file
        f.write(f"Original Story {idx + 1}:\n{original_story}\n\n")
        f.write(f"Translated Story {idx + 1}:\n{translated_story}\n\n")
        f.write("----------------------------------------------------------------------\n\n")
        
        # Stop
        if idx == stop: break

print(f"Stories have been written to {output_file}.\n")
