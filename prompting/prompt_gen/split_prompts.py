import json
import os

def split_json(file_path, output_dir, start_index, end_index, divisions, language):
    """
    Splits a JSON file into specified number of divisions.

    Parameters:
        file_path (str): Path to the input JSON file.
        output_dir (str): Directory to save the output JSON files.
        start_index (int): Starting index for slicing the data.
        end_index (int): Ending index for slicing the data.
        divisions (int): Number of divisions to split the data into.
    """
    # Load the original JSON file
    with open(file_path, "r", encoding="utf-8") as f:
        prompts = json.load(f)

    # Calculate the slice size
    if end_index is None: end_index = len(prompts)-1
    total_size = end_index - start_index
    slice_size = total_size // divisions

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Split and save the data
    for i in range(divisions):
        slice_start = start_index + i * slice_size
        slice_end = slice_start + slice_size if i < divisions - 1 else end_index

        # Slice the data
        prompts_slice = prompts[slice_start:slice_end]

        # Generate file name
        output_file = os.path.join(
            output_dir, f"prompts_complex-2+-{language}-3M-[{slice_start}-{slice_end}].json"
        )

        # Write the slice to a JSON file
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(prompts_slice, f, ensure_ascii=False, indent=4)

        print(f"Saved slice {i+1}/{divisions} to {output_file}")


# Usage
language = "marathi" # hindi, beng, marathi
file_path = f"complete/prompts_complex-2+-{language}-[3M].json"
output_dir = "complete/split"
start_index = 0
end_index = 2000000    # if None = end of file
divisions = 12           
split_json(file_path, output_dir, start_index, end_index, divisions, language)
