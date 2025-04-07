import sys

def update_model_name(new_model_value):
    # Define the file path and the new model_name value
    file_path = 'config.py'

    # Open the config.py file and read its contents
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Find and modify the line containing 'model_name'
    for i, line in enumerate(lines):
        if line.strip().startswith('model_name'):
            lines[i] = f"model_name = '{new_model_value}'\n"
            break

    # Write the modified content back to the file
    with open(file_path, 'w') as file:
        file.writelines(lines)

    print(f"\nmodel_name has been updated to '{new_model_value}' in {file_path}\n")

if __name__ == "__main__":
    # Get the model name from command-line arguments
    if len(sys.argv) > 1:
        model_name = sys.argv[1]
        update_model_name(model_name)
    else:
        print("Error: No model name provided.")
