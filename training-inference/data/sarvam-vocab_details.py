from transformers import AutoTokenizer
import tiktoken

print("")
# SARVAM
hin =  AutoTokenizer.from_pretrained('sarvamai/sarvam-1')
contents = hin.get_vocab  # Replace this with your actual data source

# Open a file in write mode and write the contents to it
with open('sarvam-vocab_details.txt', 'w', encoding='utf-8') as file:
    # If contents is a dictionary (common for vocab)
    if isinstance(contents, dict):
        for key, value in contents.items():
            file.write(f"{key}: {value}\n")
    # If contents is a list or any iterable
    elif isinstance(contents, (list, tuple, set)):
        for item in contents:
            file.write(f"{item}\n")
    # If contents is a string or other type, write it directly
    else:
        file.write(str(contents))

print("Contents have been written to 'sarvam-vocab_details.txt'")