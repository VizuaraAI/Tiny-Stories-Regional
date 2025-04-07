# Hugging Face
from transformers import AutoTokenizer
from datasets import load_dataset # huggingface datasets
# General
from tqdm import tqdm
import numpy as np
import tiktoken
import os


# number of workers in .map() call
# good number to use is ~order number of cpu cores // 2
num_proc = 16

# number of workers in load_dataset() call
num_proc_load_dataset = 1

################ Tokenizer Choice ################ 
# Ensure that any HF tokenizer name has "_hf" appended to it 
# eg. sutra_hf and NOT sutra (as sutra is from HF)
# eg. tik and NOT tik_hf (as tiktoken here isnt downloaded from HF)

tokenizer = "sutra_hf"         
if tokenizer == "sarvam_hf": 
    enc = AutoTokenizer.from_pretrained('sarvamai/sarvam-1')
    print("\nSarvam Tokenizer | bos: <s> , eos: </s>")
elif tokenizer == "sutra_hf":
    enc = AutoTokenizer.from_pretrained('TWO/sutra-mlt256-v2')
    print("\nSUTRA Tokenizer | bos: <s> , eos: </s>")
elif tokenizer == "tik": 
    enc = tiktoken.get_encoding("gpt2")
    print("\nTiktoken GPT-2 Tokenizer")

################ Tokenizer Choice ################ 


################ Dataset Choice ################

dataset_choice = "beng_4omini" 
if dataset_choice == "hin_4omini": dataset_name = "TinyStories-Regional/hindi-generated_4o-mini_2M"
elif dataset_choice == "mar_4omini": dataset_name = "TinyStories-Regional/marathi-generated_4o-mini_2M"
elif dataset_choice == "beng_4omini": dataset_name = "TinyStories-Regional/beng-generated_4o-mini_2M"
elif dataset_choice == "hin_trans": dataset_name = "TinyStories-Regional/hindi-translated_nllb600M_1M"
elif dataset_choice == "beng_trans": dataset_name = "TinyStories-Regional/beng-translated_deeptrans_1M"
print(dataset_name)

################ Dataset Choice ################

# load dataset
dataset = load_dataset(dataset_name, num_proc=num_proc_load_dataset)

# sanity check
if "trans" not in dataset_choice: 
    print(f"\nSanity Check: {dataset['train']['story'][16]}\n")
    print(f"Number of stories: {len(dataset['train'])}\n")
else: 
    print(f"\nSanity Check: {dataset['train']['Translated Story'][16]}")
    print(f"Number of stories: {len(dataset['train'])}\n")

# only 'train' split present, so create a test split
split_dataset = dataset['train'].train_test_split(test_size=0.025, seed=2357, shuffle=True)
split_dataset['val'] = split_dataset.pop('test') # rename the test split to val

# this results in:
# >>> split_dataset
# DatasetDict({
#     train: Dataset({
#         features: ['text'],
#         num_rows: 8009762
#     })
#     val: Dataset({
#         features: ['text'],
#         num_rows: 4007
#     })
# })

# sanity check 
sanity_check_ids = False

# we now want to tokenize the dataset. 
def process(example, tokenizer=tokenizer):
    
    # Sanity check of dataset
    if sanity_check_ids: print(f'\nexmaple is: {example}\n')
    
    # English (Tiktoken)
    if tokenizer == "tik": 
        ids = enc.encode_ordinary(example['story']) # encode_ordinary ignores any special tokens
        ids.append(enc.eot_token) # Add the end of text token
        
    # Hindi (Sarvam)
    elif "hf" in tokenizer: 
        # Dataset choice
        if "trans" not in dataset_choice: col = 'story'
        else: col = 'Translated Story'

        # Tokenize
        pairs_ids_masks = enc(example[col], return_tensors=None)
        
        # HF 
        ids = pairs_ids_masks["input_ids"]
        if "sutra" in tokenizer: ids[0] = 256002
        if "sutra" not in tokenizer: ids.append(enc.eos_token_id) 
        
        if sanity_check_ids: 
            print(f'token ids are: {ids}\n')
            while True: pass
            
    # Return
    out = {'ids': ids, 'len': len(ids)}
    return out

# tokenize the dataset
if "mar" in dataset_choice: 
    tokenized = split_dataset.map(
    process,
    remove_columns=['story'],
    desc="Tokenizing the splits",
    num_proc=num_proc,
)
elif "trans" not in dataset_choice: 
    tokenized = split_dataset.map(
    process,
    remove_columns=['story', 'ID'],
    desc="Tokenizing the splits",
    num_proc=num_proc,
)
else: 
    tokenized = split_dataset.map(
        process,
        remove_columns=['Translated Story'],
        desc="Tokenizing the splits",
        num_proc=num_proc,
    )

# concatenate all the ids in each dataset into one large file we can use for training
for split, dset in tokenized.items():
    
    arr_len = np.sum(dset['len'], dtype=np.uint64)
    filename = os.path.join(os.path.dirname(__file__), f'{split}.bin')
    dtype = np.uint32 # tiktoken size = 50,256 (unit16 i.e. 2**16 = 65k is enough) | sarvam size = ~68k (unit32 required)
    arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
    total_batches = 1024

    idx = 0
    for batch_idx in tqdm(range(total_batches), desc=f'writing {filename}'):
        # Batch together samples for faster write
        batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
        arr_batch = np.concatenate(batch['ids'])
        # Write into mmap
        arr[idx : idx + len(arr_batch)] = arr_batch
        idx += len(arr_batch)
    arr.flush()