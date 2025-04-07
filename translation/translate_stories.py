from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
#from IndicTransToolkit import IndicProcessor
from datasets import load_dataset
from torch.amp import autocast
from tqdm import tqdm
import numpy as np
import torch
import json
import h5py
import time
import os

# Multi GPU
from accelerate import PartialState

# Script Imports
from get_model_token import get_model_and_tokenizer
from gpu_memory import print_gpu_memory_usage
from translate_batch import translate_batch
from hdf5 import create_output_hdf
from hdf5 import append_to_hdf5

# Logging imports
from progress import send_email
from progress import OverwriteFile


def translate_stories(input_hdf5, output_hdf5, batch_size, chunk_size, model_name="nllb3", print_trans=False, debug=False, save_translations=True, gpu_mem=False, p1 = False, p2 = False):
    """
    Translates stories from English to Hindi using a specified model and tokenizer, leveraging Accelerate for multi-GPU inferencing.
    """
    
    # Debug settings
    if debug[0]:
        batch_size = debug[1]["batch_size"]
        chunk_size = debug[1]["chunk_size"]

    # Initialize the distributed state
    distributed_state = PartialState()
    device = distributed_state.device
    
    # Get process info
    num_processes = distributed_state.num_processes
    process_index = distributed_state.process_index

    # Load model and tokenizer
    model_name, model, tokenizer = get_model_and_tokenizer(model_name, distributed_state)

    # Start translation process
    start = time.time()
    translated_data = []
    incomplete = 0

    # Logging
    progress_file = OverwriteFile('logs.out', process_index)
    
    # Open the HDF5 file and read texts
    with h5py.File(input_hdf5, "r") as h5f:
        texts = h5f["text"]
        total_texts = texts.shape[0]
        if p2: print(f"Total texts to translate: {total_texts}\n")

        # Compute the indices for this process
        samples_per_process = total_texts // num_processes
        remainder = total_texts % num_processes
        # Logic to divide dataset
        if process_index < remainder:
            start_idx = process_index * (samples_per_process + 1)
            end_idx = start_idx + samples_per_process + 1
        else:
            start_idx = process_index * samples_per_process + remainder
            end_idx = start_idx + samples_per_process
        # Final Indices
        if end_idx > total_texts: end_idx = total_texts;
        num_local_texts = end_idx - start_idx

        # Only create output file if the process has data to process
        if num_local_texts > 0:
            # Modify output_hdf5 per process
            output_hdf5_process = f"{output_hdf5}_{distributed_state.process_index}"
            create_output_hdf(output_hdf5=output_hdf5_process)
        else:
            print(f"Process {distributed_state.process_index} has no data to process.")
            return  # Exit the function early
            
        # Now, get local texts
        local_texts = texts[start_idx:end_idx]
        num_batches = (num_local_texts + batch_size - 1) // batch_size
        if debug[0]: num_batches = debug[1]["num_batches"]

        # Translation loop
        progress_bar = tqdm(range(num_batches), 
                        desc=f"Process {distributed_state.process_index} Translating from {start_idx}:{end_idx}", unit="batch",
                        position=distributed_state.process_index, 
                        leave=True)
        for i in progress_bar:
            
            # Current batch
            start_idx = i * batch_size
            end_idx = min(start_idx + batch_size, num_local_texts)
            batch_texts = local_texts[start_idx:end_idx].tolist()

            # Translate the batch
            try:
                hindi_texts = translate_batch(batch_texts, tokenizer, model, model_name, distributed_state)
            except Exception as e:
                print(f"Error translating batch {i + 1}/{num_batches}: {e}")
                hindi_texts = [""] * len(batch_texts)
                incomplete += 1

            # Append translations
            translated_data.extend([
                {"original_story": english_text, "translated_story": hindi_text}
                for english_text, hindi_text in zip(batch_texts, hindi_texts)
            ])
            if print_trans:
                for idx, story in enumerate(translated_data):
                    eng = story["original_story"]
                    hin = story["translated_story"]
                    print(f"Original Story {(idx+1)*(i+1)}: {eng}\n")
                    print(f"Translated Story {(idx+1)*(i+1)}: {hin}\n")

            # HDF5 and Progress mail
            if save_translations and len(translated_data) >= chunk_size:
                # Appending to HDF5
                append_to_hdf5(output_hdf5_process, translated_data)
                translated_data.clear()  # Clear the list after writing to free memory
                # Logging
                progress_file.write(progress_bar.format_meter(i, num_batches, 1)) 
                # Sending progress mail
                with open('logs.out', 'r') as file:
                    body = file.read()
                    send_email(f"Translation Progress", body, "nirvan.ajit.patil@gmail.com")

            # GPU memory usage check
            if gpu_mem and i == num_batches - 1:
                print_gpu_memory_usage() 

        # Write remaining data
        if translated_data:
            append_to_hdf5(output_hdf5_process, translated_data)


    # Distributed translation process completed
    end = time.time()
    time.sleep(3)

    # Report 
    if incomplete > 0:
        print(f"There were {incomplete} incomplete translations. Rest saved.")
    print(f"Translation completed. Time required {distributed_state.process_index} = {end - start:.3f}s")
