from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
#from IndicTransToolkit import IndicProcessor
from huggingface_hub import get_full_repo_name
from datasets import load_dataset
from torch.amp import autocast
from tqdm.notebook import tqdm
import numpy as np
import torch
import json
import h5py
import time
import os

# Multi GPU
from accelerate import PartialState


# Save TinyStories to HDF5
def save_dataset_to_hdf5(dataset_name, split, hdf5_path):
    """
    Saves the 'text' field of a Hugging Face dataset to an HDF5 file.

    Args:
        dataset_name (str): The name of the dataset on Hugging Face.
        split (str): The split of the dataset to load (e.g., 'train', 'test').
        hdf5_path (str): The path where the HDF5 file will be saved.
    """

    # Loading Dataset
    print(f"Loading dataset '{dataset_name}' split '{split}'.")
    dataset = load_dataset(dataset_name, split=split)
    texts = dataset['text']

    # Saving Dataset to HDF5 file
    total_texts = len(texts)
    print(f"Number of texts to save: {total_texts}.")
    print(f"Saving texts to HDF5 file at '{hdf5_path}'.")
    with h5py.File(hdf5_path, "w") as h5f:
        # Define a variable-length string dtype
        dt = h5py.string_dtype(encoding='utf-8')
        # Create the dataset
        h5f.create_dataset("text", data=texts, dtype=dt)
    print(f"Dataset successfully saved to '{hdf5_path}'.")


# HDF5 setup
def create_output_hdf(output_hdf5, p1 = False):
    with h5py.File(f"{output_hdf5}.h5", "w") as hdf:
        # Define the data type for structured array (original_story and translated_story as variable-length UTF-8 strings)
        dt = np.dtype([('original_story', h5py.string_dtype(encoding='utf-8')),
                       ('translated_story', h5py.string_dtype(encoding='utf-8'))])
        # Use a consistent dataset name
        hdf.create_dataset("translated_stories", shape=(0,), maxshape=(None,), dtype=dt)
        if p1: print(f"{output_hdf5}.h5 created with dataset 'translated_stories'")


# Append to a given HDF5 file
def append_to_hdf5(output_hdf5, translated_data):
    """Appends a chunk of dictionary data to the HDF5 file."""
    with h5py.File(f"{output_hdf5}.h5", "a") as hdf:
        # Access the dataset using a consistent name
        translated_dataset = hdf["translated_stories"]
        # Convert list of dicts to structured array
        new_data = np.array([(d['original_story'], d['translated_story']) for d in translated_data], dtype=translated_dataset.dtype)
        # Resize dataset to accommodate new data
        translated_dataset.resize(translated_dataset.shape[0] + len(new_data), axis=0)
        # Append new data
        translated_dataset[-len(new_data):] = new_data


def combine_hdf5(input_foler, combined_hdf, num_gpu):
    """ Iterates through each hdf5 appending it to master hdf5 """
    # Master file path
    master_file = f"{input_foler}/{combined_hdf}.h5"
    
    # Iterate
    print(f"\nAppending {num_gpu} files to {combined_hdf}:")
    for i in range(num_gpu):
        # Current hdf5
        source_file = f"{input_foler}/translated_stories_{i}.h5"
        # Open current hdf5
        with h5py.File(source_file, "r") as src, h5py.File(master_file, 'a') as dest:
            # Access the dataset in the source file
            src_data = src["translated_stories"]
            # Access the dataset in the destination file
            dest_data = dest["translated_stories"]
            # Calculate the new size of the destination dataset
            new_size = dest_data.shape[0] + src_data.shape[0]
            print(f"{source_file} has {src_data.shape[0]} stories")
            # Resize the destination dataset to accommodate the new data
            dest_data.resize((new_size,))
            # Append the data from the source dataset to the destination dataset
            dest_data[-src_data.shape[0]:] = src_data[:]
    
    # Check combined file
    with h5py.File(master_file, "r") as master:
        stories = master["translated_stories"]
        print("Thus,")
        print(f"{master_file} contains {stories.shape[0]} stories")
        print("Appending Complete")