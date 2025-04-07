from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
#from IndicTransToolkit import IndicProcessor
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


def get_model_and_tokenizer(model_name, distributed_state, p1 = False):
    # Define a dictionary to simulate a switch statement for model choice
    model_dict = {
        "mbart": "facebook/mbart-large-50-many-to-many-mmt",
        "indictrans2": "ai4bharat/indictrans2-en-indic-1B",
        "mt5": "google/mt5-large",
        "opus_mt": "Helsinki-NLP/opus-mt-en-hi",
        "indicbart": "ai4bharat/IndicBART",
        "nllb": "facebook/nllb-200-distilled-600M",
        "nllb2": "facebook/nllb-200-distilled-1.3B",
        "nllb3": "facebook/nllb-200-3.3B",
        "m2m100": "facebook/m2m100_418M"
    }

    # Get the model name from the dictionary based on the input model choice
    model_name = model_dict.get(model_name.lower())

    if model_name is None:
        raise ValueError(f"Model {model_name} not recognized. Please choose from {', '.join(model_dict.keys())}.")

    # Device and GPUs
    device = distributed_state.device
    num_gpus = torch.cuda.device_count()
    if p1: 
        print(f"Using device: {device}")
        print(f"Number of GPUs available: {num_gpus}\n")
    
    # Load the model and tokenizer
    model     = AutoModelForSeq2SeqLM.from_pretrained(model_name, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if p1: print(f"Loading tokenizer and model '{model_name}'")
    
    # Prepare model and tokenizer for multi-GPU inference
    model.to(device)
    # Convert to FP16 if supported
    model.half()
    if p1: print("Model converted to FP16 for faster computation.\n");

    
    return model_name, model, tokenizer