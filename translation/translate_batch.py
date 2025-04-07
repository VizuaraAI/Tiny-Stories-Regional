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

def translate_batch(batch_texts, tokenizer, model, model_name, distributed_state):
    """
    Translates a batch of texts from English to Hindi.
    Args:
        batch_texts (list of str): List of English texts to translate.
        tokenizer: The tokenizer corresponding to the translation model.
        model: The translation model.
        device (str): Device to run the model on ('cuda' or 'cpu').

    Returns:
        list of str: Translated Hindi texts.
    """

    # Device
    device = distributed_state.device

    # Ensure all texts are strings | Decode if required
    cleaned_texts = []
    for idx, text in enumerate(batch_texts):
        # Text already string
        if isinstance(text, str):
            cleaned_texts.append(text)
        # Text is decoded string
        elif isinstance(text, bytes):
            try:
                decoded_text = text.decode('utf-8')
                cleaned_texts.append(decoded_text)
            except UnicodeDecodeError:
                print(f"Warning: Text at index {idx} could not be decoded. Replacing with empty string.")
                cleaned_texts.append("")
        # Non-string
        else:
            print(f"Warning: Text at index {idx} is of type {type(text)}. Replacing with empty string.")
            cleaned_texts.append("")

    
    # Tokenizing input
    if model_name == "facebook/mbart-large-50-many-to-many-mmt": 
        inputs = tokenizer(cleaned_texts, return_tensors="pt", padding=True, truncation=True).to(device)
        # Set the source language to English and target language to Hindi
        tokenizer.src_lang = "en_XX"  # English language
        tokenizer.tgt_lang = "hi_IN"  # Hindi language
    
    # IndicTrans2 
    # elif model_name == "ai4bharat/indictrans2-en-indic-1B":
    #     # Inference
    #     ip = IndicProcessor(inference=True)
    #     # Target and Source Lang
    #     src_lang, tgt_lang = "eng_Latn", "hin_Deva"
    #     # Forming batch
    #     batch = ip.preprocess_batch(
    #         cleaned_texts,
    #         src_lang=src_lang,
    #         tgt_lang=tgt_lang,
    #     )
    #     # Tokenizing
    #     inputs = tokenizer(
    #         batch,
    #         truncation=True,
    #         padding="longest",
    #         return_tensors="pt",
    #         return_attention_mask=True,
    #     ).to(device)

    # Helsinki-NLP/opus-mt-en-hi, facebook/nllb <variants>, facebook/m2m100_418M
    else:
        inputs = tokenizer(cleaned_texts, return_tensors="pt", padding=True, truncation=True).to(device)
        
        
    # Autocast
    if device.type == 'cuda':
        autocast_context = torch.cuda.amp.autocast()
    else:
        from contextlib import nullcontext
        autocast_context = nullcontext()

    # Inferencing LLM
    with torch.inference_mode(), autocast_context: # autocast handles mix-precision tarining and inferencing (no need to explicitly convert inputs from FP32 to FP116) 
        if model_name == "facebook/mbart-large-50-many-to-many-mmt": 
            outputs = model.generate(
                **inputs,
                forced_bos_token_id=tokenizer.lang_code_to_id[tokenizer.tgt_lang],  # Use Hindi as target language
                num_beams=3,          # More num_beams more possiblities for translation considered
                max_length=500,       # Adjust based on expected translation length
                early_stopping=True
            ) 
            
        elif model_name == "ai4bharat/indictrans2-en-indic-1B":
            outputs = model.generate(
                **inputs,
                use_cache=True,
                min_length=0,
                max_length=500,
                num_beams=5,
                num_return_sequences=1
            )

        elif model_name == "facebook/m2m100_418M":
            outputs = model.generate(
                **inputs, 
                forced_bos_token_id=tokenizer.get_lang_id("hi"), 
                num_beams=5
            )

        elif model_name == "facebook/nllb-200-distilled-600M" or "facebook/nllb-200-distilled-1.3B" or "facebook/nllb-200-3.3B":
            outputs = model.generate(
                **inputs, 
                forced_bos_token_id=tokenizer.convert_tokens_to_ids("hin_Deva"), 
                max_length=500,
                num_beams=5
            )
            
        else:
            outputs = model.generate(
                **inputs,
                num_beams=3,          # More num_beams more possiblities for translation considered
                max_length=500,       # Adjust based on expected translation length
                early_stopping=True
            )


    # Move outputs to CPU for decoding to save GPU memory
    outputs = outputs.cpu()

    # Converting output tokens to Natural Language
    # if model_name == "ai4bharat/indictrans2-en-indic-1B":
    #     # Decode the generated tokens into text
    #     with tokenizer.as_target_tokenizer():
    #         hindi_tokens = tokenizer.batch_decode(
    #             outputs,
    #             skip_special_tokens=True,
    #             clean_up_tokenization_spaces=True,
    #         )
    #     # Postprocess the translations, including entity replacement
    #     translated_texts = ip.postprocess_batch(hindi_tokens, lang=tgt_lang)
        
    translated_texts = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

    
    return translated_texts