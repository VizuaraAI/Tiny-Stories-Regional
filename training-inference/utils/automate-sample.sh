#!/bin/bash

# SUTRA
python3 utils/update_model_name.py "ckpt_sutra-beng_val=0.608"
sleep 1
python sample.py config.py --tokenizer=sutra_hf --lang=beng --max_new_tokens=350

sleep 5

python3 utils/update_model_name.py "ckpt_sutra-hindi_val=0.522"
sleep 1
python sample.py config.py --tokenizer=sutra_hf --lang=hindi --max_new_tokens=350


# Tiktoken
python3 utils/update_model_name.py "ckpt_tiktoken-beng_val=0.135"
sleep 1
python sample.py config.py --tokenizer=tik --lang=beng --max_new_tokens=2000

sleep 5

python3 utils/update_model_name.py "ckpt_tiktoken-hindi_val=0.149"
sleep 1
python sample.py config.py --tokenizer=tik --lang=hindi --max_new_tokens=2000

sleep 5

python3 utils/update_model_name.py "ckpt_tiktoken-marathi_val=0.167"
sleep 1
python sample.py config.py --tokenizer=tik --lang=marathi --max_new_tokens=2000



