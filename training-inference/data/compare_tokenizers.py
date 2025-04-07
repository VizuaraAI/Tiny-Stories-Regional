from transformers import AutoTokenizer
import tiktoken


print("")
# SARVAM
hin =  AutoTokenizer.from_pretrained('sarvamai/sarvam-1')
# vocab
# print(f"Sarvam-2b-tokenizer vocab: {hin.get_vocab}") # detailed list
#print(f"Sarvam-2b-tokenizer vocab: {hin.vocab_size}") 
# eos
print(f"Sarvam-2b-tokenizer eos/end-of-text: {hin.eos_token}")
eos_pair = hin(hin.eos_token, return_tensors=None)
eos_id = eos_pair["input_ids"][1]
print(f"Sarvam-2b-tokenizer eos token ID: {eos_id}")
# Input
hin_text = "कर्नाटक की राजधानी है:"
print(f"Hindi input text: {hin_text}")
# Tokenize Example
hin_ids = hin(hin_text, return_tensors=None, add_special_tokens=False)["input_ids"] 
hin_ids.append(eos_id) # bos (beginning of sentence token automatically added i.e. no need to prepend it)
print(f"Hindi tokenzied: {hin_ids}")
# Tokens to text
text  = hin.convert_ids_to_tokens(hin_ids)
text2 = hin.decode(hin_ids)
print(f"Hindi ids to text (sanity-check-1): {text}")
print(f"Hindi ids to text (sanity-check-2): {text2}\n")

""""
RESULTS (SARVAM): 
Sarvam-2b-tokenizer vocab: 68096
Sarvam-2b-tokenizer eos/end-of-text: </s>
Sarvam-2b-tokenizer eos token ID: 2
Hindi input text: कर्नाटक की राजधानी है:
Hindi tokenzied: [1, 57169, 4508, 25049, 4432, 67736, 2]
Hindi ids to text (sanity-check): ['<s>', '▁कर्नाटक', '▁की', '▁राजधानी', '▁है', ':', '</s>']
कर्नाटक की राजधानी है:</s>
"""

# BPE
eng = tiktoken.get_encoding("gpt2")
eng_text = "Hello World! Nirvan here"
eng_ids  = eng.encode_ordinary(eng_text)
eng_ids.append(eng.eot_token) 
#test_ids = [2159, 0, 31487, 10438, 994, 50256] # ID 15496 = Hello
decoded_text = eng.decode(eng_ids) 
print(f"English input: {eng_text}")
print(f"Tiktoken eot_token ID: {eng.eot_token}")
print(f"English tokenzied: {eng_ids}")
print(f"Tiktoken decoded text: {decoded_text}\n")

"""
RESULTS (BPE GPT2)
English input: Hello World! Nirvan here
Tiktoken eos token ID: 50256
English tokenzied: [15496, 2159, 0, 31487, 10438, 994, 50256]
"""