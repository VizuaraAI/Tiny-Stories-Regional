import time 


############## TRAIN SETTINGS ##############

# ---------------------------------- #
# Data settings 
language = 'marathi'               # Choices: hindi, marathi, beng
tokenizer_train = 'sarvam'         # Tokenizer name
dataset = f'{language}-generated-{tokenizer_train}' 

# Eval only
eval_only = False                       # 1 epoch to eval model
if not eval_only: init_from = 'scratch' # scratch or resume

# Load Model
load_from_HF = False                    # Refer train.py line 143 for HF repository from where the SLMs is loaded if load_from_HF = True  
load_name = f'ckpt_sarvam_XM_val=Y_[2M]_[l=2, h=8, e=64]'   # Specify exact model (without .pt extension) name as it is on Hugging Face/saved locally 
#load_name = f'ckpt_{tokenizer_train}-{language}_val=0.609' 

# Wandb
if not eval_only: wandb_log = True
# ---------------------------------- #

# Model I/O settings
out_dir = 'out'         # Default output directory (make sure directory exists)
save_folder = language  # Folder in output directory ('.' to save in out_dir)

# Model parameters
n_layer = 7      # Default number of transformer layers
n_head = 8       # Default number of attention heads
n_embd = 1024    # Small 64 | Med 512 | Large 1024
dropout = 0.0    # Default dropout rate (no dropout)
bias = False
vocab_size = 68096    # Sarvam = 68096 | Tiktoken = 50304 | SUTRA = 256064 (rounded up to be divisible by 64)

# Training Intervals 
eval_interval = 200   # Evaluate after every 2000 iterations
log_interval = 2      # Log metrics after every iteration

# Training settings
batch_size = 96       # Default micro-batch size (136 for 5M | 128 for 66.4M | 100 for 72M | 88 for 150M)
block_size = 1024     # Default sequence length (context window size)
max_iters  = 5001     # Default total training iterations
eval_iters = 50       # Default iterations for evaluating validation loss
gradient_accumulation_steps = 40  # Must be divisible by num_gpu
always_save_checkpoint = False    # Always save checkpoint after evaluation

# Optimizer settings
learning_rate = 0.8e-3  # Max LR before decay (4e-3 <=10M | 1.5e-3 ~50M | 1e-3 70M+)
min_lr = 6e-5         # Default minimum learning rate
beta1 = 0.9           # Default AdamW beta1
beta2 = 0.95          # Default AdamW beta2
weight_decay = 1e-1   # Default weight decay regularization
grad_clip = 1.0       # Default gradient clipping valueclearc

# Learning rate decay
decay_lr = True                      # Enable cosine decay
warmup_iters = 450                   # 200 <=10M | 300 ~50M | 400 70M+
lr_decay_iters = int(max_iters*2)    # Default iterations over which to decay

# HF Settings
# HF_ACCESS_TOKEN = '' # For private repositories 

# WandB settings
wandb_project = f'Vizuara-marathi_generated_4o-mini_2M'  
#wandb_project = f'Vizuara-{language}_generated-tokenizers'  
WANDB_API_KEY = ''    # Your WANDBAPI Key

# Debug
if eval_only: debug_loss = True
else: debug_loss  = False   # True to print step and iter loss to stdout
debug_checkpoints = False   # True to print "saving checkpoint" whenever new best model found. Interrupts tqdm

# System
device = "cuda" # cuda or cpu
compile = True

# DDP settings
backend = 'nccl'  # Default backend for distributed data parallel


############## TEST/SAMPLE SETTINGS ##############

# I/O
lang  = "marathi"        # hindi, marathi, beng
output_mode = "json"     # json (for gpt-eval) or text  
start = f"FILE:prompt-{lang}.txt" 

# Tokenizer and load model
load_from_hf = False         # loads model from HF
tokenizer = "sarvam_hf"      # sarvam_hf or sutra_hf or tik
sample_init_from = "resume"  
model_name = 'ckpt_sarvam_XM_val=Y_[2M]_[l=7, h=8, e=1024]'

# Inference Settings
num_samples = 3           # number of samples to draw
max_new_tokens = 350      # sarvam and sutra = 350 | tik = 2000
temperature = 0.8         # 1.0 = no change, < 1.0 = less random
top_k = 15                # retain only the top_k most likely tokens
# top_p = 0.9             # sample from tokens with cum sum of prob >= 0.9
# rep_penalty = 1.2       # Slight penalty to reduce repeated words/phrases
# no_rep_ngram = 3        # Prevent immediate tri-gram repeats

##################################################
