#!/bin/bash

# Function to run the training command
run_training() {
  local n_layer=$1
  local n_embd=$2
  local batch_size=$3
  local learning_rate=$4
  local warmup_iters=$5
  echo ""
  echo "Running training with n_layer=${n_layer}, n_embd=${n_embd}, batch_size=${batch_size}, learning_rate=${learning_rate}, warmup_iters=${warmup_iters}"
  echo ""
  # torchrun --standalone --nproc_per_node=6 train.py config.py --n_layer=${n_layer} --n_head=8 --n_embd=${n_embd} --batch_size=${batch_size} --learning_rate=${learning_rate} --warmup_iters=${warmup_iters}
  python train.py config.py --n_layer=${n_layer} --n_head=8 --n_embd=${n_embd} --batch_size=${batch_size} --learning_rate=${learning_rate} --warmup_iters=${warmup_iters}
  python -c "import time; time.sleep(5)"
}

# Configuration 1: Embedding 64
run_training 2 64 144 5e-3 200
run_training 6 64 144 4.5e-3 200
run_training 12 64 140 4e-3 200

# Configuration 2: Embedding 512
run_training 2 512 128 1.4e-3 250
run_training 6 512 118 1.2e-3 300
run_training 12 512 102 1e-3 350

# Configuration 3: Embedding 1024
run_training 2 1024 112 0.65e-3 400
run_training 7 1024 96 0.6e-3 450
