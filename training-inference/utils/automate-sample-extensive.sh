#!/bin/bash

# Hindi Models
model_names_hindi=(
    "ckpt_sarvam_4.46M_val=1.408_[2M]_[l=2, h=8, e=64]"
    "ckpt_sarvam_4.65M_val=1.182_[2M]_[l=6, h=8, e=64]"
    "ckpt_sarvam_5M_val=1.057_[2M]_[l=12, h=8, e=64]"
    "ckpt_sarvam_9M_val=1.022_[2M]_[l=2, h=8, e=128]"
    "ckpt_sarvam_10M_val=0.864_[2M]_[l=6, h=8, e=128]"
    "ckpt_sarvam_19M_val=0.819_[2M]_[l=2, h=8, e=256]"
    "ckpt_sarvam_27M_val=0.618_[2M]_[l=12, h=8, e=256]"
    "ckpt_sarvam_41M_val=0.654_[2M]_[l=2, h=8, e=512]"
    "ckpt_sarvam_53M_val=0.518_[2M]_[l=6, h=8, e=512]"
    "ckpt_sarvam_73M_val=0.519_[2M]_[l=12_ h=8, e=512]"
    "ckpt_sarvam_66M_val=0.605_[2M]_[l=2, h=8, e=768]"
    "ckpt_sarvam_95M_val=0.517_[2M]_[l=6, h=8, e=768]"
    "ckpt_sarvam_94M_val=0.581_[2M]_[l=2, h=8, e=1024]"
    "ckpt_sarvam_153M_val=0.513_[2M]_[l=7, h=8, e=1024]"
)

# Hindi Models
model_names_marathi=(
    "ckpt_sarvam_4.46M_val=1.552_[2M]_[l=2, h=8, e=64]"
    "ckpt_sarvam_4.65M_val=1.351_[2M]_[l=6, h=8, e=64]"
    "ckpt_sarvam_4.95M_val=1.197_[2M]_[l=12, h=8, e=64]"
    "ckpt_sarvam_41.16M_val=0.731_[2M]_[l=2, h=8, e=512]"
    "ckpt_sarvam_54M_val=0.645_[2M]_[l=6, h=8, e=512]"
    "ckpt_sarvam_73M_val=0.603_[2M]_[l=12, h=8, e=512]"
    "ckpt_sarvam_95M_val=0.665_[2M]_[l=2, h=8, e=1024]"
    "ckpt_sarvam_157M_val=0.619_[2M]_[l=7, h=8, e=1024]"
)

# Hindi Models
model_names_beng=(
    "ckpt_sarvam_4.46M_val=1.514_[2M]_[l=2, h=8, e=64]"
    "ckpt_sarvam_4.65M_val=1.245_[2M]_[l=6, h=8, e=64]"
    "ckpt_sarvam_5M_val=1.136_[2M]_[l=12, h=8, e=64]"
    "ckpt_sarvam_41M_val=0.6932_[2M]_[l=2, h=8, e=512]"
    "ckpt_sarvam_54M_val=0.569_[2M]_[l=6, h=8, e=512]"
    "ckpt_sarvam_73M_val=0.544_[2M]_[l=12, h=8, e=512]"
    "ckpt_sarvam_95M_val=0.609_[2M]_[l=2, h=8, e=1024]"
    "ckpt_sarvam_157M_val=0.557_[2M]_[l=7, h=8, e=1024]"
)

# HINDI
for model_name in "${model_names_hindi[@]}"; do
    # Call the Python script with each model_name value
    python3 utils/update_model_name.py "$model_name"
    # Wait for 2 seconds
    sleep 1
    # Run inference
    python sample.py config.py --lang=hindi
    # Wait for 5 seconds before the next update
    sleep 2
done

# MARATHI
for model_name in "${model_names_marathi[@]}"; do
    # Call the Python script with each model_name value
    python3 utils/update_model_name.py "$model_name"
    # Wait for 2 seconds
    sleep 1
    # Run inference
    python sample.py config.py --lang=marathi
    # Wait for 5 seconds before the next update
    sleep 2
done

# BENGALI
for model_name in "${model_names_beng[@]}"; do
    # Call the Python script with each model_name value
    python3 utils/update_model_name.py "$model_name"
    # Wait for 2 seconds
    sleep 1
    # Run inference
    python sample.py config.py --lang=beng
    # Wait for 5 seconds before the next update
    sleep 2
done
