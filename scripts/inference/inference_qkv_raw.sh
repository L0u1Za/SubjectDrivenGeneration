#!/bin/bash
cd /home/ldnigogosova/inference/dreambooth

mkdir ../generated_images/dog/t2i/qkv_raw
mkdir ../generated_images/dog/i2i/qkv_raw
python3 inference_dreambooth.py -m /home/ldnigogosova/train/trained_dreambooth_qkv_dog/checkpoint-1500 -p ../dog_t2i_prompt.txt -s ../generated_images/dog/t2i/qkv_raw
python3 inference_dreambooth.py -m /home/ldnigogosova/train/trained_dreambooth_qkv_dog/checkpoint-1500 -p ../dog_i2i_prompt.txt -s ../generated_images/dog/i2i/qkv_raw -n 40

mkdir ../generated_images/backpack/t2i/qkv_raw
mkdir ../generated_images/backpack/i2i/qkv_raw
python3 inference_dreambooth.py -m /home/ldnigogosova/train/trained_dreambooth_qkv_backpack/checkpoint-1400 -p ../backpack_t2i_prompt.txt -s ../generated_images/backpack/t2i/qkv_raw
python3 inference_dreambooth.py -m /home/ldnigogosova/train/trained_dreambooth_qkv_backpack/checkpoint-1400 -p ../backpack_i2i_prompt.txt -s ../generated_images/backpack/i2i/qkv_raw -n 40

mkdir ../generated_images/cat/t2i/qkv_raw
mkdir ../generated_images/cat/i2i/qkv_raw
python3 inference_dreambooth.py -m /home/ldnigogosova/train/trained_dreambooth_qkv_cat/checkpoint-1500 -p ../cat_t2i_prompt.txt -s ../generated_images/cat/t2i/qkv_raw
python3 inference_dreambooth.py -m /home/ldnigogosova/train/trained_dreambooth_qkv_cat/checkpoint-1500 -p ../cat_i2i_prompt.txt -s ../generated_images/cat/i2i/qkv_raw -n 40