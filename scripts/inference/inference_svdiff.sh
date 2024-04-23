#!/bin/bash
cd ~/inference/svdiff

python3 inference.py -m ~/svdiff-pytorch/output_dog/checkpoint-1400 -p ../dog_t2i_prompt.txt -s ../generated_images/dog/t2i/svdiff
python3 inference.py -m ~/svdiff-pytorch/output_dog/checkpoint-1400 -p ../dog_i2i_prompt.txt -s ../generated_images/dog/i2i/svdiff -n 40


python3 inference.py -m ~/svdiff-pytorch/output_backpack/checkpoint-1400 -p ../backpack_t2i_prompt.txt -s ../generated_images/backpack/t2i/svdiff
python3 inference.py -m ~/svdiff-pytorch/output_backpack/checkpoint-1400 -p ../backpack_i2i_prompt.txt -s ../generated_images/backpack/i2i/svdiff -n 40

python3 inference.py -m ~/svdiff-pytorch/output_cat/checkpoint-1000 -p ../cat_t2i_prompt.txt -s ../generated_images/cat/t2i/svdiff
python3 inference.py -m ~/svdiff-pytorch/output_cat/checkpoint-1000 -p ../cat_i2i_prompt.txt -s ../generated_images/cat/i2i/svdiff -n 40