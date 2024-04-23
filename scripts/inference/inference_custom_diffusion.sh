#!/bin/bash
cd ~/inference/custom_diffusion

python3 inference.py -m ~/diffusers/examples/custom_diffusion/output_dog/checkpoint-800 -p ../dog_t2i_prompt.txt -s ../generated_images/dog/t2i/custom_diffusion
python3 inference.py -m ~/diffusers/examples/custom_diffusion/output_dog/checkpoint-800 -p ../dog_i2i_prompt.txt -s ../generated_images/dog/i2i/custom_diffusion -n 40


python3 inference.py -m ~/diffusers/examples/custom_diffusion/output_backpack/checkpoint-1200 -p ../backpack_t2i_prompt.txt -s ../generated_images/backpack/t2i/custom_diffusion
python3 inference.py -m ~/diffusers/examples/custom_diffusion/output_backpack/checkpoint-1200 -p ../backpack_i2i_prompt.txt -s ../generated_images/backpack/i2i/custom_diffusion -n 40

python3 inference.py -m ~/diffusers/examples/custom_diffusion/output_cat/checkpoint-200 -p ../cat_t2i_prompt.txt -s ../generated_images/cat/t2i/custom_diffusion
python3 inference.py -m ~/diffusers/examples/custom_diffusion/output_cat/checkpoint-200 -p ../cat_i2i_prompt.txt -s ../generated_images/cat/i2i/custom_diffusion -n 40