#!/bin/bash
cd ~/inference/textual_inversion

python3 inference.py -m ~/diffusers/examples/textual_inversion/textual_inversion_dog/ -p ../dog_t2i_prompt.txt -s ../generated_images/dog/t2i/textual_inversion
python3 inference.py -m ~/diffusers/examples/textual_inversion/textual_inversion_dog -p ../dog_i2i_prompt.txt -s ../generated_images/dog/i2i/textual_inversion -n 40


python3 inference.py -m ~/diffusers/examples/textual_inversion/textual_inversion_backpack -p ../backpack_t2i_prompt.txt -s ../generated_images/backpack/t2i/textual_inversion
python3 inference.py -m ~/diffusers/examples/textual_inversion/textual_inversion_backpack -p ../backpack_i2i_prompt.txt -s ../generated_images/backpack/i2i/textual_inversion -n 40

python3 inference.py -m ~/diffusers/examples/textual_inversion/textual_inversion_cat -p ../cat_t2i_prompt.txt -s ../generated_images/cat/t2i/textual_inversion
python3 inference.py -m ~/diffusers/examples/textual_inversion/textual_inversion_cat -p ../cat_i2i_prompt.txt -s ../generated_images/cat/i2i/textual_inversion -n 40