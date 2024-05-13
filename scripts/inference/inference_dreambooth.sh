#!/bin/bash
cd ~/inference/dreambooth

python3 inference_dreambooth.py -m ~/diffusers/examples/dreambooth/trained_dreambooth_dog/checkpoint-600 -p ../dog_t2i_prompt.txt -s ../generated_images/dog/t2i/dreambooth
python3 inference_dreambooth.py -m ~/diffusers/examples/dreambooth/trained_dreambooth_dog/checkpoint-600 -p ../dog_i2i_prompt.txt -s ../generated_images/dog/i2i/dreambooth -n 40


python3 inference_dreambooth.py -m ~/diffusers/examples/dreambooth/trained_dreambooth_backpack/checkpoint-600 -p ../backpack_t2i_prompt.txt -s ../generated_images/backpack/t2i/dreambooth
python3 inference_dreambooth.py -m ~/diffusers/examples/dreambooth/trained_dreambooth_backpack/checkpoint-600 -p ../backpack_i2i_prompt.txt -s ../generated_images/backpack/i2i/dreambooth -n 40

python3 inference_dreambooth.py -m ~/diffusers/examples/dreambooth/trained_dreambooth_cat/checkpoint-600 -p ../cat_t2i_prompt.txt -s ../generated_images/cat/t2i/dreambooth
python3 inference_dreambooth.py -m ~/diffusers/examples/dreambooth/trained_dreambooth_cat/checkpoint-600 -p ../cat_i2i_prompt.txt -s ../generated_images/cat/i2i/dreambooth -n 40