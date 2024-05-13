#!/bin/bash
cd ~/inference/dreambooth

python3 inference_lora.py -m ~/diffusers/examples/dreambooth/trained_lora_dog_rank_4/checkpoint-1000 -p ../dog_t2i_prompt.txt -s ../generated_images/dog/t2i/lora_rank_4
python3 inference_lora.py -m ~/diffusers/examples/dreambooth/trained_lora_dog_rank_4/checkpoint-1000 -p ../dog_i2i_prompt.txt -s ../generated_images/dog/i2i/lora_rank_4 -n 40

python3 inference_lora.py -m ~/diffusers/examples/dreambooth/trained_lora_backpack_rank_4/checkpoint-1000 -p ../backpack_t2i_prompt.txt -s ../generated_images/backpack/t2i/lora_rank_4
python3 inference_lora.py -m ~/diffusers/examples/dreambooth/trained_lora_backpack_rank_4/checkpoint-1000 -p ../backpack_i2i_prompt.txt -s ../generated_images/backpack/i2i/lora_rank_4 -n 40

python3 inference_lora.py -m ~/diffusers/examples/dreambooth/trained_lora_cat_rank_4/checkpoint-700 -p ../cat_t2i_prompt.txt -s ../generated_images/cat/t2i/lora_rank_4
python3 inference_lora.py -m ~/diffusers/examples/dreambooth/trained_lora_cat_rank_4/checkpoint-700 -p ../cat_i2i_prompt.txt -s ../generated_images/cat/i2i/lora_rank_4 -n 40