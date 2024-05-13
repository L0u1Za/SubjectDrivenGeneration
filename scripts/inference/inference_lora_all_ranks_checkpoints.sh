#!/bin/bash
cd ~/inference/dreambooth

for i in {2..8}
do
    j=$((2**$i))

    for c in 500 1000 2000
    do
        mkdir ../generated_images/dog/t2i/lora_checkpoint_${c}_rank_$j
        mkdir ../generated_images/dog/i2i/lora_checkpoint_${c}_rank_$j
        python3 inference_lora.py -m ~/diffusers/examples/dreambooth/trained_lora_dog_rank_$j/checkpoint-${c} -p ../dog_t2i_prompt.txt -s ../generated_images/dog/t2i/lora_checkpoint_${c}_rank_$j
        python3 inference_lora.py -m ~/diffusers/examples/dreambooth/trained_lora_dog_rank_$j/checkpoint-${c} -p ../dog_i2i_prompt.txt -s ../generated_images/dog/i2i/lora_checkpoint_${c}_rank_$j -n 40

        mkdir ../generated_images/backpack/t2i/lora_checkpoint_${c}_rank_$j
        mkdir ../generated_images/backpack/i2i/lora_checkpoint_${c}_rank_$j
        python3 inference_lora.py -m ~/diffusers/examples/dreambooth/trained_lora_backpack_rank_$j/checkpoint-${c} -p ../backpack_t2i_prompt.txt -s ../generated_images/backpack/t2i/lora_checkpoint_${c}_rank_$j
        python3 inference_lora.py -m ~/diffusers/examples/dreambooth/trained_lora_backpack_rank_$j/checkpoint-${c} -p ../backpack_i2i_prompt.txt -s ../generated_images/backpack/i2i/lora_checkpoint_${c}_rank_$j -n 40

        mkdir ../generated_images/cat/t2i/lora_checkpoint_${c}_rank_$j
        mkdir ../generated_images/cat/i2i/lora_checkpoint_${c}_rank_$j
        python3 inference_lora.py -m ~/diffusers/examples/dreambooth/trained_lora_cat_rank_$j/checkpoint-${c} -p ../cat_t2i_prompt.txt -s ../generated_images/cat/t2i/lora_checkpoint_${c}_rank_$j
        python3 inference_lora.py -m ~/diffusers/examples/dreambooth/trained_lora_cat_rank_$j/checkpoint-${c} -p ../cat_i2i_prompt.txt -s ../generated_images/cat/i2i/lora_checkpoint_${c}_rank_$j -n 40
    done
done