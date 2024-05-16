#!/bin/bash
cd ~/metrics

for i in {2..8}
do
    j=$((2**$i))
    echo "dog rank $j"
    echo "CLIP-T"
    python text-image-similarity-score.py -p ../inference/generated_images/dog/t2i/qkv_rank_$j
    echo "CLIP-I"
    python image-image-similarity-score.py -g ../inference/generated_images/dog/i2i/qkv_rank_$j -r ../inference/generated_images/dog/source
    echo "G-R"
    python image-image-similarity-score.py -g ../inference/generated_images/dog/t2i/qkv_rank_$j -r ../inference/generated_images/dog/source

    echo "backpack rank $j"
    echo "CLIP-T"
    python text-image-similarity-score.py -p ../inference/generated_images/backpack/t2i/qkv_rank_$j
    echo "CLIP-I"
    python image-image-similarity-score.py -g ../inference/generated_images/backpack/i2i/qkv_rank_$j -r ../inference/generated_images/backpack/source
    echo "G-R"
    python image-image-similarity-score.py -g ../inference/generated_images/backpack/t2i/qkv_rank_$j -r ../inference/generated_images/backpack/source

    echo "cat rank $j"
    echo "CLIP-T"
    python text-image-similarity-score.py -p ../inference/generated_images/cat/t2i/qkv_rank_$j
    echo "CLIP-I"
    python image-image-similarity-score.py -g ../inference/generated_images/cat/i2i/qkv_rank_$j -r ../inference/generated_images/cat/source
    echo "G-R"
    python image-image-similarity-score.py -g ../inference/generated_images/cat/t2i/qkv_rank_$j -r ../inference/generated_images/cat/source
done