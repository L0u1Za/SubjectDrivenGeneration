cd ~/metrics

echo "dog"
echo "CLIP-T"
python text-image-similarity-score.py -p ../inference/generated_images/dog/t2i/textual_inversion
echo "CLIP-I"
python image-image-similarity-score.py -g ../inference/generated_images/dog/i2i/textual_inversion -r ../inference/generated_images/dog/source
echo "G-R"
python image-image-similarity-score.py -g ../inference/generated_images/dog/t2i/textual_inversion -r ../inference/generated_images/dog/source

echo "backpack"
echo "CLIP-T"
python text-image-similarity-score.py -p ../inference/generated_images/backpack/t2i/textual_inversion
echo "CLIP-I"
python image-image-similarity-score.py -g ../inference/generated_images/backpack/i2i/textual_inversion -r ../inference/generated_images/backpack/source
echo "G-R"
python image-image-similarity-score.py -g ../inference/generated_images/backpack/t2i/textual_inversion -r ../inference/generated_images/backpack/source

echo "cat"
echo "CLIP-T"
python text-image-similarity-score.py -p ../inference/generated_images/cat/t2i/textual_inversion
echo "CLIP-I"
python image-image-similarity-score.py -g ../inference/generated_images/cat/i2i/textual_inversion -r ../inference/generated_images/cat/source
echo "G-R"
python image-image-similarity-score.py -g ../inference/generated_images/cat/t2i/textual_inversion -r ../inference/generated_images/cat/source