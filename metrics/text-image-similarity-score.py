from transformers import CLIPModel, CLIPProcessor, CLIPTokenizer
from glob import glob
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import argparse
import os
import re

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', required=True)
    args = parser.parse_args()

    path = args.path + '/*' # 'data/test_images/*'


    banned_words = ['sks', 'cat', 'dog', 'backpack']
    images = {}
    for file in glob(path):
        filename, ext = os.path.basename(file).split('.')
        if ext == 'txt':
            if filename in images:
                with open(file, 'r') as f:
                    images[filename]["prompt"] = f.read()
            else:
                with open(file, 'r') as f:
                    images[filename] = {
                        "prompt": f.read()
                    }
            for banned_word in banned_words:
                images[filename]["prompt"] = images[filename]["prompt"].replace(banned_word, '')
            images[filename]["prompt"] = re.sub(' +', ' ', images[filename]["prompt"])
        else:
            if filename in images:
                images[filename]["image"] = Image.open(file)
            else:
                images[filename] = {
                    "image": Image.open(file)
                }

    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    inputs_text = tokenizer(
        [it["prompt"] for it in images.values()], return_tensors="pt", padding=True
    )
    inputs_image = processor(
        images=[it["image"] for it in images.values()], return_tensors="pt", padding=True
    )

    embeds_text = model.get_text_features(**inputs_text)
    embeds_image = model.get_image_features(**inputs_image)

    scores = []
    for i in range(len(embeds_text)):
        score = cosine_similarity([embeds_text[i].cpu().detach().numpy()], [embeds_image[i].cpu().detach().numpy()])
        scores.append(score)
    print(sum(scores) / len(scores))