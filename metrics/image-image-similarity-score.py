from transformers import CLIPModel, CLIPProcessor
from glob import glob
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--real', required=True)
    parser.add_argument('-g', '--generated', required=True)
    args = parser.parse_args()

    real_path = args.real + '/*' # '../images/*'
    generated_path = args.generated + '/*' # '../images/*'

    images_real, images_generated = [], []
    for file in glob(real_path):
        images_real.append(Image.open(file))
    for file in glob(generated_path):
        filename, ext = os.path.basename(file).split('.')
        if ext == 'png':
            images_generated.append(Image.open(file))

    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    inputs_real = processor(
        images=images_real, return_tensors="pt", padding=True
    )
    inputs_generated = processor(
        images=images_generated, return_tensors="pt", padding=True
    )

    embeds_real = model.get_image_features(**inputs_real)
    embeds_generated = model.get_image_features(**inputs_generated)

    scores = []
    for i in range(len(embeds_real)):
        for j in range(len(embeds_generated)):
            score = cosine_similarity([embeds_real[i].cpu().detach().numpy()], [embeds_generated[j].cpu().detach().numpy()])
            scores.append(score)
    print(sum(scores) / len(scores))