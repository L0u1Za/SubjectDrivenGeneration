from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
import torch

from svdiff_pytorch import load_unet_for_svdiff, load_text_encoder_for_svdiff

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', required=True)
    parser.add_argument('-s', '--save_dir', required=True)
    parser.add_argument('-p', '--prompt', required=True)
    parser.add_argument('-n', '--num_images_per_prompt', default=4)

    args = parser.parse_args()

    model_id = args.model
    save_dir = args.save_dir
    with open(args.prompt, 'r') as f:
        prompts = f.read().split('\n')
    num_images_per_prompt = int(args.num_images_per_prompt)

    text_encoder = load_text_encoder_for_svdiff("runwayml/stable-diffusion-v1-5", spectral_shifts_ckpt=model_id, subfolder="text_encoder")
    unet = load_unet_for_svdiff("runwayml/stable-diffusion-v1-5", spectral_shifts_ckpt=model_id, subfolder="unet")

    # load pipe
    pipe = DiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        unet=unet,
        text_encoder=text_encoder,
        safety_checker = None
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.to("cuda")

    for prompt in prompts:
        for i in range(num_images_per_prompt):
            image = pipe(prompt, num_inference_steps=25).images[0]

            image.save(f"{save_dir}/{prompt}_{i}.png")
            with open(f"{save_dir}/{prompt}_{i}.txt", "w") as f:
                f.write(prompt)