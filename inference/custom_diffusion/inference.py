
from diffusers import StableDiffusionPipeline, DDIMScheduler
import torch
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


	pipe = StableDiffusionPipeline.from_pretrained("/home/ldnigogosova/stable-diffusion-2-base", torch_dtype=torch.float16, safety_checker = None)
	pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
	pipe.to("cuda")
	pipe.load_textual_inversion(model_id, weight_name="sks.bin")
	pipe.unet.load_attn_procs(model_id, weight_name="pytorch_custom_diffusion_weights.bin")

	for prompt in prompts:
		for i in range(num_images_per_prompt):
			image = pipe(prompt, num_inference_steps=100, guidance_scale=6.0, eta=1.0).images[0]
			image.save(f"{save_dir}/{prompt}_{i}.png")
			with open(f"{save_dir}/{prompt}_{i}.txt", "w") as f:
				f.write(prompt)
