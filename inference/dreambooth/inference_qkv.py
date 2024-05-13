from accelerate import Accelerator
from diffusers import StableDiffusionPipeline, UNet2DConditionModel, DDIMScheduler
from diffusers.models.cross_attention import LoRACrossAttnProcessor
from torch import nn
import torch
import tensorly as tl
tl.set_backend('pytorch')

import argparse
import os

def apply_svd_lora(model_id, args):
	unet = UNet2DConditionModel.from_pretrained("/home/ldnigogosova/stable-diffusion-2-base", subfolder="unet").to("cuda")
	original_weights = {}
	for (name, param) in unet.named_parameters():
		if 'to_q' in name or 'to_k' in name or 'to_v' in name:
			linear_name = name.split('.')[-2]
			name = '.'.join(name.split('.')[:-2])
			if name in original_weights:
				original_weights[name][linear_name] = param.clone().detach()
			else:
				original_weights[name] = {linear_name: param.clone().detach()}

	finetuned_unet = UNet2DConditionModel.from_pretrained(model_id, use_safetensors=True).to("cuda")
	finetuned_weights = {}
	for (name, param) in finetuned_unet.named_parameters():
		if 'to_q' in name or 'to_k' in name or 'to_v' in name:
			linear_name = name.split('.')[-2]
			name = '.'.join(name.split('.')[:-2])
			if name in finetuned_weights:
				finetuned_weights[name][linear_name] = param.clone().detach()
			else:
				finetuned_weights[name] = {linear_name: param.clone().detach()}

	lora_attn_procs = {}
	for name in unet.attn_processors.keys():
		cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
		if name.startswith("mid_block"):
			hidden_size = unet.config.block_out_channels[-1]
		elif name.startswith("up_blocks"):
			block_id = int(name[len("up_blocks.")])
			hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
		elif name.startswith("down_blocks"):
			block_id = int(name[len("down_blocks.")])
			hidden_size = unet.config.block_out_channels[block_id]

		if cross_attention_dim:
			rank = min(cross_attention_dim, hidden_size, args.rank)
		else:
			rank = min(hidden_size, args.rank)

		lora_attn_procs[name] = LoRACrossAttnProcessor(
			hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, rank=rank
		)
		layer_name = '.'.join(name.split('.')[:-1])
		for ll in ['to_q', 'to_k', 'to_v']:
			w1 = finetuned_weights[layer_name][ll]
			w0 = original_weights[layer_name][ll]
			delta = w1 - w0
			if ll == 'to_q':
				u, s, vh = tl.truncated_svd(delta, n_eigenvecs=rank)
				dw = torch.diag(torch.sqrt(s)) @ vh
				uw = u @ torch.diag(torch.sqrt(s))
				lora_attn_procs[name].to_q_lora.down.weight = nn.Parameter(dw.clone().to(dtype=torch.float32))
				lora_attn_procs[name].to_q_lora.up.weight = nn.Parameter(uw.clone().to(dtype=torch.float32))
			elif ll == 'to_v':
				u, s, vh = tl.truncated_svd(delta, n_eigenvecs=rank)
				dw = torch.diag(torch.sqrt(s)) @ vh
				uw = u @ torch.diag(torch.sqrt(s))
				lora_attn_procs[name].to_v_lora.down.weight = nn.Parameter(dw.clone().to(dtype=torch.float32))
				lora_attn_procs[name].to_v_lora.up.weight = nn.Parameter(uw.clone().to(dtype=torch.float32))
			elif ll == 'to_k':
				u, s, vh = tl.truncated_svd(delta, n_eigenvecs=rank)
				dw = torch.diag(torch.sqrt(s)) @ vh
				uw = u @ torch.diag(torch.sqrt(s))
				lora_attn_procs[name].to_v_lora.down.weight = nn.Parameter(dw.clone().to(dtype=torch.float32))
				lora_attn_procs[name].to_v_lora.up.weight = nn.Parameter(uw.clone().to(dtype=torch.float32))
	unet.set_attn_processor(lora_attn_procs)
	return unet

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('-m', '--model', required=True)
	parser.add_argument('-s', '--save_dir', required=True)
	parser.add_argument('-p', '--prompt', required=True)
	parser.add_argument('-n', '--num_images_per_prompt', default=4)
	parser.add_argument('-r', '--rank', type=int, default=4)

	args = parser.parse_args()

	model_id = args.model + '/unet'
	save_dir = args.save_dir
	with open(args.prompt, 'r') as f:
		prompts = f.read().split('\n')
	num_images_per_prompt = int(args.num_images_per_prompt)

	# Use text_encoder if `--train_text_encoder` was used for the initial training
	unet = apply_svd_lora(model_id, args)

	# Rebuild the pipeline with the unwrapped models (assignment to .unet and .text_encoder should work too)
	pipe = StableDiffusionPipeline.from_pretrained(
		"/home/ldnigogosova/stable-diffusion-2-base",
		unet=unet,
		safety_checker = None
	)
	pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
	pipe.to("cuda")

	for prompt in prompts:
		for i in range(num_images_per_prompt):
			image = pipe(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]

			image.save(f"{save_dir}/{prompt}_{i}.png")
			with open(f"{save_dir}/{prompt}_{i}.txt", "w") as f:
				f.write(prompt)
