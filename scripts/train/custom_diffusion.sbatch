#!/bin/bash
#SBATCH --job-name=train_custom_diffusion            # Заместо aboba напишите ваше незвание эксперимента
#SBATCH --error=runs/train_custom_diffusion-%j.err    # Файл для вывода ошибок
#SBATCH --output=runs/train_custom_diffusion-%j.out   # Файл для вывода результатов
#SBATCH --gpus=1                     # Количество запрашиваемых гпу
#SBATCH --cpus-per-task=1            # Выполнение расчёта на 8 ядрах CPU
#SBATCH --time=6:00:00               # Максимальное время выполнения (чч:мм:сс), после его окончания програмаа просто сбрасывается
#SBATCH --constraint="[type_a|type_b|type_c]"   # Выбор, какие узлы использовать. 
																		 # Узлы a, b и c - содержат v100, d без гпу, e - содержит a100, простым смертным недоступен (нужно писать Айбеку)

module load Python/Anaconda_v03.2023
module load CUDA/12.2

source activate cuda

cd ~/diffusers/examples/custom_diffusion

accelerate launch train_custom_diffusion.py --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" --instance_data_dir="/home/ldnigogosova/datasets/dreambooth/dataset/backpack" --output_dir="output_backpack" --with_prior_preservation --prior_loss_weight=1.0 --class_prompt="a backpack" --class_data_dir=./prior_images_backpack --instance_prompt="a sks backpack" --resolution=512 --train_batch_size=2 --learning_rate=1e-5 --lr_warmup_steps=0 --max_train_steps=1500 --scale_lr --hflip  --modifier_token "sks" --validation_prompt="a sks backpack in the jungle[EOS]a sks backpack on the beach[EOS]a sks backpack on top of a wooden floor[EOS]a wet sks backpack[EOS]a purple sks backpack" --validation_steps=200 --checkpointing_steps=200 --report_to="wandb" --no_safe_serialization