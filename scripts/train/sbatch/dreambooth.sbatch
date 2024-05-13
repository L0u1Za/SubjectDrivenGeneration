#!/bin/bash
#SBATCH --job-name=train_dreambooth           # Заместо aboba напишите ваше незвание эксперимента
#SBATCH --error=runs/train_dreambooth-%j.err    # Файл для вывода ошибок
#SBATCH --output=runs/train_dreambooth-%j.out   # Файл для вывода результатов
#SBATCH --gpus=1                     # Количество запрашиваемых гпу
#SBATCH --cpus-per-task=1            # Выполнение расчёта на 8 ядрах CPU
#SBATCH --time=6:00:00               # Максимальное время выполнения (чч:мм:сс), после его окончания програмаа просто сбрасывается
#SBATCH --constraint="[type_a|type_b|type_c]"   # Выбор, какие узлы использовать.
																		 # Узлы a, b и c - содержат v100, d без гпу, e - содержит a100, простым смертным недоступен (нужно писать Айбеку)

module load Python/Anaconda_v03.2023
module load CUDA/12.2

source activate cuda

cd ~/diffusers/examples/dreambooth

accelerate launch train_dreambooth.py --pretrained_model_name_or_path="/home/ldnigogosova/stable-diffusion-2-base" --instance_data_dir="/home/ldnigogosova/datasets/dreambooth/dataset/backpack" --class_data_dir="dreambooth_prior_images_backpack" --output_dir="trained_dreambooth_backpack" --with_prior_preservation --prior_loss_weight=1.0 --instance_prompt="a sks backpack" --class_prompt="a backpack" --resolution=512 --train_batch_size=4 --gradient_accumulation_steps=1 --learning_rate=5e-6 --lr_scheduler="constant" --lr_warmup_steps=0 --max_train_steps=2000 --report_to="wandb" --validation_prompt="a sks backpack in the jungle[EOS]a sks backpack on the beach[EOS]a sks backpack on top of a wooden floor[EOS]a wet sks backpack[EOS]a purple sks backpack" --validation_steps=250 --checkpointing_steps=250 --seed="0" --num_class_images=1000 --sample_batch_size=10
