#!/bin/bash
#SBATCH --job-name=train_svdiff          # Заместо aboba напишите ваше незвание эксперимента
#SBATCH --error=runs/train_svdiff-%j.err    # Файл для вывода ошибок
#SBATCH --output=runs/train_svdiff-%j.out   # Файл для вывода результатов
#SBATCH --gpus=1                     # Количество запрашиваемых гпу
#SBATCH --cpus-per-task=1            # Выполнение расчёта на 8 ядрах CPU
#SBATCH --time=6:00:00               # Максимальное время выполнения (чч:мм:сс), после его окончания програмаа просто сбрасывается
#SBATCH --constraint="[type_a|type_b|type_c]"   # Выбор, какие узлы использовать. 
																		 # Узлы a, b и c - содержат v100, d без гпу, e - содержит a100, простым смертным недоступен (нужно писать Айбеку)

module load Python/Anaconda_v03.2023
module load CUDA/12.2

source activate svdiff

cd ~/svdiff-pytorch

accelerate launch train_svdiff.py --pretrained_model_name_or_path="/home/ldnigogosova/stable-diffusion-2-base" --instance_data_dir="/home/ldnigogosova/datasets/dreambooth/dataset/backpack" --class_data_dir="prior_images_backpack" --output_dir="output_backpack" --with_prior_preservation --prior_loss_weight=1.0 --instance_prompt="a sks backpack" --class_prompt="a backpack" --resolution=512 --train_batch_size=4 --gradient_accumulation_steps=1 --learning_rate=1e-3 --learning_rate_1d=1e-6 --train_text_encoder --lr_scheduler="constant" --lr_warmup_steps=0 --num_class_images=200 --max_train_steps=2000 --report_to="wandb" --validation_prompt="a sks backpack in the jungle[EOS]a sks backpack on the beach[EOS]a sks backpack on top of a wooden floor[EOS]a wet sks backpack[EOS]a purple sks backpack" --validation_epochs=1 --checkpointing_steps=200 --seed="0"