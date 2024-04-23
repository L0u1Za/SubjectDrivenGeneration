#!/bin/bash
#SBATCH --job-name=train_lora           # Заместо aboba напишите ваше незвание эксперимента
#SBATCH --error=runs/train_lora-%j.err    # Файл для вывода ошибок
#SBATCH --output=runs/train_lora-%j.out   # Файл для вывода результатов
#SBATCH --gpus=1                     # Количество запрашиваемых гпу
#SBATCH --cpus-per-task=1            # Выполнение расчёта на 8 ядрах CPU
#SBATCH --time=6:00:00               # Максимальное время выполнения (чч:мм:сс), после его окончания програмаа просто сбрасывается
#SBATCH --constraint="[type_a|type_b|type_c]"   # Выбор, какие узлы использовать.
																		 # Узлы a, b и c - содержат v100, d без гпу, e - содержит a100, простым смертным недоступен (нужно писать Айбеку)

module load Python/Anaconda_v03.2023
module load CUDA/12.2

source activate cuda

cd ~/diffusers/examples/dreambooth

accelerate launch train_dreambooth_lora.py --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" --instance_data_dir="/home/ldnigogosova/datasets/dreambooth/dataset/dog6" --class_data_dir="lora_prior_images_dog" --output_dir="trained_lora_dog_rank_8" --with_prior_preservation --prior_loss_weight=1.0 --instance_prompt="a sks dog" --class_prompt="a dog" --resolution=512 --train_batch_size=1 --gradient_accumulation_steps=1 --checkpointing_steps=100 --learning_rate=1e-4 --lr_scheduler="constant" --lr_warmup_steps=0 --max_train_steps=1000 --validation_prompt="a sks dog in the jungle[EOS]a sks dog on the beach[EOS]a sks dog on top of a wooden floor[EOS]a sks dog wearing a rainbow scarf[EOS]a purple sks dog" --validation_epochs=1 --seed="0" --report_to="wandb" --num_class_images=200 --rank=8