#!/bin/bash
#SBATCH --job-name=train_textual_inversion           # Заместо aboba напишите ваше незвание эксперимента
#SBATCH --error=runs/train_textual_inversion-%j.err    # Файл для вывода ошибок
#SBATCH --output=runs/train_textual_inversion-%j.out  # Файл для вывода результатов
#SBATCH --gpus=1                     # Количество запрашиваемых гпу
#SBATCH --cpus-per-task=1            # Выполнение расчёта на 8 ядрах CPU
#SBATCH --time=6:00:00               # Максимальное время выполнения (чч:мм:сс), после его окончания програмаа просто сбрасывается
#SBATCH --constraint="[type_a|type_b|type_c]"   # Выбор, какие узлы использовать.
																		 # Узлы a, b и c - содержат v100, d без гпу, e - содержит a100, простым смертным недоступен (нужно писать Айбеку)

module load Python/Anaconda_v03.2023
module load CUDA/12.2

source activate cuda

cd ~/diffusers/examples/textual_inversion

accelerate launch textual_inversion.py --pretrained_model_name_or_path="/home/ldnigogosova/stable-diffusion-2-base" --train_data_dir="/home/ldnigogosova/datasets/dreambooth/dataset/dog6" --learnable_property="subject" --placeholder_token="sks" --initializer_token="dog" --resolution=512 --train_batch_size=4 --gradient_accumulation_steps=1 --max_train_steps=10000 --learning_rate=5.0e-04 --scale_lr --lr_scheduler="constant" --lr_warmup_steps=0 --output_dir="textual_inversion_dog" --checkpointing_steps=500 --validation_prompt="a sks dog in the jungle[EOS]a sks dog on the beach[EOS]a sks dog on top of a wooden floor[EOS]a sks dog wearing a rainbow scarf[EOS]a purple sks dog" --validation_steps=500 --seed="0" --report_to="wandb"