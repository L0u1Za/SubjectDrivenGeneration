#!/bin/bash
#SBATCH --job-name=test_dreambooth             # Заместо aboba напишите ваше незвание эксперимента
#SBATCH --error=runs/test_dreambooth-%j.err    # Файл для вывода ошибок
#SBATCH --output=runs/test_dreambooth-%j.out   # Файл для вывода результатов
#SBATCH --gpus=1                     # Количество запрашиваемых гпу
#SBATCH --cpus-per-task=8            # Выполнение расчёта на 8 ядрах CPU
#SBATCH --time=3:00:00               # Максимальное время выполнения (чч:мм:сс), после его окончания програмаа просто сбрасывается
#SBATCH --constraint="[type_a|type_b|type_c]"   # Выбор, какие узлы использовать. 
																		 # Узлы a, b и c - содержат v100, d без гпу, e - содержит a100, простым смертным недоступен (нужно писать Айбеку)

module load Python/Anaconda_v03.2023
module load CUDA/12.2

source deactivate
source activate python311

cd diffusers/examples/dreambooth

export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export INSTANCE_DIR="dog"
export OUTPUT_DIR="output_lora"

accelerate launch train_dreambooth_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a photo of sks dog" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --checkpointing_steps=200 \
  --learning_rate=1e-4 \
  --report_to="wandb" \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=1000 \
  --validation_prompt="A photo of sks dog in a bucket" \
  --validation_epochs=10 \
  --seed="0"