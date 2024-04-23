#!/bin/bash
#SBATCH --job-name=test_dreambooth             # Заместо aboba напишите ваше незвание эксперимента
#SBATCH --error=runs/test_dreambooth-%j.err    # Файл для вывода ошибок
#SBATCH --output=runs/test_dreambooth-%j.log   # Файл для вывода результатов
#SBATCH --gpus=1                     # Количество запрашиваемых гпу
#SBATCH --cpus-per-task=4            # Выполнение расчёта на 8 ядрах CPU
#SBATCH --time=12:00:00               # Максимальное время выполнения (чч:мм:сс), после его окончания програмаа просто сбрасывается
#SBATCH --constraint="[type_a|type_b|type_c]"   # Выбор, какие узлы использовать.
																		 # Узлы a, b и c - содержат v100, d без гпу, e - содержит a100, простым смертным недоступен (нужно писать Айбеку)

module load Python/Anaconda_v03.2023
module load CUDA/12.2

source deactivate
source activate python311

cd ~/inference/dreambooth

python3 inference_dreambooth.py -m ~/diffusers/examples/dreambooth/output_dreambooth_dog/checkpoint-600/unet -p ../dog_t2i_prompt.txt -s ../generated_images/dog/t2i/dreambooth