#!/bin/bash

#SBATCH --job-name=quantest

#SBATCH --partition=ais-gpu

#SBATCH --mail-type=ALL

#SBATCH --mail-user=d.osin@skoltech.ru

#SBATCH --output=output_10.txt

#SBATCH --time=6-00

#SBATCH --mem=32G

#SBATCH --nodes=1

#SBATCH -c 16

#SBATCH --gpus=2

srun singularity exec --bind /home/d.osin/:/home --bind /gpfs/gpfs0/d.osin/data_main:/home/dev/data_main -f --nv quantnas.sif bash -c '
    cd /home/QuanToaster;
    nvidia-smi;
    (sleep 10; python batch_exp.py -v 0 -d entropy_8_debug_old -g 0 -c entropy_8_debug_old.yaml) &
    python batch_exp.py -v 1e-6 -d entropy_8_debug_old -g 0 -c entropy_8_debug_old.yaml &
    python batch_exp.py -v 1e-5 -d entropy_8_debug_old -g 1 -c entropy_8_debug_old.yaml &
    python batch_exp.py -v 1e-4 -d entropy_8_debug_old -g 1 -c entropy_8_debug_old.yaml &

    wait
'

