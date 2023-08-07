#!/bin/bash

#SBATCH --job-name=quantest

#SBATCH --partition=ais-gpu

#SBATCH --mail-type=ALL

#SBATCH --mail-user=d.osin@skoltech.ru

#SBATCH --output=output_19.txt

#SBATCH --time=6-00

#SBATCH --mem=200G

#SBATCH --nodes=1

#SBATCH -c 16

#SBATCH --gpus=2

srun singularity exec --bind /gpfs/gpfs0/d.osin/:/home --bind /gpfs/gpfs0/d.osin/data_main:/home/dev/data_main -f --nv quantnas.sif bash -c '
    cd /home/QuanToaster;
    nvidia-smi;
    (sleep 60; python batch_exp.py -v 1e-4 -d entropy_4_esa -g 0 -c entropy_4.yaml) &
    python batch_exp.py -v 1e-5 -d entropy_4_8_esa -g 0 -c entropy_4_8.yaml &
    python batch_exp.py -v 1e-6 -d entropy_4_8_esa -g 1 -c entropy_4_8.yaml &

    wait
'

