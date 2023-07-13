#!/bin/bash

#SBATCH --job-name=quantest

#SBATCH --partition=ais-gpu

#SBATCH --mail-type=ALL

#SBATCH --mail-user=d.osin@skoltech.ru

#SBATCH --output=output_3.txt

#SBATCH --time=6-00

#SBATCH --mem=64G

#SBATCH --nodes=1

#SBATCH -c 64

#SBATCH --gpus=2

srun singularity exec --bind /home/d.osin/:/home --bind /gpfs/gpfs0/d.osin/data_main:/home/dev/data_main -f --nv quantnas.sif bash -c '
    cd /home/QuanToaster;
    nvidia-smi;
    (sleep 30; python batch_exp.py -v 0 -d entropy_8 -g 0 -c entropy_8.yaml) &
    python batch_exp.py -v 0 -d entropy_32 -g 0 -c entropy_32.yaml &

    (sleep 40; python batch_exp.py -v 1e-6 -d entropy_8 -g 1 -c entropy_8.yaml) &
    python batch_exp.py -v 1e-6 -d entropy_32 -g 1 -c entropy_32.yaml &

    wait
'
