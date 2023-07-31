#!/bin/bash

#SBATCH --job-name=quantest

#SBATCH --partition=ais-gpu

#SBATCH --mail-type=ALL

#SBATCH --mail-user=d.osin@skoltech.ru

#SBATCH --output=output_5.txt

#SBATCH --time=6-00

#SBATCH --mem=64G

#SBATCH --nodes=1

#SBATCH -c 64

#SBATCH --gpus=2

srun singularity exec --bind /home/d.osin/:/home --bind /gpfs/gpfs0/d.osin/data_main:/home/dev/data_main -f --nv quantnas.sif bash -c '
    cd /home/QuanToaster;
    nvidia-smi;
    (sleep 30; python batch_exp.py -v 0 -d entropy_4_8_adn -g 0 -c entropy_4_8_adn.yaml) &
    python batch_exp.py -v 0 -d entropy_4_adn -g 0 -c entropy_4_adn.yaml &

    (sleep 40; python batch_exp.py -v 1e-6 -d entropy_4_8_adn -g 1 -c entropy_4_8_adn.yaml) &
    python batch_exp.py -v 1e-6 -d entropy_4_adn -g 1 -c entropy_4_adn.yaml &

    wait
'

