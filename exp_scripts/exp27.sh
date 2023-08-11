#!/bin/bash

#SBATCH --job-name=quantest

#SBATCH --partition=ais-gpu

#SBATCH --mail-type=ALL

#SBATCH --mail-user=d.osin@skoltech.ru

#SBATCH --output=output_27.txt

#SBATCH --time=6-00

#SBATCH --mem=400G

#SBATCH --nodes=1

#SBATCH -c 32

#SBATCH --gpus=4

srun singularity exec --bind /gpfs/gpfs0/d.osin/:/home --bind /gpfs/gpfs0/d.osin/data_main:/home/dev/data_main -f --nv quantnas.sif bash -c '
    cd /home/QuanToaster;
    nvidia-smi;
    (sleep 60; python batch_exp.py -v 5e-5 -d entropy_4_8_esadn -g 0 -c entropy_4_8_adn.yaml) &
    python batch_exp.py -v 1e-3 -d entropy_4_8_esadn_noise -g 1 -c entropy_4_8_adn_noise.yaml &
    python batch_exp.py -v 1e-3 -d entropy_4_esadn -g 2 -c entropy_4_adn.yaml &
    python batch_exp.py -v 1e-3 -d entropy_8_esadn_noise -g 3 -c entropy_8_adn_noise.yaml &

    wait
'

