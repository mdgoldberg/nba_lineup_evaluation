#!/bin/bash

#SBATCH -p general
#SBATCH -J select_model
#SBATCH -n 8
#SBATCH -N 1
#SBATCH -t 3-00:00
#SBATCH --mem 200000
#SBATCH -o logs/select_model.out
#SBATCH -e logs/select_model.err

source new-modules.sh
module load git python/2.7.13-fasrc01

source activate thesis

mkdir -p data/models

python -m src.models.select_model
