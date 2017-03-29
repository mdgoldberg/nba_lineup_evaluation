#!/bin/bash

#SBATCH -p general
#SBATCH -J compare_model
#SBATCH -n 8
#SBATCH -N 1
#SBATCH -t 1-00:00
#SBATCH --mem 160000
#SBATCH -o logs/compare_model.out
#SBATCH -e logs/compare_model.err

source new-modules.sh
module load git python/2.7.13-fasrc01

source activate thesis

mkdir -p data/models
mkdir -p models

python -m src.models.compare_model
