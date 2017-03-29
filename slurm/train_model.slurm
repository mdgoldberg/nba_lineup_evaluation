#!/bin/bash

#SBATCH -p general
#SBATCH -J train_model
#SBATCH -n 8
#SBATCH -N 1
#SBATCH -t 3-00:00
#SBATCH --mem 250000
#SBATCH -o logs/train_model.out
#SBATCH -e logs/train_model.err

source new-modules.sh
module load git python/2.7.13-fasrc01

source activate thesis

mkdir -p data/models

python src/models/train_model.py
