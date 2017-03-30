#!/bin/bash

#SBATCH -p serial_requeue
#SBATCH -J train_model
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -t 1-00:00
#SBATCH --mem 80000
#SBATCH -o logs/train_model.out
#SBATCH -e logs/train_model.err

source new-modules.sh
module load git python/2.7.13-fasrc01

source activate thesis

mkdir -p data/models
mkdir -p models

python -m src.models.train_model
