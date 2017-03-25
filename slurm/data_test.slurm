#!/bin/bash

#SBATCH -p general
#SBATCH -J data_test
#SBATCH -n 11
#SBATCH -N 1
#SBATCH -t 0-02:00
#SBATCH --mem-per-cpu 7000
#SBATCH -o logs/data_test.out
#SBATCH -e logs/data_test.err

source new-modules.sh
module load git python/2.7.13-fasrc01

source activate thesis

mkdir -p data/testing

python src/testing/data_test.py
