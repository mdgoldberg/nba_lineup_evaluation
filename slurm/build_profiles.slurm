#!/bin/bash

#SBATCH -p general
#SBATCH -J build_profiles
#SBATCH -n 5
#SBATCH -N 1
#SBATCH -t 1-00:00
#SBATCH --mem-per-cpu 30000
#SBATCH -o logs/build_profiles.out
#SBATCH -e logs/build_profiles.err

source new-modules.sh
module load git python/2.7.13-fasrc01

source activate thesis

mkdir -p data/processed

luigid --logdir logs --background

PYTHONPATH='.' luigi --module src.features.build_profiles RangeProfiles \
    --start-year 2007 --end-year 2016 --workers 5
