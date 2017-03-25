#!/bin/bash

#SBATCH -p general
#SBATCH -J pbp_fetch
#SBATCH -n 32
#SBATCH -N 1
#SBATCH -t 0-06:00
#SBATCH --mem-per-cpu 4000
#SBATCH -o logs/pbp_fetch.out
#SBATCH -e logs/pbp_fetch.err

source new-modules.sh
module load git python/2.7.13-fasrc01

source activate thesis

mkdir -p data/raw

luigid --logdir logs --background
PYTHONPATH='.' luigi --module src.data.pbp_fetch_data PBPRangeFetcher \
    --start-year 2006 --end-year 2016 --workers 4
