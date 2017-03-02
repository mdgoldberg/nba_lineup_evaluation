#!/bin/bash

#SBATCH -p general
#SBATCH -J pbp_fetch
#SBATCH -n 16
#SBATCH -N 1
#SBATCH -t 1-00:00
#SBATCH --mem-per-cpu 8000
#SBATCH -o logs/pbp_fetch.out
#SBATCH -e logs/pbp_fetch.err

source new-modules.sh
module load git python/2.7.13-fasrc01

mkdir logs
luigid --background --logdir logs
PYTHONPATH='.' luigi --module src.data.pbp_fetch_data PBPRangeFetcher --start-year 2002 --end-year 2016
