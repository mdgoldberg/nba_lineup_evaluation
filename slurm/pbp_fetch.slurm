#!/bin/bash

#SBATCH -p general
#SBATCH -J pbp_fetch
#SBATCH -n 16
#SBATCH -N 1
#SBATCH -t 1-00:00
#SBATCH --mem 8000
#SBATCH -o pbp_fetch.out
#SBATCH -e pbp_fetch.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mgoldberg@college.harvard.edu

source new-modules.sh
module load git python/2.7.13-fasrc01

PYTHONPATH='.'
mkdir logs
luigid --background --logdir logs
luigi --module src.data.pbp_fetch_data PBPRangeFetcher --start-year 2002 --end-year 2016
