#! /bin/bash

#SBATCH -A mahamid
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --mem 50G
#SBATCH --time 0-2:00
#SBATCH -o slurm.%N.%j.out
#SBAtCH -e slurm.%N.%j.err



export PYTHONPATH="/struct/cmueller/fung/bin/3d-unet/src"

source="/struct/path/"
destination="/struct/blablabla"

python3 match_spectrum.py -i $source -o $destination -t spectrum.tsv
