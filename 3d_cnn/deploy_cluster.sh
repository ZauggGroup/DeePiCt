#!/bin/bash

export srcdir="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
config_file=$1
export PYTHONPATH=${srcdir}/src
echo PYTHONPATH=$PYTHONPATH

source ~/.bashrc
conda activate snakemake

srun -t 3-00:00:00 -c 1 --mem 4G \
    snakemake \
    --snakefile "${srcdir}/snakefile" \
    --cluster "sbatch" \
    --config config="${config_file}" gpu=$CUDA_VISIBLE_DEVICES \
    --jobscript "${srcdir}/jobscript.sh" \
    --jobs 20 \
    --use-conda \
    --printshellcmds \
    --latency-wait 30 \
    --max-jobs-per-second 1 \
    --max-status-checks-per-second 0.1 
