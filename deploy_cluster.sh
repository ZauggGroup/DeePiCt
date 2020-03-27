#!/bin/bash

srcdir=$(dirname ${BASH_SOURCE[0]})
config_file=$1

srun -t 3:00:00 -c 1 --mem 128M \
    snakemake \
    --snakefile "${srcdir}/snakefile.py" \
    --cluster "sbatch" \
    --config config="${config_file}" \
    --jobscript "${jobscript}" \
    --jobs 20 \
    --use-conda \
    --printshellcmds \
    --dryrun
