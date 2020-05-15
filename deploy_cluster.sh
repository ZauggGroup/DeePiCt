#!/bin/bash

srcdir=$(dirname ${BASH_SOURCE[0]})
config_file=$1

srun -t 6:00:00 -c 1 --mem 4G \
    snakemake \
    --snakefile "${srcdir}/snakefile.py" \
    --cluster "sbatch" \
    --config config="${config_file}" \
    --jobscript "${srcdir}/jobscript.sh" \
    --jobs 20 \
    --use-conda \
    --printshellcmds \
    --latency-wait 30
