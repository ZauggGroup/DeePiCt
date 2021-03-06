#!/bin/bash

srcdir=$(dirname ${BASH_SOURCE[0]})
config_file=$1

snakemake \
    --snakefile "${srcdir}/snakefile.py" \
    --config config="${config_file}" \
    --forceall \
    --use-conda \
    --printshellcmds \
    --cores 1 --resources gpu=1
