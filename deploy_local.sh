#!/bin/bash

export srcdir="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
config_file=$1
echo CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
export PYTHONPATH=${srcdir}/src
echo PYTHONPATH=$PYTHONPATH

snakemake \
    --snakefile "${srcdir}/Snakefile" \
    --config config="${config_file}" gpu=$CUDA_VISIBLE_DEVICES \
    --forceall \
    --use-conda \
    --printshellcmds \
    --cores 1 --resources gpu=1

