#!/bin/bash

export srcdir="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
config_file=$1
export PYTHONPATH=${srcdir}/src
echo PYTHONPATH=$PYTHONPATH


snakemake \
    --snakefile "${srcdir}/Snakefile" \
    --config config="${config_file}" \
    --forceall \
    --use-conda \
#    --printshellcmds \
#    --cores 1 --resources gpu=1 --dryrun

