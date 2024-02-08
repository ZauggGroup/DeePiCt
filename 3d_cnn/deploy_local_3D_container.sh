#!/bin/bash

export srcdir="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
config_file=$1
export CUDA_VISIBLE_DEVICES=
for i in $(seq 1 1 $(nvidia-smi --list-gpus | wc -l))
do
   if [ $i -eq 1 ]
   then
      export CUDA_VISIBLE_DEVICES=$(expr $i - 1)
   else
      export CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES,$(expr $i - 1)
   fi
done
echo $CUDA_VISIBLE_DEVICES
export PYTHONPATH=${srcdir}/src
echo PYTHONPATH=$PYTHONPATH

snakemake \
    --snakefile "${srcdir}/snakefile" \
    --config config="${config_file}" gpu=$CUDA_VISIBLE_DEVICES \
    --printshellcmds \
    --cores $(nvidia-smi --list-gpus | wc -l) \
    --resources gpu=$(nvidia-smi --list-gpus | wc -l)
