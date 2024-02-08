#!/bin/bash

srcdir=$(dirname ${BASH_SOURCE[0]})
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

snakemake \
    --snakefile "${srcdir}/snakefile.py" \
    --config config="${config_file}" \
    --forceall \
    --printshellcmds \
    --cores $(nvidia-smi --list-gpus | wc -l) \
    --resources gpu=$(nvidia-smi --list-gpus | wc -l)