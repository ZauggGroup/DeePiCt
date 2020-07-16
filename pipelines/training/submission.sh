#! /bin/bash

#SBATCH -A mahamid
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --mem 30G
#SBATCH --time 0-10:00
#SBATCH -o training.slurm.%N.%j.out
#SBATCH -e training.slurm.%N.%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH -p gpu
#SBATCH --gres=gpu:4 -n1 -c4



echo "Activating virtual environment"
source activate /struct/mahamid/Irene/segmentation_ribo/.snakemake/conda/50db6a03
echo "done"
export src_dir="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
export PYTHONPATH=${src_dir%/*/*}/src
echo PYTHONPATH=$PYTHONPATH
export QT_QPA_PLATFORM='offscreen'

usage()

{
    echo "usage: [[ [-set][-set set ]
                  [-config] [-config config] | [-h]]"
}


while [ "$1" != "" ]; do
    case $1 in
        -set | --set )   shift
                                set=$1
                                ;;
        -config | --config )   shift
                                config=$1
                                ;;
        -h | --help )           usage
                                exit
                                ;;
        * )                     usage
                                exit 1
    esac
    shift
done

export set=$set
export config=$config

echo "Submitting job for set" $set
echo "Generating training partitions"
python3 $src_dir/generate_training_data.py -config $config -set $set

echo "Starting training script for set" $set
python3 $src_dir/training.py -config $config -set $set -gpu $CUDA_VISIBLE_DEVICES
