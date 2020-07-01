#!/usr/bin/env bash

#SBATCH -A mahamid
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --mem 40G
#SBATCH --time 0-6:00
#SBATCH -o predict.slurm.%N.%j.out
#SBAtCH -e predict.slurm.%N.%j.err
#SBATCH --mail-type=END,FAIL
#SBAtCH --mail-user=irene.de.teresa@embl.de
#SBATCH -p gpu

echo "Activating virtual environment"
source activate 3d-cnn
echo "done"
export src_dir="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
export PYTHONPATH=${src_dir%/*/*}/src

echo PYTHONPATH=$PYTHONPATH
export QT_QPA_PLATFORM='offscreen'
usage()

{
    echo "usage: [[ [-config config_file] [-set set_number] ] | [-h]]"
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

export yaml_file=$config
export set=$set
echo "Analyzing set" $set "and config file " $config

echo "Partitioning dataset"
python3 $src_dir/partition.py -yaml_file $yaml_file -tomos_set $set

echo "Segmenting partition" $set
python3 $src_dir/segment.py -yaml_file $yaml_file -tomos_set $set --gpu $CUDA_VISIBLE_DEVICES

echo "Reconstructing segmentation"
python $src_dir/assemble.py -yaml_file $yaml_file -tomos_set $set

echo "Thresholding and clustering" $set
python3 $src_dir/cluster_motl.py -config $yaml_file -set $set