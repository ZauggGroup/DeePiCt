#!/usr/bin/env bash

#SBATCH -A mahamid
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --mem 20G
#SBATCH --time 0-00:20
#SBATCH -o mask_motl.slurm.%N.%j.out
#SBAtCH -e mask_motl.slurm.%N.%j.err
#SBATCH --mail-type=END,FAIL
#SBAtCH --mail-user=irene.de.teresa@embl.de

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

export config=$config
export set=$set
echo "Analyzing set" $set "and config file " $config

echo "Partitioning dataset"
python3 $src_dir/mask_motl.py -yaml_file $config -tomos_set $set
