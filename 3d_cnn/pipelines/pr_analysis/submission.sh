#! /bin/bash

#SBATCH -A mahamid
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --mem 20G
#SBATCH --time 0-2:00
#SBATCH -o clustering.slurm.%N.%j.out
#SBATCH -e clustering.slurm.%N.%j.err
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
python3 $src_dir/pr_analysis.py -yaml_file $config -tomos_set $set
