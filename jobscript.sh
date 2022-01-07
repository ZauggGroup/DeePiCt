#!/bin/bash
#SBATCH --output={job.params.logdir}/%j-%x.out
#SBATCH --job-name={job.rule}
#SBATCH --parsable

#SBATCH --mem={job.params.memory}
#SBATCH --time={job.params.walltime}
#SBATCH --ntasks={job.params.nodes}
#SBATCH --cpus-per-task={job.params.cores}
{job.params.gres}

export QT_QPA_PLATFORM='offscreen'
export DISPLAY=0.0
export PYTHONUNBUFFERED=1
export gpu=$CUDA_VISIBLE_DEVICES

echo CUDA_VISIBLE_DEVICES_in_job_script=$gpu
echo CLUSTER NAME: $SLURM_CLUSTER_NAME
echo CLUSTER NODE: $SLURMD_NODENAME
echo CONDA ENV: $CONDA_DEFAULT_ENV

{exec_job}
