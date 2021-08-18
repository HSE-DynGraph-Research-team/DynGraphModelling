#!/bin/bash
#SBATCH --job_name=graph_training
#SBATCH --error=/home/lfsherstyuk/stderr.txt
#SBATCH --output=/home/lfsherstyuk/stdout.txt
#SBATCH --partition=normal
#SBATCH --gpus=8
#SBATCH --array=1-320
#SBATCH --cpus-per-task 4
#SBATCH --gpus-per-task 1
#SBATCH --mem-per-cpu=4g

#SBATCH --time=4:00:00

module purge
module restore module_graph_env
export RUN_NAME=first_parallel_try
source deactivate
source activate graph_env
echo "working on node `hostname`"
cd /home/lfsherstyuk/DynGraph-modelling-1/


srun python scenarios.py $SLURM_ARRAY_TASK_ID