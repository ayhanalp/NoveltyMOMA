#!/bin/bash
# SLURM job template for running NoveltyMOMA on an HPC cluster
# Replace variables below or set them via environment when submitting.

# Job name
#SBATCH --job-name=noveltymoma
# Time (D-HH:MM:SS)
#SBATCH --time=1-00:00:00
# Partition/queue
#SBATCH --partition=standard
# Number of tasks (processes)
#SBATCH --ntasks=1
# CPUs per task
#SBATCH --cpus-per-task=4
# Memory
#SBATCH --mem=16G
# GPU (uncomment if needed)
##SBATCH --gres=gpu:1
# Output and error files
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

# Load modules or activate conda env as required by your cluster
# Example for conda:
# module load anaconda
# conda activate /path/to/conda/env

# Or use environment.yml to create an environment at project setup time:
# conda env create -f /path/to/NoveltyMOMA/environment.yml -n noveltymoma
# conda activate noveltymoma

# Run the experiment. Change the args as needed:
# Usage: python main.py <alg_name> <domain_name> <data_dirpath> <alg_config> <env_config> <seed> <label> <traj_write_freq>
python3 main.py nsga2 rover data/alg_runs/ algconfig.yaml envconfig.yaml 42 mylabel 100

# -------------------------------
# Example: resume an interrupted run
# -------------------------------
# If your previous run was interrupted, use the resume helper to continue from
# the latest checkpoint. Replace <run_dir> with the absolute path to the
# experiment folder created by the original run (e.g. /scratch/$USER/noveltymoma/data/mylabel_1).
#
# Activate the same environment as the original job, then call the resume script:
# module load anaconda
# conda activate /path/to/noveltymoma
# python3 /path/to/NoveltyMOMA/scripts/resume_nsga2.py /absolute/path/to/<run_dir>

