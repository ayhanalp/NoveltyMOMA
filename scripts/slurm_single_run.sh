#!/bin/bash
LABEL=$1
BETA=$2
SEED=$3

sbatch \
  --job-name=${LABEL}_${BETA}_${SEED} \
  --partition=share \
  --constraint=skylake \
  --output=./logs/${LABEL}_${BETA}_${SEED}.out \
  --error=./logs/${LABEL}_${BETA}_${SEED}.err \
  --export=ALL,BETA=${BETA},SEED=${SEED} \
  -c 1 \
  --mem=12G \
  --time=36:00:00 \
<<'EOT'
#!/bin/bash

# Troubleshooting data
hostname
echo $SLURM_JOBID
showjob $SLURM_JOBID

set -euo pipefail

cd /nfs/stak/users/santjami/hpc-share/repos/NoveltyMOMA/

module load anaconda
#source $(conda info --base)/etc/profile.d/conda.sh
#conda activate env
eval "$(conda shell.bash hook)"
conda activate env

ALG_NAME="nsga2"
DOMAIN="rover"
ALG_CONFIG="config/generic/generic_DMOConfig.yaml"
ENV_CONFIG="config/generic/generic_MORoverEnvConfig.yaml"
TRAJ_WRITE_FREQ=500

RUN_DIR="data/b${BETA}_s${SEED}"
mkdir -p "$RUN_DIR"

echo "Running beta=$BETA seed=$SEED"

export PYTHONUNBUFFERED=1

python main.py \
    "$ALG_NAME" \
    "$DOMAIN" \
    "$RUN_DIR" \
    "$ALG_CONFIG" \
    "$ENV_CONFIG" \
    "$SEED" \
    "$TRAJ_WRITE_FREQ" \
    "$BETA"
EOT
