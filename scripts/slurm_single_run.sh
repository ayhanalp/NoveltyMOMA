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

# Get the repository root directory
REPO_ROOT=$(pwd)

ALG_NAME="nsga2"
DOMAIN="rover"
ALG_CONFIG="$REPO_ROOT/config/generic/generic_DMOConfig.yaml"
ENV_CONFIG="$REPO_ROOT/config/generic/generic_MORoverEnvConfig.yaml"
TRAJ_WRITE_FREQ=500
DATA_DIR="$REPO_ROOT/data"

echo "Running beta=$BETA seed=$SEED"

export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO_ROOT"
# Force single-threaded behaviour for numerical libraries to reduce nondeterminism
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
# Fix Python hash seed so hashed iteration orders are reproducible
export PYTHONHASHSEED=${SEED}

python "$REPO_ROOT/main.py" \
    "$ALG_NAME" \
    "$DOMAIN" \
    "$DATA_DIR" \
    "$ALG_CONFIG" \
    "$ENV_CONFIG" \
    "$SEED" \
    "$TRAJ_WRITE_FREQ" \
    "$BETA"
EOT
