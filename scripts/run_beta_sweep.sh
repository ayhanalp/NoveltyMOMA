#!/usr/bin/env bash
set -euo pipefail

# -----------------------------
# Resolve repo root robustly
# -----------------------------
# This gets the directory where *this script* lives,
# then assumes the repo root is the parent (adjust if needed)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
REPO_ROOT="$( cd "$SCRIPT_DIR/.." &> /dev/null && pwd )"

# -----------------------------
# (Optional) activate conda env
# -----------------------------
# Uncomment and edit if needed
# source "$(conda info --base)/etc/profile.d/conda.sh"
# conda activate env

# -----------------------------
# Experiment parameters
# -----------------------------
ALG_NAME="nsga2"
DOMAIN="rover"
DATA_DIR="$REPO_ROOT/data"

ALG_CONFIG="config/generic/generic_DMOConfig.yaml"
ENV_CONFIG="config/generic/generic_MORoverEnvConfig.yaml"

#SEED=2024
#SEED=$(date +%s)
#SEEDs=(1 2 3 4 5 7 8 9)
TRAJ_WRITE_FREQ=50

# We want:
# - betas 0.0, 0.5, 1.0 for seeds 6-10
# - betas 0.05, 0.1 for seeds 1-10
SEEDS_LOW=(1 2 3 4 5 6 7 8 9 10)
SEEDS_HIGH=(6 7 8 9 10)
BETAS_LOW=(0.05 0.1)
BETAS_HIGH=(0.0 0.5 1.0)

# -----------------------------
# Sanity checks
# -----------------------------
[[ -d "$DATA_DIR" ]] || { echo "Data dir not found: $DATA_DIR"; exit 1; }
[[ -f "$REPO_ROOT/$ALG_CONFIG" ]] || { echo "Alg config not found: $ALG_CONFIG"; exit 1; }
[[ -f "$REPO_ROOT/$ENV_CONFIG" ]] || { echo "Env config not found: $ENV_CONFIG"; exit 1; }

# -----------------------------
# Run experiment
# -----------------------------
## Run low betas (0.05, 0.1) for seeds 1-10
for SEED in "${SEEDS_LOW[@]}"; do
    for BETA in "${BETAS_LOW[@]}"; do
        echo "Running seed=$SEED beta=$BETA"
        PYTHONUNBUFFERED=1 python "$REPO_ROOT/main.py" \
            "$ALG_NAME" \
            "$DOMAIN" \
            "$DATA_DIR" \
            "$REPO_ROOT/$ALG_CONFIG" \
            "$REPO_ROOT/$ENV_CONFIG" \
            "$SEED" \
            "$TRAJ_WRITE_FREQ" \
            "$BETA"
    done
done

## Run high betas (0.0, 0.5, 1.0) for seeds 6-10
for SEED in "${SEEDS_HIGH[@]}"; do
    for BETA in "${BETAS_HIGH[@]}"; do
        echo "Running seed=$SEED beta=$BETA"
        PYTHONUNBUFFERED=1 python "$REPO_ROOT/main.py" \
            "$ALG_NAME" \
            "$DOMAIN" \
            "$DATA_DIR" \
            "$REPO_ROOT/$ALG_CONFIG" \
            "$REPO_ROOT/$ENV_CONFIG" \
            "$SEED" \
            "$TRAJ_WRITE_FREQ" \
            "$BETA"
    done
done