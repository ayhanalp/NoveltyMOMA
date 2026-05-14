#!/bin/bash
SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Experiment setup for time-objectives with entropy shaping
# Using short label to fit SLURM's 15-char job name limit
BASE_LABEL="time_obj"

# Desired sweep:
# - betas 0.0, 0.5, 1.0 for seeds 6-10 (inclusive)
# - betas 0.05, 0.1 for seeds 1-10 (inclusive)
BETAS_HIGH=(0.0 0.5 1.0)
SEEDS_HIGH_START=1
SEEDS_HIGH_END=10

BETAS_LOW=(0.05 0.1)
SEEDS_LOW_START=1
SEEDS_LOW_END=10

# Precompute expected job count for messaging
COUNT_HIGH=$(( (${SEEDS_HIGH_END} - ${SEEDS_HIGH_START} + 1) * ${#BETAS_HIGH[@]} ))
COUNT_LOW=$(( (${SEEDS_LOW_END} - ${SEEDS_LOW_START} + 1) * ${#BETAS_LOW[@]} ))
TOTAL_JOBS=$((COUNT_HIGH + COUNT_LOW))

# Submit jobs
JOB_NUM=1

for BETA in "${BETAS_LOW[@]}"; do
    for SEED in $(seq ${SEEDS_LOW_START} ${SEEDS_LOW_END}); do
        BETA_SHORT=$(printf "%.2f" $BETA | sed 's/\./p/')
        LABEL="${BASE_LABEL}_B${BETA_SHORT}_S${SEED}"
        echo "[$JOB_NUM/${TOTAL_JOBS}] Submitting: $LABEL (BETA=$BETA, SEED=$SEED)"
        bash "$SCRIPT_DIR/slurm_single_run.sh" "$LABEL" "$BETA" "$SEED"
        sleep 1
        ((JOB_NUM++))
    done
done

for BETA in "${BETAS_HIGH[@]}"; do
    for SEED in $(seq ${SEEDS_HIGH_START} ${SEEDS_HIGH_END}); do
        BETA_SHORT=$(printf "%.2f" $BETA | sed 's/\./p/')
        LABEL="${BASE_LABEL}_B${BETA_SHORT}_S${SEED}"
        echo "[$JOB_NUM/${TOTAL_JOBS}] Submitting: $LABEL (BETA=$BETA, SEED=$SEED)"
        bash "$SCRIPT_DIR/slurm_single_run.sh" "$LABEL" "$BETA" "$SEED"
        sleep 1
        ((JOB_NUM++))
    done
done

echo ""
echo "✓ Submitted all ${TOTAL_JOBS} jobs for beta sweep"
echo "Monitor with: squ"
