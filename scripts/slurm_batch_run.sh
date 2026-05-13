#!/bin/bash
SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Experiment setup for time-objectives with entropy shaping
LABEL=A2_P12_time_objectives
BETA_VALUES=(0.0 0.5 1.0 1.5 2.0)
NUM_SEEDS=5

# Run beta sweep with multiple seeds
for BETA in "${BETA_VALUES[@]}"; do
    for SEED in $(seq 1 $NUM_SEEDS); do
        echo "Submitting: LABEL=$LABEL, BETA=$BETA, SEED=$SEED"
        bash "$SCRIPT_DIR/slurm_single_run.sh" "$LABEL" "$BETA" "$SEED"
        # Small delay to avoid overwhelming the job scheduler
        sleep 1
    done
done

echo "Submitted all jobs for beta sweep: ${BETA_VALUES[*]}"
echo "Total jobs: $((${#BETA_VALUES[@]} * NUM_SEEDS)) runs"
