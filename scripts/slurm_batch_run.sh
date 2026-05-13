#!/bin/bash
SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Experiment setup for time-objectives with entropy shaping
# Using short label to fit SLURM's 15-char job name limit
BASE_LABEL="time_obj"
BETA_VALUES=(0.0 0.5 1.0 1.5 2.0)
NUM_SEEDS=5

# Run beta sweep with multiple seeds
JOB_NUM=1
for BETA in "${BETA_VALUES[@]}"; do
    for SEED in $(seq 1 $NUM_SEEDS); do
        # Create unique but concise label: "time_obj_B0_S1", "time_obj_B0p5_S1", etc.
        BETA_SHORT=$(printf "%.2f" $BETA | sed 's/\./p/')  # Convert 0.5 -> 0p5
        LABEL="${BASE_LABEL}_B${BETA_SHORT}_S${SEED}"
        
        echo "[$JOB_NUM/25] Submitting: $LABEL (BETA=$BETA, SEED=$SEED)"
        bash "$SCRIPT_DIR/slurm_single_run.sh" "$LABEL" "$BETA" "$SEED"
        # Small delay to avoid overwhelming the job scheduler
        sleep 1
        ((JOB_NUM++))
    done
done

echo ""
echo "✓ Submitted all 25 jobs for beta sweep: ${BETA_VALUES[*]}"
echo "Monitor with: squ"
