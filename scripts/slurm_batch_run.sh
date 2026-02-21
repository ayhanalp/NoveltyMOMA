#!/bin/bash
SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

LABEL=A4_P8
BETA=0.0
SEED=1

bash "$SCRIPT_DIR/slurm_single_run.sh" "$LABEL" "$BETA" "$SEED"

