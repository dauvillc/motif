#!/bin/bash
# Submit the minimal preprocessing pipeline on Jean-Zay (TC-PRIMED download → prepare → split → norm).
# Run from the repository root: bash slurm/jz/submit_pipeline.sh

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${WORK}/motif}"
cd "${REPO_ROOT}"

mkdir -p slurm/jz/logs

J1=$(sbatch --parsable slurm/jz/download/download_tc_primed.sh)
J2=$(sbatch --parsable --dependency=afterok:"${J1}" slurm/jz/preprocess/prepare_pmw_concat.sh)
J3=$(sbatch --parsable --dependency=afterok:"${J2}" slurm/jz/preprocess/prepare_infrared.sh)
J4=$(sbatch --parsable --dependency=afterok:"${J3}" slurm/jz/preprocess/train_val_test_split.sh)
J5=$(sbatch --parsable --dependency=afterok:"${J4}" slurm/jz/preprocess/compute_normalization_constants.sh)

echo "Submitted pipeline:"
echo "  download_tc_primed:              ${J1}"
echo "  prepare_pmw_concat (after ${J1}): ${J2}"
echo "  prepare_infrared (after ${J2}):   ${J3}"
echo "  train_val_test_split (after ${J3}): ${J4}"
echo "  compute_normalization_constants (after ${J4}): ${J5}"
echo "Logs: slurm/jz/logs/"
