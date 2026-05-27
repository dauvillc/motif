#!/bin/bash
#SBATCH --job-name=prep_pmw
#SBATCH --output=slurm/jz/logs/%x.out
#SBATCH --error=slurm/jz/logs/%x.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --time=12:00:00
#SBATCH --hint=nomultithread
#SBATCH --account=xyw@cpu

set -euo pipefail
REPO_ROOT="${REPO_ROOT:-${WORK}/motif}"
# shellcheck source=../_common.sh
source "${REPO_ROOT}/slurm/jz/_common.sh"

python preproc/tc_primed/prepare_pmw_concat.py \
  paths="${PATHS}" \
  +check_older="3h" \
  +num_workers=${NUM_WORKERS}
