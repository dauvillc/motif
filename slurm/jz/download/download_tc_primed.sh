#!/bin/bash
#SBATCH --job-name=dl_tc_primed
#SBATCH --output=slurm/jz/logs/%x.out
#SBATCH --error=slurm/jz/logs/%x.err
#SBATCH --partition=prepost
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --time=20:00:00
#SBATCH --hint=nomultithread
#SBATCH --account=xyw@cpu

set -euo pipefail
REPO_ROOT="${REPO_ROOT:-${WORK}/motif}"
# shellcheck source=../_common.sh
source "${REPO_ROOT}/slurm/jz/_common.sh"

python preproc/tc_primed/download_tc_primed.py \
  paths="${PATHS}" \
  tc_primed_download.workers=32 \
  ${TC_PRIMED_EXTRA:-}
