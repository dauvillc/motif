#!/bin/bash
#SBATCH --job-name=dl_sar
#SBATCH --output=slurm/jz/logs/%x.out
#SBATCH --error=slurm/jz/logs/%x.err
#SBATCH --partition=prepost
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --time=04:00:00
#SBATCH --hint=nomultithread
#SBATCH --account=xyw@cpu

set -euo pipefail
REPO_ROOT="${REPO_ROOT:-${WORK}/motif}"
# shellcheck source=../_common.sh
source "${REPO_ROOT}/slurm/jz/_common.sh"

python preproc/sar/download_sar_cyclobs.py \
  paths="${PATHS}" \
  sar_download.workers=8 \
  ${SAR_EXTRA:-}
