# Shared environment for Jean-Zay preprocessing SLURM jobs (sourced, not submitted).

# shellcheck disable=SC1090
source "${HOME}/.bash_profile" 2>/dev/null || true
# shellcheck disable=SC1090
source "${HOME}/.bash_aliases" 2>/dev/null || true

module purge
module load pytorch-gpu/py3/2.8.0

REPO_ROOT="${REPO_ROOT:-${WORK}/motif}"
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}"

PATHS="${PATHS:-jz}"
NUM_WORKERS="${NUM_WORKERS:-39}"
