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

# Preproc parallelism comes from ProcessPoolExecutor; cap intra-process thread
# pools to 1 so 39 workers don't each spawn ~40 OpenMP/BLAS threads and trip
# Jean-Zay's per-user thread limit (libgomp: "Resource temporarily unavailable").
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

PATHS="${PATHS:-jz}"
NUM_WORKERS="${NUM_WORKERS:-20}"
