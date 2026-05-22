# Jean-Zay preprocessing (SLURM)

Bash job scripts for downloading raw data and building the preprocessed dataset on [Jean-Zay](https://www.idris.fr/jean-zay/).

Paths default to `paths=jz` ([`configs/paths/jz.yaml`](../../configs/paths/jz.yaml)): raw data on `$WORK`, preprocessed output on `$SCRATCH`.

## Prerequisites

- Repository cloned at `$WORK/motif` (override with `REPO_ROOT=...` if needed). Job scripts source `_common.sh` via this absolute path because SLURM copies scripts to `/var/spool/slurmd/...` (relative paths would break).
- Submit jobs from the **repository root** so log paths resolve.
- `tc_primed_ifovs.yaml` at `paths.tc_primed_ifovs` before `prepare_pmw_concat` (not downloaded by these scripts).
- Python env: `module load pytorch-gpu/py3/2.8.0` (set in [`_common.sh`](_common.sh); no `uv`).

## Layout

| Directory | SLURM partition | Purpose |
|-----------|-----------------|---------|
| [`download/`](download/) | `prepost` (internet) | Raw dataset downloads |
| [`preprocess/`](preprocess/) | default CPU | Prepare, split, normalization |

Logs: `slurm/jz/logs/%x.out` and `%x.err`.

## Full minimal pipeline

Chains: TC-PRIMED download → PMW prepare → IR prepare → train/val/test split → normalization.

```bash
cd $WORK/motif
bash slurm/jz/submit_pipeline.sh
```

SAR and ERA5 downloads are **not** included; submit them separately when needed.

## Individual jobs

```bash
cd $WORK/motif

# Downloads (prepost)
sbatch slurm/jz/download/download_tc_primed.sh
sbatch slurm/jz/download/download_sar_cyclobs.sh
sbatch slurm/jz/download/download_era5.sh

# Preprocess (CPU)
sbatch slurm/jz/preprocess/prepare_pmw_concat.sh
sbatch slurm/jz/preprocess/prepare_infrared.sh
sbatch slurm/jz/preprocess/train_val_test_split.sh
sbatch slurm/jz/preprocess/compute_normalization_constants.sh
```

## Overrides

Environment variables are read by [`_common.sh`](_common.sh) and job scripts:

| Variable | Default | Description |
|----------|---------|-------------|
| `REPO_ROOT` | `$WORK/motif` | Repository directory |
| `PATHS` | `jz` | Hydra paths config (`paths=...`) |
| `NUM_WORKERS` | `39` | Parallel workers for prepare/norm (40 CPUs, no HT) |

Examples:

```bash
PATHS=jz_regular NUM_WORKERS=20 sbatch slurm/jz/preprocess/prepare_pmw_concat.sh

TC_PRIMED_EXTRA="tc_primed_download.year=28 tc_primed_download.basin=AL" \
  sbatch slurm/jz/download/download_tc_primed.sh
```

Optional Hydra extras per download script: `TC_PRIMED_EXTRA`, `SAR_EXTRA`, `ERA5_EXTRA`. For normalization: `NORM_EXTRA` (e.g. to drop `process_only` and normalize all sources).

## Resource defaults

- 1 node, 40 CPUs, `--hint=nomultithread`, account `xyw@cpu`
- Wall times are set per script; increase `#SBATCH --time` if a step fails with timeout
