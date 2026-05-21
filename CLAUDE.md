# motif — agent context

Multi-source spatiotemporal interpolation for tropical cyclones (PyTorch 2.8, Lightning, Hydra). Models fuse microwave, infrared, radar, and other observations to reconstruct masked sources via flow-matching (primary) or deterministic (MSE) training.

## Active experiment program

Primary comparisons (configs in `configs/experiment/`, HPC recipes in [commands.md](commands.md)):

1. **Source modality (SSL)**: microwave-only **M** (`fm_ssl_M_w6h`), infrared-only **I** (`fm_ssl_I_w6h`), microwave+infrared **MI** (`fm_ssl_IM_w6h`) — all self-supervised, 6h training window.
2. **Supervised baseline**: **sup** (`fm_sup_IM_w6h`) — GMI-only target on the MI source setup.
3. **Architecture on MI** (planned): **MOTIFGen** (`MultisourceGeneralBackbone` in `src/motif/models/motif/backbone.py`); original **MOTIF** with anchor cross-attention and other baselines are **not implemented yet**. Standard `fm_ssl_*` / `fm_sup_*` commands use `model=motif_12b_d512`; alternate backbones will use a dedicated model config when added.

Legacy presets (`det_gpm`, `fm_pmw`, `fm_PI`, …) live under `configs/experiment/old/`.

## Environment

- Python **3.12**; dependencies via [uv](https://docs.astral.sh/uv/): `uv sync` from repo root.
- Run commands with `uv run …` or `source .venv/bin/activate`.
- Format/lint: **Ruff** ([pyproject.toml](pyproject.toml)) — line length 100, double quotes, format on save in VS Code/Cursor.
- Torch pinned to **2.8.0**; type-checking scope: `src/motif`.

## Repository layout

```
configs/          Hydra configs (experiment, model, paths, setup, inference_cfg, eval_class, …)
scripts/          Entry points: train.py, make_predictions.py, eval.py
src/motif/        Package
  data/           MultiSourceDataset, Source, collate_fn, resampling
  lightning_module/  base_module, deterministic, flow_matching
  models/         Backbone (DiT-style), embeddings, output heads
  eval/           Metrics and visualizations (inherit AbstractEvaluationMetric)
  utils/          Checkpoints, cfg helpers, callbacks
preproc/          Dataset download and preprocessing (tc_primed/, splits, normalization)
```

Machine-specific roots: copy [configs/paths/example.yaml](configs/paths/example.yaml) → `configs/paths/<your_env>.yaml`. Compute presets: [configs/setup/example.yaml](configs/setup/example.yaml).

## Core architecture

- **Batches**: dicts keyed by `(source_name, obs_index)` with `coords`, `landmask`, `values`, `availability` — see [multi_source_dataset.py](src/motif/data/multi_source_dataset.py) and [source.py](src/motif/data/source.py) (`Source` input/output masks, target variables).
- **Collation**: `multi_source_collate_fn` in [collate_fn.py](src/motif/data/collate_fn.py).
- **Lightning modules**: shared logic in [base_module.py](src/motif/lightning_module/base_module.py) (`preproc_input`, `compute_loss_mask`); [deterministic.py](src/motif/lightning_module/deterministic.py) (learned mask token); [flow_matching.py](src/motif/lightning_module/flow_matching.py) (noise schedule, optional CFG, optional deterministic teacher).
- **Models**: backbone and heads from `configs/model/*.yaml`, wired via `configs/lightning_module/*.yaml` — not hardcoded in Python.

## Modeling invariants

- Coordinates normalized (sin/cos; optional cross-source norm); **NaNs → 0** in values.
- **Availability masks** drive masking and loss filtering — reuse existing hooks; do not add ad-hoc masking in new modules.
- New sources/backbones must respect `Source` input/output masks; overlap- and land-aware loss masking already exists in `base_module`.
- New Python classes: `hydra.utils.instantiate` with `_target_`; datasets may use `_convert_: partial`.

## Data and preprocessing

Under `paths.preprocessed_dataset`, expect:

```
prepared/<source>/source_metadata.json
samples_metadata.json
split CSVs (train/val/test)
```

If missing: run scripts in `preproc/` (e.g. `preproc/tc_primed/prepare_pmw_concat.py`, `prepare_infrared.py`, `train_val_test_split.py`, `compute_normalization_constants.py`) with `paths=<your_paths_config>`.

Dataset knobs: `configs/dataset/default.yaml` (`dt_max`, `dt_max_norm`, `min_available_sources`, forecasting options).

## Hydra workflows

All scripts use `@hydra.main(config_path="../configs", config_name=…)`.

### Training — `scripts/train.py` (`configs/train.yaml`)

Required overrides (groups marked `???` in defaults): `paths`, `sources`, `model`, `experiment`, `setup`, plus `dataloader.batch_size`, `dataloader.num_workers`, `wandb.name`.

```bash
# Local debug (no SLURM)
python scripts/train.py experiment=fm_ssl_M_w6h model=motif_12b_d512 setup=local \
  paths=local dataloader.batch_size=2 wandb.name=debug +launch_without_submitit=true

# Cluster H100 (Submitit → SLURM) — SSL microwave+infrared
python scripts/train.py experiment=fm_ssl_IM_w6h model=motif_12b_d512 setup=jz_8xh100 \
  paths=jz dataloader.batch_size=4 wandb.name=fm_ssl_IM_w6h
```

- **Resume**: `+resume_run_id=<id>` and optional `+resume_mode=fine_tune` (default `resume`). Checkpoints: [checkpoints.py](src/motif/utils/checkpoints.py); optional `reset_output_layers`.
- **W&B**: offline by default ([configs/wandb/default.yaml](configs/wandb/default.yaml)); disable locally with `+wandb.mode=disabled`.
- **Artifacts**: checkpoints → `paths.checkpoints/<run_id>`; validation figures → `paths.validation`.

Experiment presets: `fm_ssl_M_w6h`, `fm_ssl_I_w6h`, `fm_ssl_IM_w6h`, `fm_sup_IM_w6h` (see [commands.md](commands.md)). H100 setups: `jz_h100` (1 GPU), `jz_4xh100`, `jz_8xh100`, `jz_16xh100` (`slurm_constraint: h100`) — use `dataloader.batch_size=4`. Legacy presets in `configs/experiment/old/`.

### Predictions — `scripts/make_predictions.py`

Reloads training config from checkpoint; merges CLI overrides; writes to `paths.predictions/<run_id>/<pred_name>/`.

```bash
python scripts/make_predictions.py run_id=<id> inference_cfg=fm_gpm_PI_dt6 split=test \
  setup=jz_16xv100_2h +dataloader.batch_size=2
```

Multirun: comma-separated `run_id=id1,id2` with `--multirun`. Inference configs in `configs/inference_cfg/` (e.g. `fm_gpm_dt6`, `det_gpm_PI_dt6_past`).

### Evaluation — `scripts/eval.py`

Reads saved predictions; `models` maps display names to `[run_id, inference_cfg]` (must match prediction run).

```bash
python scripts/eval.py \
  'models={FM: [4yspbbv7-4, gpm_PI_dt6]}' \
  eval_class="[quantitative, visual]" +eval_name=my_eval setup=jz_cpu num_workers=8 split=test
```

Eval classes in `configs/eval_class/` must subclass [AbstractEvaluationMetric](src/motif/eval/abstract_evaluation_metric.py) (`quantitative`, `visual`, `spectrum`, `sources`, `footprint_map`, …).

**Full HPC command recipes**: [commands.md](commands.md).

## Local development shortcuts

From `.vscode/launch.json` patterns:

- `+launch_without_submitit=true` — train without Submitit
- `+wandb.mode=disabled` — no W&B during debug
- `+dataset.train.limit_samples=N` / `+dataset.val.limit_samples=N` — smoke tests
- `dataloader.num_workers=0` and `dataloader.persistent_workers=false` — easier debugging

## Agent editing guidelines

- **Minimize scope**; match existing naming and Hydra group layout when adding configs.
- **Do not commit**: secrets, `outputs/`, `wandb/`, `lightning_logs/`, `*.ckpt`, personal `configs/paths/*` with real machine paths (use `example.yaml` as template).
- **Do not** reinvent masking, collation, or checkpoint loading — extend `base_module` / existing utils.
- Prefer Lightning callbacks already in use (`ModelCheckpoint`, `LearningRateMonitor`).
- Human onboarding and experiment tables: [README.md](README.md).
