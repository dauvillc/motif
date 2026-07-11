---
name: motif-jz
description: >-
  Drive the Jean-Zay (IDRIS) cluster for the motif project — syncmotif rsync, submitit SLURM
  submission (setup=jz_*), job monitoring and log/debug, quota checks, W&B offline sync, and
  checkpoint resume. Use for any motif task involving `ssh jz`, $WORK/$SCRATCH/$STORE, setup=jz_*,
  paths=jz, H100/V100/CPU SLURM jobs, or running motif on Jean-Zay.
allowed-tools: Bash, Read, AskUserQuestion
---

# motif — Jean-Zay cluster operations

General-purpose skill for driving Jean-Zay (IDRIS) from the local machine for the **motif** project:
syncing, submitting SLURM jobs via submitit, monitoring/debugging, quota checks, W&B offline sync,
and resuming runs. SSH alias `jz` is assumed configured in `~/.ssh/config`.

For the canonical per-experiment command recipes (train / predict / eval), see
[`commands.md`](../../../commands.md) and [`CLAUDE.md`](../../../CLAUDE.md) — this skill covers the
cluster mechanics, not the experiment matrix.

## When to use

- Syncing the local project to Jean-Zay before any cluster job.
- Submitting, monitoring, or cancelling SLURM jobs (training, predictions, eval, preprocessing).
- Managing W&B in offline mode and syncing runs from the login node.
- Picking the right `setup=jz_*` config for a given hardware target.
- Resuming a training run from a submitit checkpoint.
- Checking quotas, hour consumption, environment, or doing ad-hoc cluster work.

## Cluster identity

**JZ username:** `ute68qj` · **project group:** `xyw` · **accounts:** `xyw@h100`, `xyw@v100`,
`xyw@cpu`. These are shared across the user's Jean-Zay projects.

## Local CLI tools

These commands live in the local shell (`~/.bash_aliases`, `~/.local/bin`). They SSH into Jean-Zay
and, where noted, pipe results through Claude. Prefer them over raw SSH for the matching tasks.

### Sync

| Alias | What it does |
|---|---|
| `syncmotif` | `rsync -azvP --exclude-from='.rsyncignore' ~/inria/motif jz:work` — syncs the local project to `$WORK/motif/` on JZ. Excludes `.git/`, `.venv/`, `uv.lock`, `outputs/`, `logs/`, `submitit/`, caches (see [`.rsyncignore`](../../../.rsyncignore)). |

### Queue and monitoring (shell functions, `~/.bash_aliases`)

| Command | Usage | What it does |
|---|---|---|
| `jzq` | `jzq` | One-shot `squeue` for `ute68qj` with a standard format. |
| `jzwatch` | `jzwatch` | Live-refresh the queue every 30 s (Ctrl-C to exit). |
| `jzpipe <id> …` | `jzpipe 111 222 333` | Live-refresh only the listed job IDs every 30 s — useful for a multi-step pipeline. |
| `jzlog <jobid>` | `jzlog 12345678` | Finds the stdout log (same search order as below) and `tail -f`s it live. |
| `jzinfo <jobid>` | `jzinfo 12345678` | Runs `scontrol show job` — full SLURM metadata. |
| `jzeff <jobid>` | `jzeff 12345678` | Runs `seff` — CPU/GPU efficiency report (completed jobs only). |
| `jzcancel <jobid>` | `jzcancel 12345678` | Runs `scancel`. **Always ask the user for confirmation before calling this.** |

### AI-assisted analysis (shell functions + `~/.local/bin` scripts)

| Command | Usage | What it does |
|---|---|---|
| `jzask <jobid> "<q>"` | `jzask 111 "why is val_loss plateauing?"` | Fetches the last ~150 log lines and asks Claude a one-shot question. |
| `jzchat <jobid>` | `jzchat 12345678` | Opens an **interactive** Claude session pre-loaded with the log. |
| `jzstatus [-n N]` | `jzstatus` / `jzstatus --lines 120` | SLURM queue + last N log lines for every active job; Claude summarises progress, metrics, health. |
| `jzdebug <jobid>` | `jzdebug 12345678` | Pulls stdout, stderr, `sacct` for a failed job; Claude gives root cause + fix. |
| `jzreport <jobid> [--save]` | `jzreport 12345678 --save` | Structured markdown report (summary, config, results, efficiency, issues). |
| `jzcompare <id1> <id2> …` | `jzcompare 111 222 333` | Compares 2–5 runs side-by-side: config diffs, metrics, verdict. |

**Log search order** (used by `jzlog`, `jzask`, `jzchat`, `jzdebug`, `jzreport`, `jzstatus`):
1. `scontrol` `StdOut` field
2. `$WORK/motif/submitit/*/<jobid>_*_log.out`  ← **motif nests logs under a `<name>_<timestamp>/` run folder**
3. `$WORK/motif/slurm/jz/logs/slurm-<jobid>.out` (if a plain-sbatch path is ever used)

## Agent behavior rules

1. **Sync before every job.** Run `syncmotif` from the local project root before submitting any SLURM
   job, no exceptions.
2. **Verify the job launched.** After submission, immediately run `squeue -u $USER` on the login node
   and confirm the job ID appears. Report the job ID and its state to the user.
3. **Use local tools for monitoring and debugging.** For status use `jzstatus`; for a failure use
   `jzdebug <jobid>`; for post-run analysis use `jzreport <jobid>`. Fall back to raw SSH +
   `squeue`/`sacct` only when these are unavailable.
4. **Ask before cancelling.** Never call `scancel` / `jzcancel` without explicit user confirmation.
5. **Pick the right partition.** Downloads that need internet → `setup=jz_cpu_prepost` (`prepost`).
   Preprocessing / eval without internet → `setup=jz_cpu`. Training → an `jz_*h100` / `jz_*v100`
   config. See SLURM hardware configs below.
6. **`paths=jz` is implied by `setup=jz_*`.** Every `configs/setup/jz_*.yaml` does
   `override /paths: jz`, so **do not** add a separate `paths=jz` override — it is redundant and can
   conflict.
7. **Never hardcode paths — and always use a login shell for SSH.** Reference cluster paths as
   `$WORK/motif` or `$SCRATCH/...`, not absolute paths. `$WORK`, `$SCRATCH`, and `$STORE` are **not
   set in non-interactive SSH sessions** — wrap every payload in a login shell so IDRIS profile
   scripts are sourced:
   ```bash
   ssh jz "bash -l -c '<command>'"
   ```
   Do **not** use bare `ssh jz "$WORK/..."` — `$WORK` will be empty. `$HOME` is a separate linkhome
   dir — project files live under `$WORK`, not `$HOME`.
8. **Report errors clearly.** If an SSH command fails, show the exact error and suggest a fix before
   proceeding.

## Storage layout

| Variable | Use | Notes |
|---|---|---|
| `$WORK` | Code, checkpoints (via `paths=jz`… but see below), virtual env | Persistent, backed up, slower I/O. **`$HOME` is a separate linkhome dir — do not use it for project files.** |
| `$SCRATCH` | Raw + preprocessed data, checkpoints, W&B logs, predictions, validation figures | Fast NVMe — purged regularly. **motif's `paths=jz` points data, checkpoints, and W&B logs here** (see [`configs/paths/jz.yaml`](../../../configs/paths/jz.yaml)). |
| `$JOBSCRATCH` | Intra-job temp files, fast DataLoader cache during a run | Fastest NVMe — deleted when the job ends, not backed up. |
| `$STORE` | Long-term archival of final weights / processed datasets | Cold storage, not for training I/O. Only mounted on `prepost` nodes. |

> Note: motif's [`configs/paths/jz.yaml`](../../../configs/paths/jz.yaml) writes preprocessed data,
> `checkpoints`, `wandb_logs`, `predictions`, `validation`, and `results` under **`$SCRATCH`** — so
> DataLoaders, checkpoints, and W&B offline runs all live on `$SCRATCH`. Back up anything you need to
> keep to `$STORE` before the 30-day-ish purge.

### Resolved absolute paths

| Variable | Absolute path |
|---|---|
| `$WORK` | `/lustre/fswork/projects/rech/xyw/ute68qj` |
| `$SCRATCH` | `/lustre/fsn1/projects/rech/xyw/ute68qj` |
| `$STORE` | `/lustre/fsstor/projects/rech/xyw/ute68qj` |
| `$HOME` | `/linkhome/rech/genini01/ute68qj` |

Prefer `bash -l -c '...'` (login shell resolves these automatically) over substituting them directly.
Use the table only when you truly need a hardcoded path.

## Environment activation

motif does **not** use raw `module load` on JZ. The SLURM `slurm_setup` blocks source the user's
profile and call a shell function that loads modules + activates the env:

| Function | Use for |
|---|---|
| `h100` | H100 GPU work (training on `jz_*h100`) |
| `v100` | V100 GPU work **and** CPU / preprocessing / eval work (the CPU configs reuse `v100`) |

So for any ad-hoc login-node command that needs the Python env:
```bash
ssh jz "bash -l -c 'source ~/.bash_profile && h100 && <command>'"   # H100 env
ssh jz "bash -l -c 'source ~/.bash_profile && v100 && <command>'"   # V100 / CPU / preproc env
```

**Internet access on compute nodes:** only the **`prepost`** partition (`setup=jz_cpu_prepost`,
`jz_cpu_visu`) has outbound internet. Regular CPU and GPU compute nodes have none. The login node has
internet for pip installs, W&B auth, and ad-hoc transfers.

## Job submission workflow

Follow these steps in order every time a job is submitted.

### Step 1 — Sync
Run `syncmotif` from the local project root (`~/inria/motif`).

### Step 2 — Submit (submitit)
Submission is a plain `python scripts/*.py setup=jz_* ...` — submitit reads `setup` and submits the
SLURM job itself. `paths=jz` is pulled in by the setup config (rule 6). Use `h100` for H100, `v100`
for V100/CPU.

```bash
# Training — H100, 2 nodes × 4 GPUs (default). See commands.md for experiment presets.
ssh jz "bash -l -c 'cd \$WORK/motif && source ~/.bash_profile && h100 && \
  python scripts/train.py experiment=fm_PI model=motif_12b_d512 setup=jz_8xh100 \
  dataloader.batch_size=2 wandb.name=<name>'"

# Training — H100 dev smoke test (1 GPU, 2 h, dev QoS)
ssh jz "bash -l -c 'cd \$WORK/motif && source ~/.bash_profile && h100 && \
  python scripts/train.py experiment=fm_PI model=motif_12b_d512 setup=jz_h100_dev \
  dataloader.batch_size=2 wandb.name=debug'"

# Predictions — V100
ssh jz "bash -l -c 'cd \$WORK/motif && source ~/.bash_profile && v100 && \
  python scripts/make_predictions.py run_id=<id> inference_cfg=<cfg> split=test \
  setup=jz_16xv100_2h +dataloader.batch_size=2'"

# Evaluation — CPU (no internet)
ssh jz "bash -l -c 'cd \$WORK/motif && source ~/.bash_profile && v100 && \
  python scripts/eval.py models=<...> eval_class=<...> +eval_name=<name> setup=jz_cpu num_workers=39'"

# Preprocessing — CPU (no internet)
ssh jz "bash -l -c 'cd \$WORK/motif && source ~/.bash_profile && v100 && \
  python preproc/tc_primed/prepare_pmw_concat.py paths=jz setup=jz_cpu'"

# Data download — prepost (internet access)
ssh jz "bash -l -c 'cd \$WORK/motif && source ~/.bash_profile && v100 && \
  python preproc/<downloader>.py paths=jz setup=jz_cpu_prepost'"
```

> Preprocessing/eval scripts take `paths=jz` explicitly since some may not go through a `setup` that
> overrides paths — match the pattern in `commands.md`. Training/predict use `setup=jz_*` which
> already implies `paths=jz`.

### Step 3 — Verify launch
```bash
ssh jz "bash -l -c 'squeue -u \$USER --format=\"%.10i %.15j %.8T %.10M %.10l %R\"'"
```
Confirm the new job ID appears and report it. `scripts/train.py` prints `Submitted job <id>` on
success. If nothing appears within ~10 s, check for an immediate failure:
```bash
ssh jz "bash -l -c 'ls -lt \$WORK/motif/submitit/ | head -5'"
```

### Step 4 — Monitor / debug
Prefer the local `jz*` tools (`jzstatus`, `jzlog`, `jzdebug`, `jzreport`). Raw fallbacks below.

## Job monitoring — raw SSH fallbacks

### Queue snapshot
```bash
ssh jz "bash -l -c 'squeue -u \$USER --format=\"%.10i %.20j %.10T %.12M %.12l %.6D %R\" --sort=-T'"
```
Format as a table — highlight `RUNNING` (good), `PENDING` (show the reason), `FAILED`/`CANCELLED`
(alert the user).

### Recently completed jobs (last 3 days)
```bash
ssh jz "bash -l -c 'sacct -u \$USER \
  --format=JobID,JobName,State,Start,End,Elapsed,ExitCode \
  --starttime=\$(date -d \"3 days ago\" +%Y-%m-%d) -n'"
```
Flag non-zero `ExitCode` as a potential failure.

### Live log tail (when `jzlog` is unavailable)
```bash
# motif nests logs under submitit/<name>_<timestamp>/
ssh jz "bash -l -c 'tail -f \$WORK/motif/submitit/*/<jobid>_0_log.out'"
```
Log files: `$WORK/motif/submitit/<name>_<timestamp>/<jobid>_<task>_log.out` / `…_log.err`.
Always show both stdout and stderr when diagnosing a failure.

### Quota and hour consumption
```bash
ssh jz "bash -l -c 'idr_quota_user -m'"      # personal $WORK and $SCRATCH usage
ssh jz "bash -l -c 'idr_quota_user -s'"      # personal $STORE usage
ssh jz "bash -l -c 'idr_quota_project -w'"   # project-level $WORK quota
ssh jz "bash -l -c 'idracct'"                # CPU + GPU hour consumption for the project
ssh jz "bash -l -c 'idr_compuse'"            # % of allocation used
```
Warn if usage exceeds 80%. Check hours when jobs are slow to schedule.

## SLURM hardware configs

All configs live in [`configs/setup/`](../../../configs/setup/) and each does `override /paths: jz`.
Hardware is selected via `slurm_constraint` + `slurm_account`, not partition. Use
`dataloader.batch_size=2` for training. Env function: `h100` for H100, `v100` for V100/CPU.

| Config | Hardware | Nodes×GPU | Account | Walltime | Use |
|---|---|---|---|---|---|
| `jz_8xh100` | H100 80 GB | 2 × 4 | `xyw@h100` | 20 h | **Default training** |
| `jz_4xh100` | H100 80 GB | 1 × 4 | `xyw@h100` | 20 h | Training (single node) |
| `jz_16xh100` | H100 80 GB | 4 × 4 | `xyw@h100` | 20 h | Large training |
| `jz_32xh100` | H100 80 GB | 8 × 4 | `xyw@h100` | 20 h | Very large training |
| `jz_h100_dev` / `jz_4xh100_dev` | H100 80 GB | 1 GPU / 1×4 | `xyw@h100` | 2 h (`qos_gpu_h100-dev`) | **Smoke test / debug** |
| `jz_16xv100_2h` / `_3h` | V100 16 GB | 4 × 4 | `xyw@v100` | 2 h / 3 h | Predictions (short) |
| `jz_16xv100` / `jz_32xv100` | V100 16 GB | 4×4 / 8×4 | `xyw@v100` | 20 h | Training / predictions |
| `jz_*v100_32g` | V100 32 GB (`v100-32g`) | varies | `xyw@v100` | varies | When 32 GB VRAM is needed |
| `jz_*_dev` (v100) | V100 | 1 GPU | `xyw@v100` | 2 h (`qos_gpu-dev`) | Smoke test / debug |
| `jz_cpu` | CPU | 1 task, 40 CPU | `xyw@cpu` | 4 h | **Preprocessing / eval (no internet)** |
| `jz_cpu_60min` | CPU | 1 task, 40 CPU | `xyw@cpu` | 1 h | Short CPU jobs |
| `jz_cpu_prepost` / `jz_cpu_visu` | CPU (`prepost`) | 1 task, 40 CPU | `xyw@cpu` | 4 h | **Downloads / anything needing internet** |

> The `jz_*_dev` variants use a dev QoS with shorter queues — the right choice for smoke tests and
> quick debugging. Override individual SLURM params on the CLI, e.g.
> `setup.timeout_min=6000`.

Local debug without submitit (runs the job in-process on the current machine):
```bash
python scripts/train.py experiment=<name> ... +run_local=true +wandb.mode=disabled
```

## W&B offline mode

Jobs run with W&B **offline** by default ([`configs/wandb/default.yaml`](../../../configs/wandb/default.yaml));
offline run folders are written under `$SCRATCH` (`paths.wandb_logs`). Disable W&B entirely during
local debug with `+wandb.mode=disabled`.

Sync offline runs from the login node (has internet) — safe to run repeatedly; wandb's `.synced`
marker skips already-synced folders:
```bash
ssh jz "bash -l -c 'cd \$WORK/motif && source ~/.bash_profile && v100 && \
  wandb sync \$SCRATCH/wandb/offline-run-*/'"
```
(Confirm the exact offline-run location under `$SCRATCH` if it differs — `paths.wandb_logs` in
`configs/paths/jz.yaml` is the root; wandb writes `wandb/offline-run-*` beneath the configured dir.)

## Checkpoint and resume

`TrainJob` implements `submitit.helpers.Checkpointable` ([`scripts/train.py`](../../../scripts/train.py)).
On SIGUSR1 (~`slurm_signal_delay_s`=120 s before timeout), submitit calls `checkpoint()` and requeues
the job with `resume_run_id` set — **resume is automatic** across the walltime limit.

To manually resume a run (per [`CLAUDE.md`](../../../CLAUDE.md)), relaunch with `+resume_run_id`:
```bash
ssh jz "bash -l -c 'cd \$WORK/motif && source ~/.bash_profile && h100 && \
  python scripts/train.py experiment=<name> model=motif_12b_d512 setup=jz_8xh100 \
  dataloader.batch_size=2 wandb.name=<name> +resume_run_id=<run_id>'"
```
Optional `+resume_mode=fine_tune` (default `resume`); `reset_output_layers` is available for
transfer. Checkpoints live under `paths.checkpoints/<run_id>` (on `$SCRATCH`). See
[`src/motif/utils/checkpoints.py`](../../../src/motif/utils/checkpoints.py).

## Maintenance

When motif's cluster conventions change — sync alias, `configs/setup/jz_*` names, the `h100`/`v100`
env functions, `configs/paths/jz.yaml`, the submitit log layout in `scripts/train.py`, or
`.rsyncignore` — update this skill in the same change so the commands stay correct.
