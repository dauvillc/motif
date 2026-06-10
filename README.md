# Spatiotemporal data interpolation via multi-source generative modeling in an application to tropical cyclones

AI agents: see [CLAUDE.md](CLAUDE.md) for project context and workflows.

This repository implements a framework for interpolating images of tropical cyclones using multiple sources as input.

The framework is built with [PyTorch](https://pytorch.org/) and [Lightning](https://lightning.ai/docs/pytorch/stable/). It relies on [Hydra](https://hydra.cc/docs/intro/) for managing configurations and [Weights and Biases](https://wandb.ai/site/) for logging experiments. It includes multiple blocks:
* A `MultiSourceDataset` and custom `collate_fn` function to quickly assemble batches with a flexible number of sources while limiting the required memory.
* A mutli-source backbone adapted from DiT adapted for geospatial data.
* Two `Lightning` modules that implements that receives the output of the `MultiSourceDataset` as input and performs the following tasks:
  * Embedding each source into two common latent spaces: values and coordinates;
  * Randomly mask one of the embedded sources in each sample;
  * Process the embedded sequences through the backbone;
  * Project the updated values sequences to their original spaces;
  * Compute the loss between the masked source's original values and the reconstructed values.
One of the modules is deterministic (trained with the MSE as loss function), while the other is generative using flow matching.
* Active research program (see [CLAUDE.md](CLAUDE.md), [commands.md](commands.md)):
  * Compare self-supervised **M** (microwave), **I** (infrared), and **IM** (both) regimes;
  * Supervised GMI-only baseline on the IM setup (`fm_sup_IM_w6h`);
  * On IM, planned architectural comparisons: **MOTIFGen** (`MultisourceGeneralBackbone`) vs original **MOTIF** (anchor cross-attention) and other baselines — standard experiments use `motif_12b_d512`.
* Training modes: self-supervised (random mask per sample) or supervised (fixed mask target; optional freeze/reset of layers).

# Repository organization
```
motif/
├── configs/               Hydra configuration files
│   ├── experiment/        Experiment definitions (model, data, and training settings)
│   ├── inference_cfg/     Inference configurations (dataset filtering, number of realizations)
│   ├── eval_class/        Evaluation class configurations
│   ├── model/             Model architecture configurations
│   ├── paths/             Environment-specific data paths
│   └── setup/             Compute setup configurations (local or SLURM)
├── preproc/               Preprocessing scripts
│   └── tc_primed/         TC-PRIMED download and preprocessing
├── scripts/               Entry-point scripts (train.py, make_predictions.py, eval.py)
└── src/motif/             Main package
    ├── data/              Dataset, collation, and source definitions
    ├── eval/              Evaluation metrics and visualizations
    ├── lightning_module/  Lightning modules (deterministic and flow matching)
    ├── models/            Model architecture (backbone, embedding, and output layers)
    └── utils/             Utility functions
```

# Running experiments
## Setting up the environment
The project requires Python 3.12 and uses [uv](https://docs.astral.sh/uv/) for dependency management. To install the dependencies and the package in a virtual environment, run
```
uv sync
```
from the repository root. The package can then be used by activating the virtual environment with `source .venv/bin/activate`, or by prefixing commands with `uv run`.

## Setting up the dataset
The first step is to set the paths for your own environment by creating your own configuration in ```configs/paths/```, using ```configs/paths/example.yaml``` as base. Raw dataset locations are defined there as derived paths from ```raw_datasets``` (```tc_primed```, ```sar_cyclobs```, ```era5_weatherbench```, etc.).
## Downloading the dataset
Raw data can be downloaded with Hydra, passing ```paths=<your_paths_config>``` (defaults to ```jz``` via ```configs/preproc.yaml```):

* TC-PRIMED: ```python preproc/tc_primed/download_tc_primed.py paths=<your_paths_config>```
* SAR (CyclObs): ```python preproc/sar/download_sar_cyclobs.py paths=<your_paths_config>```
* ERA5 WeatherBench2 Zarr: ```python preproc/era5/dl_era5_64x32.py paths=<your_paths_config>```

Optional overrides include ```+year=2015```, ```+basin=AL```, and ```+workers=32``` for TC-PRIMED (calendar year); ```sar_download.workers=4``` and ```era5_download.workers=16``` for the other downloads.
## Preprocessing
The dataset can be preprocessing using the following scripts:
* ```python preproc/tc_primed/prepare_pmw_concat.py paths=<your_paths_config> +num_workers=<n_workers>```
* ```python preproc/tc_primed/prepare_infrared.py paths=<your_paths_config> +num_workers=<n_workers>```
* ```python preproc/train_val_test_split.py paths=<your_paths_config>```
* ```python preproc/compute_normalization_constants.py paths=<your_paths_config> +num_workers=<n_workers>```

## Training models
Similarly to the paths configuration, the first step is to create a configuration in ```configs/setup/``` adapted to your computing environment, using ```configs/setup/exampe.yaml``` as base.

A training experiment can be run locally using
```python scripts/train.py experiment=<experiment_cfg> model=motif_12b_d512 setup=<your_setup_cfg> dataloader.batch_size=2 wandb.name=<name_of_the_experiment> +run_local=true```
On a SLURM cluster, the training can be directly submitted as a job using
```python scripts/train.py experiment=<experiment_cfg> model=motif_12b_d512 setup=<your_setup_cfg> dataloader.batch_size=2 wandb.name=<name_of_the_experiment>```
The following experiment configurations are available in `configs/experiment/` (6-hour training window, flow matching):

* ```fm_ssl_M_w6h```: Self-supervised, microwave only.
* ```fm_ssl_I_w6h```: Self-supervised, infrared only (GMI masked).
* ```fm_ssl_IM_w6h```: Self-supervised, microwave + infrared.
* ```fm_sup_IM_w6h```: Supervised baseline, GMI target on the MI source setup.

Legacy presets (`det_gpm`, `fm_pmw`, `fm_PI`, …) are in `configs/experiment/old/`. HPC command recipes: [commands.md](commands.md).

Every training run generates a run id that is printed out by the script. That run id is used as Weights and Biases id.

## Making predictions and evaluating

### Making predictions

Predictions are generated using `scripts/make_predictions.py`. The script requires a `run_id` (the Weights and Biases id of the trained model), an inference configuration, and a data split:
```
python scripts/make_predictions.py run_id=<run_id> inference_cfg=<inference_cfg> split=<val|test> setup=<your_setup_cfg> +dataloader.batch_size=2
```
To run predictions for multiple models in a single command, provide a comma-separated list of run ids and add the `--multirun` flag:
```
python scripts/make_predictions.py run_id=<run_id_1>,<run_id_2> inference_cfg=<inference_cfg> split=<val|test> setup=<your_setup_cfg> +dataloader.batch_size=2 --multirun
```
The following inference configurations are available in `configs/inference_cfg/`:
* `fm_gpm_dt1`: Flow matching model, GPM target, 1-hour time window.
* `det_gpm_dt1`: Deterministic model, GPM target, 1-hour time window.
* `fm_gpm_dt6`: Flow matching model, GPM target, 6-hour time window.
* `fm_gpm_PI_dt6`: Flow matching model, GPM target, 6-hour time window, requires at least one microwave and one infrared source.
* `det_gpm_PI_dt6`: Deterministic model, GPM target, 6-hour time window, requires at least one microwave and one infrared source.
* `fm_gpm_PI_dt6_past`: Same as `fm_gpm_PI_dt6`, restricting auxiliary sources to those observed before the target.
* `det_gpm_PI_dt6_past`: Same as `det_gpm_PI_dt6`, restricting auxiliary sources to those observed before the target.

### Evaluation

Evaluation is performed using `scripts/eval.py`. The `models` argument is a dictionary mapping display names to `[run_id, inference_cfg]` pairs, where `inference_cfg` must match the one used during prediction:
```
python scripts/eval.py models="{<model_name>: [<run_id>, <inference_cfg>]}" eval_class="[<eval_class_1>, <eval_class_2>]" +eval_name=<name> setup=<your_setup_cfg> num_workers=<n_workers> split=<val|test>
```
Multiple models can be compared by including additional entries in the `models` dictionary. The following evaluation classes are available in `configs/eval_class/`:
* `quantitative`: Computes quantitative metrics (RMSE, bias) aggregated across the dataset.
* `visual`: Generates visual comparisons of predictions and targets.
* `spectrum`: Compares the power spectra of predictions and targets.
* `sources`: Analyses the influence of the number and type of auxiliary sources on performance.