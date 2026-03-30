# Spatiotemporal data interpolation via multi-source generative modeling in an application to tropical cyclones
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
* Options to perform different experiments, including:
  * Training in a self-supervised setting: randomly mask and reconstruct a source for each sample;
  * Training or fine-tuning in a supervised manner by always masking the same source. The embedding layers, backbone and output layers can be frozen or reset.

# Running experiments
## Setting up the dataset
The first step is to set the paths for your own environment by creating your own configuration in ```configs/paths/```, using ```configs/paths/example.yaml``` as base.
## Downloading the dataset
TC-PRIMED can then be downloaded using ```python preproc/tc_primed/download_tcprimed.py --dest <raw_datasets_dir>/tc_primed/ --workers <n_workers>``` where ```<raw_datasets_dir>``` is the entry set in the paths YAML config.
## Preprocessing
The dataset can be preprocessing using the following scripts:
* ```python preproc/tc_primed/prepare_pmw_concat.py paths=<your_paths_config> +num_workers=<n_workers>```
* ```python preproc/tc_primed/prepare_infrared.py paths=<your_paths_config> +num_workers=<n_workers>```
* ```python preproc/train_val_test_split.py paths=<your_paths_config>```
* ```python preproc/compute_normalization_constants.py paths=<your_paths_config> +num_workers=<n_workers>```

## Training models
Similarly to the paths configuration, the first step is to create a configuration in ```configs/setup/``` adapted to your computing environment, using ```configs/setup/exampe.yaml``` as base.

A training experiment can be run locally using
```python scripts/train.py experiment=<experiment_cfg> model=motif_12b_d512 setup=<your_setup_cfg> dataloader.batch_size=2 wandb.name=<name_of_the_experiment> +launch_without_submitit=true```
On a SLURM cluster, the training can be directly submitted as a job using
```python scripts/train.py experiment=<experiment_cfg> model=motif_12b_d512 setup=<your_setup_cfg> dataloader.batch_size=2 wandb.name=<name_of_the_experiment>```
The following experiment configurations are available:
* ```det_gpm```: Deterministic model, supervised training using only GMI as target.
* ```det_pmw```: Deterministic model, self-supervised training using all sensors as targets.
* ```det_PI```: Deterministic model, self-supervised, using both microwave and infrared  data as input.
* ```fm_gpm```: Flow matching, supervised.
* ```fm_pmw```: Flow matching, self-supervised, microwave only.
* ```fm_PI```: Flow matching, self-supervised, microwave and infrared.

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