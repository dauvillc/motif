# Training

Flow-matching experiments with a 6-hour time window (`w6h`). Names encode the data regime: `M` = microwave only (self-supervised), `I` = infrared only (self-supervised), `MI` = microwave + infrared (self-supervised), `sup` = supervised baseline (GMI as the only masked target). Legacy presets live in `configs/experiment/old/`.

All `fm_ssl_*` and `fm_sup_*` presets default to **MOTIFGen** (`model=motif_12b_d512`, `MultisourceGeneralBackbone` in `src/motif/models/motif/backbone.py`). Alternate backbones are available via `model=…` — see [IM architectural variants](#im-architectural-variants) below.

All commands below use H100 setups on JZ (`paths=jz`) with `dataloader.batch_size=4`. Adjust if needed.

## Self-supervised: M vs I vs MI

### Microwave only (`fm_ssl_M_w6h`)
```bash
python scripts/train.py experiment=fm_ssl_M_w6h model=motif_12b_d512 setup=jz_4xh100 paths=jz \
  dataloader.batch_size=2 wandb.name=fm_ssl_M_w6h
```

### Infrared only (`fm_ssl_I_w6h`)
```bash
python scripts/train.py experiment=fm_ssl_I_w6h model=motif_12b_d512 setup=jz_8xh100 paths=jz \
  dataloader.batch_size=2 wandb.name=fm_ssl_I_w6h
```

### Microwave + Infrared (`fm_ssl_IM_w6h`)
```bash
python scripts/train.py experiment=fm_ssl_IM_w6h model=motif_12b_d512 setup=jz_8xh100 paths=jz \
  dataloader.batch_size=2 wandb.name=fm_ssl_IM_w6h
```

## Supervised baseline: GMI target (`fm_sup_IM_w6h`)

Microwave + infrared as input; only GMI/GPM is masked and reconstructed.

```bash
python scripts/train.py experiment=fm_sup_IM_w6h model=motif_12b_d512 setup=jz_8xh100 paths=jz \
  dataloader.batch_size=2 wandb.name=fm_sup_IM_w6h
```

## IM architectural variants

Comparisons on `fm_ssl_IM_w6h` (or any w6h preset) swap in alternate backbones via `model=…`. All variants share the same 12-block / `d512` hyperparameters; only the backbone class differs.

| Variant | Model config | Backbone |
|---------|--------------|----------|
| **MOTIFGen** (default) | `motif_12b_d512` | `MultisourceGeneralBackbone` |
| **MOTIF** (anchor cross-attention) | `motif_12b_d512_anchor` | `MultisourceAnchorBackbone` |
| **Transformer** | `transformer_12b_d512` | `TransformerBackbone` |

### MOTIF (anchor cross-attention)

```bash
python scripts/train.py experiment=fm_ssl_IM_w6h model=motif_12b_d512_anchor setup=jz_8xh100 paths=jz \
  dataloader.batch_size=4 wandb.name=fm_ssl_IM_w6h_motif_anchor
```

### Transformer

```bash
python scripts/train.py experiment=fm_ssl_IM_w6h model=transformer_12b_d512 setup=jz_8xh100 paths=jz \
  dataloader.batch_size=4 wandb.name=fm_ssl_IM_w6h_transformer
```

# Make predictions

> Legacy examples below use old experiment runs and `configs/inference_cfg/` presets. Replace `run_id` and `inference_cfg` once new runs are trained with the `fm_ssl_*` / `fm_sup_*` experiments.

## Varying DT
```bash
python scripts/make_predictions.py run_id=wbcb3op6,p9aozl3n,0y7cjuuv,bi6i4jlj,c5aod4ae,5wwdx0gx inference_cfg=fm_gpm_dt1 split=val setup=jz_16xv100_2h +dataloader.batch_size=2  --multirun
```
## GPM DT6 on samples that contain at least 1 pmw and 1 infrared
```bash
python scripts/make_predictions.py run_id=0y7cjuuv,frlwvx27-3,4yspbbv7-4 inference_cfg=fm_gpm_PI_dt6 split=test setup=jz_16xv100_3h +dataloader.batch_size=2 --multirun
python scripts/make_predictions.py run_id=k2s7ctuk inference_cfg=det_gpm_PI_dt6 split=test setup=jz_16xv100_2h +dataloader.batch_size=2
```
## GPM DT6 on samples that contain at least 1 pmw and 1 infrared, sources in the past only
```bash
python scripts/make_predictions.py run_id=0y7cjuuv,frlwvx27-3,4yspbbv7-4 inference_cfg=fm_gpm_PI_dt6_past split=test setup=jz_16xv100_3h +dataloader.batch_size=2 --multirun
python scripts/make_predictions.py run_id=k2s7ctuk inference_cfg=det_gpm_PI_dt6_past split=test setup=jz_16xv100_2h +dataloader.batch_size=2
```

# Evaluation

> Legacy examples below; update `models` once predictions exist for the new training experiments.

## Varying training dt
```bash
python scripts/eval.py models="{1h: [wbcb3op6, gpm_dt1], 3h: [p9aozl3n, gpm_dt1], 6h: [0y7cjuuv, gpm_dt1], 9h: [bi6i4jlj, gpm_dt1], 12h: [c5aod4ae, gpm_dt1], 24h: [5wwdx0gx, gpm_dt1]}" eval_class="[quantitative]" eval_class.quantitative.xlabel="Training_time_window_length_$\\Delta_t$" +eval_name=fm_gpm_dt1_vary_dt setup=jz_cpu num_workers=39 split=val
```
## SSL Multi-source, keeping all vs keeping only one
```bash
python scripts/eval.py models="{All_sources: [frlwvx27-3, gpm_dt1_multi], Single_source: [frlwvx27-3, gpm_dt1_multi_keep_one]}" eval_class="[visual, quantitative, sources, spectrum]" +eval_name=fm_gpm_dt1_multi_source setup=jz_cpu num_workers=39 checks_strictness=targets_only
```
## Det vs FM, Microwave and Infrared
```bash
python scripts/eval.py models="{Deterministic: [k2s7ctuk, gpm_PI_dt6], Flow_matching: [4yspbbv7-4, gpm_PI_dt6]}" eval_class="[visual, quantitative, spectrum]" +eval_name=gpm_PI_dt6_fm_vs_det setup=jz_cpu num_workers=39 split=test
```

## Supervised microwave vs SSL microwave vs SSL infrared
```bash
python scripts/eval.py models="{Supervised__Microwave_only: [0y7cjuuv, gpm_PI_dt6], Self-supervised__Microwave_only: [frlwvx27-3, gpm_PI_dt6], Self-supervised__Microwave+Infrared: [4yspbbv7-4, gpm_PI_dt6]}" eval_class="[quantitative, spectrum]" +eval_name=gpm_PI_dt6 setup=jz_cpu num_workers=39 split=test checks_strictness=targets_only
```

## Deterministic vs Supervised microwave vs SSL microwave vs SSL infrared
```bash
python scripts/eval.py models="{Deterministic__Self-supervised__Microwave+Infrared: [k2s7ctuk, gpm_PI_dt6], Supervised__Microwave_only: [0y7cjuuv, gpm_PI_dt6], Self-supervised__Microwave_only: [frlwvx27-3, gpm_PI_dt6], Self-supervised__Microwave+Infrared: [4yspbbv7-4, gpm_PI_dt6]}" eval_class="[quantitative, spectrum]" +eval_name=gpm_PI_dt6_with_det setup=jz_cpu num_workers=39 split=test checks_strictness=targets_only
```

## GPM DT6 Footprints
```bash
python scripts/eval.py models="{Supervised_Microwave_only: [0y7cjuuv, gpm_PI_dt6], Self-supervised__Microwave_only: [frlwvx27-3, gpm_PI_dt6], Self-supervised__Microwave+Infrared: [4yspbbv7-4, gpm_PI_dt6]}" eval_class="[footprint_map, visual]" +eval_name=gpm_dt6_footprints setup=jz_cpu_visu num_workers=39 split=test checks_strictness=targets_only +eval_class.inline_visual.sample_index=17 eval_class.footprint_map.sample_index=17

python scripts/eval.py models="{Deterministic: [k2s7ctuk, gpm_PI_dt6], Flow_matching: [4yspbbv7-4, gpm_PI_dt6]}" eval_class="[visual, footprint_map]" +eval_name=gpm_PI_dt6_fm_vs_det setup=jz_cpu num_workers=39 split=test +run_local=true +eval_class.inline_visual.sample_index=63 eval_class.footprint_map.sample_index=63
```
