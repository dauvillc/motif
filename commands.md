# Training
## GPM
### Fm
python scripts/train.py experiment=fm_gpm model=motif_12b_d512 setup=jz_32xv100_32g dataloader.batch_size=2 wandb.name=fm_gpm_dt6

## PMW
### Det
python scripts/train.py experiment=det_pmw model=motif_12b_d512 setup=jz_32xv100_32g dataloader.batch_size=2 wandb.name=det_pmw
### Fm
python scripts/train.py experiment=fm_pmw model=motif_12b_d512 setup=jz_32xv100_32g dataloader.batch_size=2 wandb.name=fm_pmw


## PMW + IR
### Det
python scripts/train.py experiment=det_PI model=motif_12b_d512 setup=jz_16xh100 dataloader.batch_size=2 wandb.name=det_PI
### Fm
python scripts/train.py experiment=fm_PI model=motif_12b_d512 setup=jz_8xh100 dataloader.batch_size=2 wandb.name=fm_PI

# Make predictions
## Varying DT
python scripts/make_predictions.py run_id=wbcb3op6,p9aozl3n,0y7cjuuv,bi6i4jlj,c5aod4ae,5wwdx0gx inference_cfg=fm_gpm_dt1 split=val setup=jz_16xv100_2h +dataloader.batch_size=2  --multirun
## GPM DT6 on samples that contain at least 1 pmw and 1 infrared
python scripts/make_predictions.py run_id=0y7cjuuv,frlwvx27-3,4yspbbv7-4 inference_cfg=fm_gpm_PI_dt6 split=test setup=jz_16xv100_3h +dataloader.batch_size=2 --multirun
python scripts/make_predictions.py run_id=k2s7ctuk inference_cfg=det_gpm_PI_dt6 split=test setup=jz_16xv100_2h +dataloader.batch_size=2
## GPM DT6 on samples that contain at least 1 pmw and 1 infrared, sources in the past only
python scripts/make_predictions.py run_id=0y7cjuuv,frlwvx27-3,4yspbbv7-4 inference_cfg=fm_gpm_PI_dt6_past split=test setup=jz_16xv100_3h +dataloader.batch_size=2 --multirun
python scripts/make_predictions.py run_id=k2s7ctuk inference_cfg=det_gpm_PI_dt6_past split=test setup=jz_16xv100_2h +dataloader.batch_size=2


# Evaluation
## Varying training dt
python scripts/eval.py models="{1h: [wbcb3op6, gpm_dt1], 3h: [p9aozl3n, gpm_dt1], 6h: [0y7cjuuv, gpm_dt1], 9h: [bi6i4jlj, gpm_dt1], 12h: [c5aod4ae, gpm_dt1], 24h: [5wwdx0gx, gpm_dt1]}" eval_class="[quantitative]" eval_class.quantitative.xlabel="Training_time_window_length_$\\Delta_t$" +eval_name=fm_gpm_dt1_vary_dt setup=jz_cpu num_workers=39 split=val
## SSL Multi-source, keeping all vs keeping only one
python scripts/eval.py models="{All_sources: [frlwvx27-3, gpm_dt1_multi], Single_source: [frlwvx27-3, gpm_dt1_multi_keep_one]}" eval_class="[visual, quantitative, sources, spectrum]" +eval_name=fm_gpm_dt1_multi_source setup=jz_cpu num_workers=39 checks_strictness=targets_only
## Det vs FM, Microwave and Infrared
python scripts/eval.py models="{Deterministic: [k2s7ctuk, gpm_PI_dt6], Flow_matching: [4yspbbv7-4, gpm_PI_dt6]}" eval_class="[visual, quantitative, spectrum]" +eval_name=gpm_PI_dt6_fm_vs_det setup=jz_cpu num_workers=39 split=test

## Supervised microwave vs SSL microwave vs SSL infrared
python scripts/eval.py models="{Supervised__Microwave_only: [0y7cjuuv, gpm_PI_dt6], Self-supervised__Microwave_only: [frlwvx27-3, gpm_PI_dt6], Self-supervised__Microwave+Infrared: [4yspbbv7-4, gpm_PI_dt6]}" eval_class="[quantitative, spectrum]" +eval_name=gpm_PI_dt6 setup=jz_cpu num_workers=39 split=test checks_strictness=targets_only

## Deterministic vs Supervised microwave vs SSL microwave vs SSL infrared
python scripts/eval.py models="{Deterministic__Self-supervised__Microwave+Infrared: [k2s7ctuk, gpm_PI_dt6], Supervised__Microwave_only: [0y7cjuuv, gpm_PI_dt6], Self-supervised__Microwave_only: [frlwvx27-3, gpm_PI_dt6], Self-supervised__Microwave+Infrared: [4yspbbv7-4, gpm_PI_dt6]}" eval_class="[quantitative, spectrum]" +eval_name=gpm_PI_dt6_with_det setup=jz_cpu num_workers=39 split=test checks_strictness=targets_only

## GPM DT6 Footprints
python scripts/eval.py models="{Supervised_Microwave_only: [0y7cjuuv, gpm_PI_dt6], Self-supervised__Microwave_only: [frlwvx27-3, gpm_PI_dt6], Self-supervised__Microwave+Infrared: [4yspbbv7-4, gpm_PI_dt6]}" eval_class="[footprint_map, visual]" +eval_name=gpm_dt6_footprints setup=jz_cpu_visu num_workers=39 split=test checks_strictness=targets_only +eval_class.inline_visual.sample_index=17 eval_class.footprint_map.sample_index=17

python scripts/eval.py models="{Deterministic: [k2s7ctuk, gpm_PI_dt6], Flow_matching: [4yspbbv7-4, gpm_PI_dt6]}" eval_class="[visual, footprint_map]" +eval_name=gpm_PI_dt6_fm_vs_det setup=jz_cpu num_workers=39 split=test +launch_without_submitit=true +eval_class.inline_visual.sample_index=63 eval_class.footprint_map.sample_index=63

