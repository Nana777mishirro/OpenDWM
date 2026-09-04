# Selected-object observation availability prototype

## Scientific contract

This prototype represents three distinct states for one real persistent
nuScenes object:

| Mode | 3D-box raster in selected interval | persistent slot | availability | memory |
| --- | --- | --- | --- | --- |
| A `clean` | target retained | active | 1 | updated from current observation |
| B `temporary_unavailable` | target removed | active | 0 | propagated from the last observation and scene context |
| C `explicit_absent` | target removed | removed | 0 | reset to zero |

B and C intentionally have identical target-deleted rasters.  They remain
distinguishable because B has an active missing slot and C has no slot.  RGB
supervision is unchanged in all three modes.

## Read-only audit of the original path

- `dwm.datasets.nuscenes.MotionDataset.get_3dbox_image` queries every
  `sample_annotation`, resolves `instance_token -> instance -> category`, and
  projects/draws class-coloured box edges.  The original output contained no
  instance identity.
- `MotionDataset.__getitem__` keeps the nuScenes tables internally but emitted
  only the rendered `3dbox_images`.  Persistent `instance_token` is therefore
  available before rasterisation and can support a stable object slot.
- Standard nuScenes `sample_annotation.json` has translation, size, rotation,
  `prev`, and `next`, but no measured object velocity field.  This prototype
  consequently sets `use_velocity=false`; it does not leak a future-derived
  finite-difference velocity.
- `PreviewDataset` loads already-rendered RGB/box/map files.  Its normal
  `data.json` has no track annotations.  Preview packages are not a sound
  standalone source for this training experiment unless an external manifest
  explicitly rejoins them to raw nuScenes metadata.
- `CrossviewTemporalSD.get_conditions` concatenates box and HDMap images along
  their channel dimension.  A sample-level `3dbox_condition_mask` replaces the
  whole box raster with `uncondition_image_color`; it cannot express an
  object-level unknown state.  The mask is sampled from
  `training_config.3dbox_condition_ratio`.
- `ImageAdapter` returns multi-scale image residuals.  In
  `DiTCrossviewTemporalConditionModel.forward`, those residuals are added just
  before selected transformer blocks.  The object adapter uses the same local
  residual boundary through a separate zero-initialised projection.
- Text, HDMap, raster image adapter, camera/action embeddings, and reference
  latent construction are otherwise untouched.

## Exact object representation

The dataset produces one stable slot dimension for the selected
`instance_token`.  The token string itself is retained only in the manifest;
the model sees a stable slot index across time, avoiding memorisation of
arbitrary dataset IDs.

Per frame, the model input contains:

- `object_class_ids [T,1]`: a class-prefix ID embedded by the adapter;
- `object_box_states [T,1,10]`: centre `(x,y,z)` in first-frame ego
  coordinates divided by `(50,50,10)`, size `(w,l,h)` divided by `(10,10,5)`,
  and a first-frame-ego-relative unit quaternion `(qw,qx,qy,qz)`;
- `object_availability [T,1]`;
- `object_slot_mask [T,1]`;
- `object_source_frame_indices [T,1]`, used by runtime leakage assertions;
- a recurrent GRU hidden state inside the adapter;
- an observed token or a learned missing token.

At B frames, `object_box_states[t]` is an exact copy of the last observed
state at `missing_start-1`; current GT target box, current/future target state,
and velocity are not model inputs.  At C frames, box/class/source are zeroed or
set to `-1`, the slot mask is false, and recurrent memory is reset.

The adapter pools the current transformer scene features, causally updates the
slot memory, applies a spatial query/slot gate, and injects the result through a
zero-initialised linear residual.  A checkpoint loaded with `strict=false`
therefore reports only adapter keys as missing and initially produces the same
clean output.

## Data construction and leakage controls

`ObjectAvailabilityDataset` wraps `MotionDataset` before the existing
`DatasetAdapter` transforms.  It selects a single target that:

- is the same real `instance_token` across the complete context;
- has at least 3 observed frames before and 2 after the interval;
- supports an interval length in `{1,3,5,7}`;
- visibly changes at least one configured camera raster in every selected
  missing frame.

Only the target is excluded from raster rendering.  `images`, `hdmap_images`,
text, calibration, timestamps, and every other base item remain copied from the
same sample.  `loss_target_roi` may be derived from GT for loss weighting, but
`CrossviewTemporalSD.get_conditions` deliberately never forwards that key.

The wrapper and model both assert source-frame causality.  Temporal-VAE frame
resampling currently raises `NotImplementedError` instead of silently
misaligning source indices.

## Configuration and modes

The checked-in profile is
`configs/experimental/object_availability/pilot_profile.json`.  The config
generator requires an explicit `--mode` value:

```bash
export PYTHONPATH=src
PY=/home/nvidia/miniconda3/envs/opendwm/bin/python
$PY -m dwm.tools.make_object_availability_pilot_config \
  --base-config configs/ctsd/single_dataset/ctsd_35_tirda_bm_nusc_a_warmup.json \
  --profile configs/experimental/object_availability/pilot_profile.json \
  --output-config /tmp/ctsd_object_B.json \
  --mode temporary_unavailable \
  --validation-mode temporary_unavailable \
  --manifest /path/to/train_manifest.json \
  --validation-manifest /path/to/val_manifest.json \
  --pretrained-model /home/nvidia/Workplace/models/sd35-medium \
  --checkpoint /home/nvidia/Workplace/OpenDWM/ckpts/ctsd_35_tirda_bm_nwao_40k.pth \
  --output-path /path/to/output
```

Use `clean`, `temporary_unavailable`, or `explicit_absent` to build A/B/C;
`mixed` is available for one training run with the profile probabilities.
Training and validation manifests must be built from their own dataset config:

```bash
$PY -m dwm.tools.build_object_availability_manifest \
  --config /tmp/ctsd_object_B.json \
  --dataset-key training_dataset \
  --output /path/to/train_manifest.json \
  --count 1024

$PY -m dwm.tools.build_object_availability_manifest \
  --config /tmp/ctsd_object_B.json \
  --dataset-key validation_dataset \
  --output /path/to/val_manifest.json \
  --count 128
```

Regenerate the config after the manifests exist.  The minimal pilot command is
then:

```bash
torchrun --nproc_per_node=16 -m dwm.train \
  -c /tmp/ctsd_object_B.json \
  -o /path/to/output \
  --log-steps 10 \
  --preview-steps 1000000 \
  --checkpointing-steps 500
```

`16` matches the example base config's `2 x 8` device mesh; change both the
launcher count and the base config's distributed settings together for the
actual pilot hardware.  This command is provided only as the next pilot step
and was not executed here.

The profile freezes the backbone and trains only the new object adapter.  ROI
weighting is present but disabled (`target_roi_loss_weight=0.0`).  No LoRA,
partial unfreeze, CLIP, DINO, or identity loss is enabled.

## Smoke-test result

The reproducible CPU test is:

```bash
$PY -m dwm.tools.object_availability_smoke \
  --nuscenes-root /home/nvidia/Datasets/nuscenes
```

The checked report at `reports/object_availability_smoke_report.json` records:

- real nuScenes `instance_token=c08be936e0394352b11760ede982d6fd`;
- 3 observed frames, a 5-frame interval `[3,8)`, and at least 2 observed
  frames after;
- target raster changed in all five interval frames, with B/C raster hashes
  identical;
- identical hashes across A/B/C for RGB, HDMap probe, text, calibration,
  timestamps, and shared noise;
- integrated tiny-DiT forward shape `[1,13,1,4,8,8]` and backward loss;
- exact clean zero-init output difference `0.0`;
- full-size (`hidden_dim=1536`) adapter parameter count `1,786,496`;
- all 25 adapter parameter tensors receiving non-zero gradients after the
  zero output projection is opened by its first step;
- B/C interval encoding L2 difference `3.4040484`, B non-zero count `40`, C
  non-zero count `0`, and distinct SHA-256 hashes.

No full training, model download, dependency upgrade, or 17 GB checkpoint load
was performed.

## Known limitations

- The prototype models one selected slot, not a general multi-object tracker.
- Only raw nuScenes `MotionDataset` is supported.  Preview-only and other
  dataset formats need an explicit track metadata bridge.
- nuScenes provides no direct measured object velocity in the current table
  path, so velocity is disabled.
- The full checkpoint was not forwarded in this environment because the GPU
  driver was unavailable; the forward/backward test uses the real integrated
  DiT class with a tiny CPU configuration.
- Temporal-VAE resampling is explicitly unsupported in this first version.
- The first optimizer step of a strict zero-initialised residual updates only
  the output projection; upstream memory/token parameters receive non-zero
  gradients from the following step.
