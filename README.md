### Where do Models go Wrong? Parameter-Space Saliency Maps for Explainability
This repository contains the implementation of parameter-saliency methods from our paper <a href = "https://arxiv.org/pdf/2108.01335.pdf">Where do Models go Wrong? Parameter-Space Saliency Maps for Explainability </a>. 

Abstract:
Conventional saliency maps highlight input features to which neural network predictions are highly sensitive. We take a different approach to saliency, in which we identify and analyze the network parameters, rather than inputs, which are responsible for erroneous decisions.  We find that samples which cause similar parameters to malfunction are semantically similar. We also show that pruning the most salient parameters for a wrongly classified sample often improves model behavior. Furthermore, fine-tuning a small number of the most salient parameters on a single sample results in error correction on other samples that are misclassified for similar reasons. Based on our parameter saliency method, we also introduce an input-space saliency technique that reveals how image features cause specific network components to malfunction.  Further, we rigorously validate the meaningfulness of our saliency maps on both the dataset and case-study levels.

Getting started
---------------
```
$ docker-compose build
$ docker-compose up -d
$ docker-compose exec pss bash
# cd /work
```

Basic Use
---------
The script `parameter_and_input_saliency.py` computes both:

- Parameter-space saliency (which filters/components are responsible for the objective)
- Input-space saliency (which pixels drive those parameter/component saliency values)

To compute the parameter saliency profile for a given image, the script accepts 

- either path to the raw image + image target label

```bash
python3 parameter_and_input_saliency.py --model resnet50 --image_path raw_images/great_white_shark_mispred_as_killer_whale.jpeg --image_target_label 2
```

- or `reference_id` (index of the image in ImageNet validation set)

```bash
python3 parameter_and_input_saliency.py --reference_id 107 --k_salient 10
```

`--reference_id` specifies the image id from ImageNet validation set.

`--k_salient` specifies the number of top salient filters to use for input-space visualization in matching mode.

### Output directory layout

All generated artifacts are grouped per image under:

```text
<output_root>/<image-id>/
```

Where:

- `<output_root>` is `--output_root` (default: `figures`)
- `<image-id>` is `--reference_id` value when `--reference_id` is used
- `<image-id>` is otherwise the basename of `--image_path` without extension

Main outputs:

```text
<output_root>/
    <image-id>/
        input_tensor.npy
        filter_saliency_<image-id>_<model-key>.png
        input_space_saliency/
            input_saliency_heatmap_<image-id>_<model-key>.png
        loss_component_saliency/
            raw_gradients/
            maps/
            images/
            <image-id>_<model-key>_metadata.json
        feature_manifest.json
        npy/
            feat_<logical_name>.npy
        detection_boxes_gt_only.png
        detection_boxes_pred_only.png
```

`loss_component_saliency/` is populated for detection task runs that produce component gradients.

The script also exports intermediate feature tensors in the output directory using the same layout as the YOLOX tooling:

```text
<output_root>/
    <image_key>/
        feature_manifest.json
        npy/
            feat_<logical_name>.npy
```

`feature_manifest.json` stores the logical layer name, actual module name, tensor shape, dtype, and relative `.npy` path for each exported feature tensor.

To export the loaded model weights to a `.pth` checkpoint while running the script, pass `--export_model_pth`:

```bash
python3 parameter_and_input_saliency.py \
    --model resnet50 \
    --image_path raw_images/great_white_shark_mispred_as_killer_whale.jpeg \
    --image_target_label 2 \
    --export_model_pth checkpoints/resnet50_exported.pth
```

The exported file is model-agnostic and stores a `state_dict` under the `state_dict` key, plus minimal metadata about how the model was loaded.

### `parameter_and_input_saliency.py` arguments

Below is a practical argument reference grouped by purpose.

#### Core run control

| Option | Default | Description |
|--------|---------|-------------|
| `--model` | `resnet50` | Torchvision model name (when `--model_source torchvision`). |
| `--model_source` | `torchvision` | Model source: `torchvision` or `custom_module`. |
| `--task` | `classification` | Task adapter: `classification` or `detection`. |
| `--output_root` | `figures` | Root output directory. |
| `--figure_folder_name` | `input_space_saliency` | Subdirectory name for input-space saliency images (inside `<output_root>/<image-id>/`). |

#### Image / dataset input

| Option | Default | Description |
|--------|---------|-------------|
| `--image_path` | `raw_images/great_white_shark_mispred_as_killer_whale.jpeg` | Raw image path for single-image run. |
| `--image_target_label` | `None` | Ground-truth label index for `--image_path` run. |
| `--reference_id` | `None` | Image index from ImageNet validation set. |
| `--data_to_use` | `ImageNet` | Dataset selector (currently ImageNet only). |
| `--imagenet_val_path` | placeholder | Path to ImageNet validation directory. |
| `--label_map_path` | `None` | YAML label map for class id to class name. |

#### Target selection

| Option | Default | Description |
|--------|---------|-------------|
| `--target_type` | `true_label` | Objective target: `true_label`, `predicted_top1`, `specified_class`. |
| `--target_class_id` | `None` | Class id used when `--target_type specified_class`. |

#### Saliency behavior

| Option | Default | Description |
|--------|---------|-------------|
| `--signed` | off | Use signed saliency instead of absolute-gradient-based behavior. |
| `--boost_factor` | `100.0` | Boost factor for top salient filters in matching mode. |
| `--k_salient` | `10` | Number of top filters to boost in matching mode. |
| `--compare_random` | off | Boost random filters instead of top salient filters. |
| `--noise_iters` | `1` | Number of noisy forward/backward iterations to average. |
| `--noise_percent` | `0` | Noise mixing ratio for SmoothGrad-like averaging. |
| `--input_saliency_method` | `auto` | `matching`, `direct_loss`, or `auto` (`classification -> matching`, `detection -> direct_loss`). |

#### Detection-specific options

| Option | Default | Description |
|--------|---------|-------------|
| `--det_annotations_json` | `None` | COCO annotations path used for GT matching and overlays. |
| `--det_conf_threshold` | `0.3` | Score threshold for predicted boxes in overlays/objective pipeline. |
| `--det_nms_iou_threshold` | `0.45` | NMS IoU threshold. |
| `--det_match_iou_threshold` | `0.5` | IoU threshold for TP/FP/FN matching against GT. |
| `--det_objective_mode` | `gt_all_instances` | Detection objective mode. |
| `--det_objective_provider` | `auto` | Optional model-specific objective provider. |
| `--det_provider_strict` | off | Raise error if requested provider is unavailable. |
| `--det_iou_weight` | `3.0` | IoU term weight in GT-instance objective. |
| `--det_score_weight` | `1.0` | Class-score term weight in GT-instance objective. |
| `--det_fp_loc_weight` | `0.0` | Weight of location-FP term (`0` disables). |
| `--det_fp_loc_iou_threshold` | `0.3` | IoU threshold for low-IoU FP localization term. |
| `--det_fp_loc_gate_sharpness` | `12.0` | Sigmoid sharpness for FP localization gating. |
| `--det_fp_loc_score_power` | `1.0` | Score exponent in FP localization term. |

#### Custom model loading options

| Option | Default | Description |
|--------|---------|-------------|
| `--model_class_path` | `None` | Fully-qualified class/callable path for `custom_module`. |
| `--model_weights_path` | `None` | Checkpoint path for custom model loading. |
| `--model_import_root` | `[]` | Extra import roots (repeatable). |
| `--model_kwargs_json` | `None` | JSON kwargs passed to model constructor/factory. |
| `--preprocess_cfg_json` | `None` | JSON preprocessing override (`resize`, `crop`, `normalize`, `scale`, etc.). |
| `--state_dict_target_path` | `None` | Dotted path for nested target to call `load_state_dict()`. |
| `--export_model_pth` | `None` | Export loaded model checkpoint while running. |

#### Deprecated (not implemented)

| Option | Status | Description |
|--------|--------|-------------|
| `--logit` | deprecated | Not implemented. |
| `--logit_difference` | deprecated | Not implemented. |

### Detection overlay color legend

In saliency heatmap overlays for detection runs, box colors are:

- `TP`: lime
- `FP` (aggregate): red
- `FP (class mismatch)`: red
- `FP (localization)`: orange
- `FN`: yellow

In exported box-only images:

- `detection_boxes_gt_only.png`: yellow boxes (GT)
- `detection_boxes_pred_only.png`: red boxes (predictions)

### Notes and troubleshooting

- `--reference_id` requires a valid `--imagenet_val_path` because the image is loaded from ImageNet validation data.
- For detection objectives that require per-image GT context (`gt_all_instances`, `gt_all_classes`), single-image `--image_path` runs are expected; `--reference_id` is currently not supported for those modes.
- If test-set saliency statistics are unavailable, the script automatically falls back from `std` mode to `naive` mode.
- For custom models, providing `--label_map_path` improves prediction summaries and overlay readability.

Demo
-----
The demo raw image is in `/raw_images`. The results are saved under `/figures/<image-id>/` by default.

Using Custom Models
-------------------
The codebase is structured around three independent abstraction layers — **Model**, **Task**, and **Target** — so that any user-defined PyTorch model can be plugged in without touching the saliency engine.

### Model layer

The **Model** layer (`model_adapter/`) is responsible for model construction, weight loading, preprocessing, and defining what counts as one *saliency unit* (by default, one output filter of each Conv2d layer).

Two adapters are provided:

| Adapter | When to use |
|---------|-------------|
| `TorchvisionAdapter` | Any `torchvision.models` model by name (default) |
| `CustomModuleAdapter` | Your own `nn.Module` class, loaded from a checkpoint |

To run with a custom model, pass `--model_source custom_module` together with the fully-qualified class path and an optional checkpoint path:

```bash
python3 parameter_and_input_saliency.py \
    --model_source custom_module \
    --model_class_path mypkg.models.MyNet \
    --model_weights_path checkpoints/mynet.pth \
    --image_path raw_images/sample.jpg \
    --image_target_label 0
```

`CustomModuleAdapter` also supports a more model-independent pattern where
`--model_class_path` points to any importable callable that returns an `nn.Module`.
This is useful when the model already has a factory function and you want to avoid
writing a model-specific wrapper.

Example: build YOLOX Tiny directly from the vendored YOLOX package via a generic
factory callable, without adding any model-specific wrapper:

```bash
python3 parameter_and_input_saliency.py \
    --task detection \
    --model_source custom_module \
    --model_import_root /work/externals/YOLOX \
    --model_class_path yolox.models.build.yolox_custom \
    --model_kwargs_json '{"exp_path":"/work/externals/YOLOX/exps/default/yolox_tiny.py","ckpt_path":"externals/YOLOX/weights/yolox_tiny.pth","device":"cpu"}' \
    --preprocess_cfg_json '{"resize":[416,416],"crop":null,"normalize":false,"scale":255.0}' \
    --image_path raw_images/sample.jpg \
    --target_type predicted_top1
```

#### Why `--model_kwargs_json` and `--preprocess_cfg_json` vary across models

`--model_kwargs_json` is forwarded as constructor/factory arguments to `--model_class_path`.
Different model types have different construction requirements:

| Model type | Constructor signature | `--model_kwargs_json` needed? | Example |
|-----------|-------------|------|---------|
| Simple `nn.Module` class | No required factory arguments | ✗ No | `torchvision.models.resnet50()` — works with defaults |
| Factory function with required config | Requires configuration parameters | ✓ Yes | Any factory function expecting `exp_path`, `config_file`, or similar required arguments |

Similarly, `--preprocess_cfg_json` encodes input preprocessing (resize/crop/normalize/scale).
Models trained on different datasets or with different input sizes may require custom preprocessing:
- Models matching ImageNet conventions → use defaults (256→224 resize/crop with ImageNet normalization)
- Models with custom input requirements → specify custom preprocessing via JSON

Supported `custom_module` CLI extensions:

| Option | Purpose |
|--------|---------|
| `--model_import_root` | prepend extra import roots before resolving `--model_class_path` |
| `--model_kwargs_json` | JSON object forwarded to the constructor/factory |
| `--preprocess_cfg_json` | JSON object configuring resize/crop/normalize/scale |
| `--state_dict_target_path` | dotted attribute path for nested `load_state_dict()` targets |

If your model expects different input statistics, pass them via `preprocess_cfg` when constructing `CustomModuleAdapter` directly in Python:

```python
from model_adapter import CustomModuleAdapter

adapter = CustomModuleAdapter(
    class_path='mypkg.models.MyNet',
    weights_path='checkpoints/mynet.pth',
    preprocess_cfg={
        'resize': 256, 'crop': 224,
        'mean': [0.5, 0.5, 0.5],
        'std':  [0.5, 0.5, 0.5],
    },
)
```

When you still need a wrapper:

1. The model cannot be built by a single importable callable.
2. The checkpoint format requires custom key remapping or partial loading.
3. The model needs bespoke forward-time input conversion beyond `resize`, `crop`, `normalize`, and `scale`.
4. Saliency units must be defined in a model-specific way that does not match the default Conv2d filter grouping.

Minimal wrapper guidelines:

1. Keep the wrapper small and focused on model construction and checkpoint compatibility.
2. Return a plain `nn.Module` from the wrapper constructor or factory.
3. Prefer generic CLI config (`--model_kwargs_json`, `--preprocess_cfg_json`, `--state_dict_target_path`) before adding wrapper-only logic.
4. If saliency units differ from Conv2d filters, prefer a custom `ModelAdapter` over embedding that policy in the wrapper.

To define a custom saliency unit granularity (e.g. attention heads, FPN levels), subclass `ModelAdapter` and override `iter_saliency_units()`:

```python
from model_adapter.base import ModelAdapter

class MyAdapter(ModelAdapter):
    def build_model(self): ...
    def get_preprocess(self): ...
    def get_inv_preprocess(self): ...

    def iter_saliency_units(self, model):
        # Return {layer_name: [flat_filter_indices]} for any unit definition
        ...
```

### Task layer

The **Task** layer (`task_adapter/`) encapsulates how a forward pass is executed and how the objective (loss) is computed.  The current implementation provides `ClassificationTaskAdapter`, which uses `CrossEntropyLoss` on the resolved target.

To support a new task (e.g. object detection), subclass `TaskAdapter`:

```python
from task_adapter.base import TaskAdapter

class DetectionTaskAdapter(TaskAdapter):
    def build_objective(self, model, inputs, true_labels, target_spec):
        # compute task-specific loss and return (loss, resolved_targets)
        ...

    def summarize_prediction(self, model, inputs, true_labels, label_map):
        ...
```

### Target layer

The **Target** layer (`target/`) specifies *what* the saliency gradient is computed with respect to, independently of the model and task.

Three target types are available via `--target_type`:

| `--target_type` | Description |
|-----------------|-------------|
| `true_label` (default) | Ground-truth class label |
| `predicted_top1` | Top-1 predicted class (useful for studying confident wrong predictions) |
| `specified_class` | A fixed class id supplied via `--target_class_id` |

Example — compute saliency with respect to a specific class:

```bash
python3 parameter_and_input_saliency.py \
    --model resnet50 \
    --image_path raw_images/great_white_shark_mispred_as_killer_whale.jpeg \
    --image_target_label 2 \
    --target_type specified_class \
    --target_class_id 148
```

To add a new target type, add a value to `TargetType` in `target/spec.py` and handle it in `TargetResolver.resolve()` in `target/resolver.py`.

### Label maps for custom models

For torchvision models, ImageNet labels are downloaded automatically.  For custom models, provide a YAML file mapping integer class indices to human-readable names:

```yaml
# label_map.yaml
0: cat
1: dog
2: bird
```

```bash
python3 parameter_and_input_saliency.py \
    --model_source custom_module \
    --model_class_path mypkg.models.MyNet \
    --label_map_path label_map.yaml \
    --image_path raw_images/sample.jpg \
    --image_target_label 0
```

### Object Detection Extension Roadmap (YOLOX Tiny)

This section describes the roadmap for supporting an object detection custom model
using `externals/YOLOX/weights/yolox_tiny.pth`.

#### 1) Gap analysis against the current implementation

Current code is classification-centric. The main gaps are:

1. Task pipeline is fixed to classification.
    - `parameter_and_input_saliency.py` always instantiates `ClassificationTaskAdapter`.
2. Target handling assumes class labels from logits `(B, C)`.
    - `target/spec.py` and `target/resolver.py` are designed for classification labels.
3. Input saliency post-processing assumes 224x224 shape.
    - `save_gradients()` reshapes gradients to `(224, 224)`, which is incompatible with
      YOLOX Tiny default input size `(416, 416)`.
4. Statistics/inference cache flow assumes ImageNet-style classification runs.
    - Detection-only runs can miss required cache entries unless explicit fallback behavior is added.
5. Existing helper script for custom models is still configured for ResNet50 classification.
    - `scripts/run_yolox_tiny_custom_model.sh` must be updated for YOLOX Tiny detection.

#### 2) Extension architecture (design policy)

Use the existing Model/Task/Target separation and extend only the task-specific path.

1. Model layer:
    - Keep `model_source=custom_module`.
        - Prefer importable model factories plus CLI-supplied `model_kwargs_json` /
            `preprocess_cfg_json` to avoid model-specific wrappers.
        - Add a wrapper only when factory-based loading and generic preprocessing are insufficient.
    - Keep saliency unit extraction default (Conv2d output filters) for the first milestone.

2. Task layer:
    - Add `DetectionTaskAdapter(TaskAdapter)`.
    - `build_objective()` converts detection outputs to a differentiable scalar objective
      (for example top objectness-class score, or class-constrained score when class id is specified).
    - `summarize_prediction()` reports top detections (class, score, box) for logging.

3. Target layer:
    - Phase 1: reuse existing `target_class_id` semantics where possible.
    - Phase 2 (optional): add detection-specific target type(s), e.g. `top_detection`,
      `specified_detection_class`, or ROI-conditioned targets.

4. Runtime and compatibility:
    - Add task selector CLI argument, e.g. `--task {classification,detection}`.
    - Keep default as classification to preserve backward compatibility.
    - Add robust fallback for saliency stats mode when dataset-level stats are unavailable.

#### 3) Implementation proposal

Proposed concrete changes:

1. Add `task_adapter/detection.py`.
    - Implement objective construction from YOLOX inference output.
2. Update `task_adapter/__init__.py`.
    - Export `DetectionTaskAdapter`.
3. Update `parameter_and_input_saliency.py`.
    - Add `--task` argument.
    - Instantiate adapter by task type.
    - Make `save_gradients()` shape-dynamic (no fixed 224x224).
    - Add safe fallback from std mode to naive mode when statistics are missing.
4. Configure YOLOX Tiny through a generic callable path.
    - Build model via `yolox.models.build.yolox_custom`.
    - Pass Exp path and checkpoint path through `--model_kwargs_json`.
5. Update `scripts/run_yolox_tiny_custom_model.sh`.
    - Point weights path to YOLOX Tiny checkpoint.
    - Use `--task detection` and a generic callable path.
    - Ensure YOLOX import path is available via `PYTHONPATH` in Docker execution.

#### 4) Step-by-step implementation plan

Recommended sequence:

1. Implement `DetectionTaskAdapter` and task selection CLI.
2. Add model-independent YOLOX Tiny loading and validate checkpoint loading only.
3. Wire end-to-end saliency run for one raw image (no ImageNet val dependency).
4. Remove fixed-size assumptions in gradient visualization.
5. Add cache/statistics fallback behavior for detection-only usage.
6. Update and validate `scripts/run_yolox_tiny_custom_model.sh` in Docker.
7. Run regression check to confirm ResNet/classification scripts still work.

#### 5) Progress tracking

Use the table below to record implementation status.

Status legend:

- `TODO`: not started
- `DOING`: in progress
- `DONE`: implemented and verified
- `BLOCKED`: waiting for dependency/fix

| ID | Task | Status | Owner | Last update | Notes |
|----|------|--------|-------|-------------|-------|
| 1 | Implement `DetectionTaskAdapter` and task selection CLI | DONE | Copilot | 2026-03-16 | Added `task_adapter/detection.py` and `--task` dispatch in main script. |
| 2 | Add model-independent YOLOX Tiny loading path and validate checkpoint loading | DONE | Copilot | 2026-03-17 | Replaced wrapper-based loading with generic callable-based loading via `yolox.models.build.yolox_custom`. |
| 3 | Wire end-to-end saliency run for one raw image | DONE | Copilot | 2026-03-16 | Executed `bash scripts/run_yolox_tiny_custom_model.sh` and generated outputs. |
| 4 | Remove fixed-size assumptions in gradient visualization | DONE | Copilot | 2026-03-16 | Replaced fixed `(224,224)` reshape with dynamic tensor shape handling. |
| 5 | Add cache/statistics fallback behavior for detection-only usage | DONE | Copilot | 2026-03-16 | Added fallback to `naive` mode when testset stats are unavailable. |
| 6 | Update and validate `scripts/run_yolox_tiny_custom_model.sh` in Docker | DONE | Copilot | 2026-03-16 | Updated to YOLOX Tiny detection config and validated successful run. |
| 7 | Run regression check for classification workflow | DONE | User | 2026-03-17 | User executed the regression test and confirmed no issues. |

Optional change log template:

```text
[YYYY-MM-DD] [ID] [STATUS] summary
example: [2026-03-16] [2] [DOING] implemented generic YOLOX Tiny loading, validation in progress

[2026-03-16] [1] [DONE] added DetectionTaskAdapter and task selection CLI
[2026-03-17] [2] [DONE] replaced YOLOX-specific wrapper with generic callable-based loading
[2026-03-16] [3] [DONE] completed one-image E2E detection saliency run
[2026-03-16] [4] [DONE] made gradient visualization shape-dynamic
[2026-03-16] [5] [DONE] added saliency stats fallback to naive mode
[2026-03-16] [6] [DONE] updated and validated YOLOX Tiny custom-model run script
[2026-03-17] [7] [DONE] user ran the classification regression test and confirmed no issues
```

Validation criteria:

1. The script runs with `externals/YOLOX/weights/yolox_tiny.pth` as custom model input.
2. Detection objective backpropagates and returns parameter saliency without runtime errors.
3. Input-space saliency figure is saved correctly for non-224 resolutions.
4. Existing classification workflow remains unchanged by default.

Project Organization
------------
    ├── README.md
    ├── LICENSE
    ├── requirements.txt
    ├── utils.py  <- helper functions
    ├── parameter_and_input_saliency.py  <- main script
    │
    ├── model_adapter/   <- Model layer: model construction, preprocessing, saliency-unit definition
    │   ├── base.py               (ModelAdapter ABC)
    │   ├── torchvision_adapter.py
    │   ├── custom_module_adapter.py
    │   └── factory.py            (build_model_adapter)
    ├── task_adapter/    <- Task layer: forward pass and objective function
    │   ├── base.py               (TaskAdapter ABC)
    │   └── classification.py     (ClassificationTaskAdapter)
    ├── target/          <- Target layer: saliency target specification and resolution
    │   ├── spec.py               (TargetSpec, TargetType)
    │   └── resolver.py           (TargetResolver)
    │
    ├── figures <- folder for resulting figures
    ├── helper_objects <- precomputed cache objects (inference results and saliency statistics)
    │   ├─ resnet50
    │   ├─ densenet121
    │   ├─ inception_v3
    │   └── vgg19
    ├── raw_images <- images to use for parameter space saliency computation
    └── parameter_saliency
        └── saliency_model_backprop.py  <- SaliencyModel class, parameter saliency engine
