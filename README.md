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
The script input_saliency.py computes both the parameter-saliency profile of an image which allows to find misbehaving filters in a neural network responsible for misclassification of a given image. In addition, the script computes the image-space saliency which highlights pixels which drive the high filter saliency values.

To compute the parameter saliency profile for a given image, the script accepts 
* either path to the raw image + image target label
```bash
python3 parameter_and_input_saliency.py --model resnet50 --image_path raw_images/great_white_shark_mispred_as_killer_whale.jpeg --image_target_label 2
```
* or reference_id -- the index of the given image in ImageNet validation set.
```bash
python3 parameter_and_input_saliency.py --reference_id 107 --k_salient 10
```

here --reference_id specifies the image id from ImageNet validation set

--k_salient specifies the number of top salient filters to use for the input-space visualization

The resulting plots (input space colormap and filter saliency plot) will be saved to /figures

To export the loaded model weights to a `.pth` checkpoint while running the script, pass `--export_model_pth`:

```bash
python3 parameter_and_input_saliency.py \
    --model resnet50 \
    --image_path raw_images/great_white_shark_mispred_as_killer_whale.jpeg \
    --image_target_label 2 \
    --export_model_pth checkpoints/resnet50_exported.pth
```

The exported file is model-agnostic and stores a `state_dict` under the `state_dict` key, plus minimal metadata about how the model was loaded.

Demo
-----
The demo raw image is in /raw_images. The results are in /figures.

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
    - Add a small YOLOX wrapper class that builds Tiny from YOLOX Exp and loads
      `yolox_tiny.pth` (`model` key or raw `state_dict`).
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
4. Add YOLOX custom-model wrapper module in this repository.
    - Build model via YOLOX Exp (Tiny config).
    - Load checkpoint from `externals/YOLOX/weights/yolox_tiny.pth`.
5. Update `scripts/run_yolox_tiny_custom_model.sh`.
    - Point weights path to YOLOX Tiny checkpoint.
    - Use `--task detection` and wrapper class path.
    - Ensure YOLOX import path is available via `PYTHONPATH` in Docker execution.

#### 4) Step-by-step implementation plan

Recommended sequence:

1. Implement `DetectionTaskAdapter` and task selection CLI.
2. Add YOLOX Tiny wrapper class and validate checkpoint loading only.
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
| 1 | Implement `DetectionTaskAdapter` and task selection CLI | TODO | - | - | |
| 2 | Add YOLOX Tiny wrapper class and validate checkpoint loading | TODO | - | - | |
| 3 | Wire end-to-end saliency run for one raw image | TODO | - | - | |
| 4 | Remove fixed-size assumptions in gradient visualization | TODO | - | - | |
| 5 | Add cache/statistics fallback behavior for detection-only usage | TODO | - | - | |
| 6 | Update and validate `scripts/run_yolox_tiny_custom_model.sh` in Docker | TODO | - | - | |
| 7 | Run regression check for classification workflow | TODO | - | - | |

Optional change log template:

```text
[YYYY-MM-DD] [ID] [STATUS] summary
example: [2026-03-16] [2] [DOING] implemented YOLOX Tiny wrapper, loading test in progress
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
