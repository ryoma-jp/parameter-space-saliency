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
