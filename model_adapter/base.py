from abc import ABC, abstractmethod
from typing import Dict, List

import torch.nn as nn
import torchvision.transforms as transforms


_YOLOX_DEFAULT_LAYER_MAPPING = {
    "backbone_s8": "backbone.backbone.dark3",
    "backbone_s16": "backbone.backbone.dark4",
    "backbone_s32": "backbone.backbone.dark5",
    "neck_s8": "backbone.C3_p3",
    "neck_s16": "backbone.C3_n3",
    "neck_s32": "backbone.C3_n4",
}


class ModelAdapter(ABC):
    """Abstract base class for model adapters.

    Encapsulates model-specific differences:
    - model construction and weight loading
    - preprocessing configuration
    - saliency unit (parameter group) definition

    Subclass responsibility
    -----------------------
    Implement ``build_model``, ``get_preprocess``, and ``get_inv_preprocess``.
    Override ``iter_saliency_units`` when the default (conv-filter) definition
    is not applicable (e.g. Transformers, detection heads).
    """

    @abstractmethod
    def build_model(self) -> nn.Module:
        """Build and return the model (not yet moved to device)."""
        ...

    @abstractmethod
    def get_preprocess(self) -> transforms.Compose:
        """Return the preprocessing transform for raw input images."""
        ...

    @abstractmethod
    def get_inv_preprocess(self) -> transforms.Compose:
        """Return the inverse preprocessing transform (for visualization)."""
        ...

    def get_label_map(self) -> Dict[int, str]:
        """Return a mapping from class index to human-readable name.

        Returns an empty dict when not applicable; label loading is handled
        by the application layer.
        """
        return {}

    def iter_saliency_units(self, model: nn.Module) -> Dict[str, List[int]]:
        """Return {layer_name: [flat_filter_index, ...]} mapping.

        Default implementation: one unit per output filter of every 4-D
        (Conv2d-style) weight tensor — identical to the original code.

        Override this method to support other unit definitions such as
        linear neurons, attention heads, or FPN levels.

        Args:
            model: The model instance returned by ``build_model``.
                   Call this *before* wrapping with DataParallel so that
                   parameter names are clean (no 'module.' prefix).

        Returns:
            OrderedDict mapping each layer name to a list of flat indices
            into the concatenated filter-saliency vector.
        """
        layer_to_filter_id: Dict[str, List[int]] = {}
        ind = 0
        for name, param in model.named_parameters():
            if len(param.size()) == 4:
                for j in range(param.size()[0]):
                    layer_to_filter_id.setdefault(name, []).append(ind + j)
                ind += param.size()[0]
        return layer_to_filter_id

    def get_feature_export_layers(self, model: nn.Module) -> Dict[str, str]:
        """Return {logical_name: module_name} mapping for feature export.

        The default policy exports activations for the same weight-owning modules
        that participate in filter-wise saliency, which keeps saved features aligned
        with the existing saliency unit definition.
        """
        layer_mapping: Dict[str, str] = {}
        module_dict = dict(model.named_modules())

        for name, param in model.named_parameters():
            if len(param.size()) != 4 or not name.endswith('.weight'):
                continue

            module_name = name.rsplit('.', 1)[0]
            if module_name not in module_dict:
                continue

            logical_name = module_name.replace('.', '_')
            suffix = 2
            while logical_name in layer_mapping:
                logical_name = f"{module_name.replace('.', '_')}_{suffix}"
                suffix += 1

            layer_mapping[logical_name] = module_name

        # If this is a YOLOX-like model, also export the same six logical
        # feature levels used by externals/YOLOX/tools/visualize_eval_results.py.
        if all(module_name in module_dict for module_name in _YOLOX_DEFAULT_LAYER_MAPPING.values()):
            for logical_name, module_name in _YOLOX_DEFAULT_LAYER_MAPPING.items():
                layer_mapping.setdefault(logical_name, module_name)

        return layer_mapping
