from abc import ABC, abstractmethod
from typing import Dict, List

import torch.nn as nn
import torchvision.transforms as transforms


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
