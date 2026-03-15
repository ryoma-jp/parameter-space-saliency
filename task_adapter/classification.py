from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from .base import TaskAdapter
from target.spec import TargetSpec
from target.resolver import TargetResolver


class ClassificationTaskAdapter(TaskAdapter):
    """Task adapter for image classification.

    Uses :class:`TargetResolver` to convert a :class:`TargetSpec` into a
    concrete label tensor, then evaluates ``criterion(outputs, resolved_targets)``.

    Args:
        criterion: Loss function to use.  Defaults to ``CrossEntropyLoss``.
    """

    def __init__(self, criterion: Optional[nn.Module] = None):
        self.criterion = criterion if criterion is not None else nn.CrossEntropyLoss()
        self._resolver = TargetResolver()

    # ------------------------------------------------------------------

    def build_objective(
        self,
        model: nn.Module,
        inputs: torch.Tensor,
        true_labels: torch.Tensor,
        target_spec: TargetSpec,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        outputs  = model(inputs)
        resolved = self._resolver.resolve(outputs, true_labels, target_spec)
        loss     = self.criterion(outputs, resolved)
        return loss, resolved

    def summarize_prediction(
        self,
        model: nn.Module,
        inputs: torch.Tensor,
        true_labels: torch.Tensor,
        label_map: Dict[int, str],
    ) -> None:
        with torch.no_grad():
            outputs   = model(inputs)
            _, predicted = outputs.max(1)
        true_idx = true_labels[0].item()
        pred_idx = predicted[0].item()
        print(
            f"\n"
            f"        Image target label:        {true_idx}\n"
            f"        Image target class name:   {label_map.get(true_idx, str(true_idx))}\n"
            f"        Image predicted label:     {pred_idx}\n"
            f"        Image predicted class name:{label_map.get(pred_idx, str(pred_idx))}\n"
        )
