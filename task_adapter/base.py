from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from target.spec import TargetSpec


class TaskAdapter(ABC):
    """Abstract base class for task adapters.

    Encapsulates task-specific logic that sits between the model and the
    saliency engine:
    - running the forward pass
    - building the objective (loss) given a TargetSpec
    - summarising predictions for logging

    Adding a new task (e.g. detection, segmentation) means subclassing
    this without touching SaliencyModel or ModelAdapter.
    """

    @abstractmethod
    def build_objective(
        self,
        model: nn.Module,
        inputs: torch.Tensor,
        true_labels: torch.Tensor,
        target_spec: TargetSpec,
        objective_context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute the scalar loss and the resolved target labels.

        Args:
            model:        The model (may be DataParallel-wrapped).
            inputs:       Preprocessed input batch (B, C, H, W).
            true_labels:  Ground-truth labels for the batch (B,).
            target_spec:  Specification of which target to use.
            objective_context: Optional task/model-specific context.

        Returns:
            (loss, resolved_targets) where loss is a scalar tensor and
            resolved_targets has the same shape as true_labels.
        """
        ...

    @abstractmethod
    def summarize_prediction(
        self,
        model: nn.Module,
        inputs: torch.Tensor,
        true_labels: torch.Tensor,
        label_map: Dict[int, str],
    ) -> None:
        """Print/log a prediction summary for a single sample."""
        ...
