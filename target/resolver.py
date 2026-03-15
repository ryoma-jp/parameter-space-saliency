import torch

from .spec import TargetSpec, TargetType


class TargetResolver:
    """Resolves a TargetSpec to a concrete target tensor at runtime."""

    def resolve(
        self,
        outputs: torch.Tensor,
        true_labels: torch.Tensor,
        spec: TargetSpec,
    ) -> torch.Tensor:
        """Return a target label tensor matching true_labels shape.

        Args:
            outputs:     Model logits (B, C).
            true_labels: Ground-truth class indices (B,).
            spec:        Target specification.

        Returns:
            Resolved target tensor (B,) on the same device as true_labels.
        """
        if spec.type == TargetType.TRUE_LABEL:
            return true_labels
        elif spec.type == TargetType.PREDICTED_TOP1:
            with torch.no_grad():
                _, predicted = outputs.detach().max(1)
            return predicted.to(true_labels.device)
        elif spec.type == TargetType.SPECIFIED_CLASS:
            return torch.full_like(true_labels, spec.class_id)
        else:
            raise ValueError(f"Unknown TargetType: {spec.type}")
