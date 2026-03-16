from typing import Dict, Tuple

import torch
import torch.nn as nn

from .base import TaskAdapter
from target.spec import TargetSpec, TargetType


class DetectionTaskAdapter(TaskAdapter):
    """Task adapter for object detection models that output (B, N, 5 + C).

    The expected prediction layout is:
    - box coordinates: [:, :, :4]
    - objectness:      [:, :, 4]
    - class scores:    [:, :, 5:]

    The objective is a differentiable scalar built from per-box detection scores.
    """

    def _resolve_target_classes(
        self,
        outputs: torch.Tensor,
        true_labels: torch.Tensor,
        target_spec: TargetSpec,
    ) -> torch.Tensor:
        """Resolve one class id per image for the detection objective."""
        obj = outputs[:, :, 4:5]
        cls = outputs[:, :, 5:]
        scores = obj * cls

        if target_spec.type == TargetType.SPECIFIED_CLASS:
            return torch.full_like(true_labels, int(target_spec.class_id))

        if target_spec.type == TargetType.TRUE_LABEL:
            return true_labels

        # predicted_top1: resolve class by max score over all boxes/classes.
        bsz, _, num_classes = scores.shape
        flat_scores = scores.reshape(bsz, -1)
        best_flat = flat_scores.argmax(dim=1)
        best_class = best_flat % num_classes
        return best_class.to(true_labels.device)

    def build_objective(
        self,
        model: nn.Module,
        inputs: torch.Tensor,
        true_labels: torch.Tensor,
        target_spec: TargetSpec,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        outputs = model(inputs)
        if outputs.ndim != 3 or outputs.size(-1) < 6:
            raise ValueError(
                "DetectionTaskAdapter expects model outputs shaped as (B, N, 5 + C)."
            )

        resolved = self._resolve_target_classes(outputs, true_labels, target_spec)

        obj = outputs[:, :, 4:5]
        cls = outputs[:, :, 5:]
        scores = obj * cls

        # Gather class-conditional detection score and take top box per image.
        gather_idx = resolved.view(-1, 1, 1).expand(-1, scores.size(1), 1)
        cls_scores = torch.gather(scores, dim=2, index=gather_idx).squeeze(-1)
        top_scores = cls_scores.max(dim=1).values

        # Convert confidence objective to a loss-like scalar.
        loss = -torch.log(top_scores.clamp_min(1e-8)).mean()
        return loss, resolved

    def summarize_prediction(
        self,
        model: nn.Module,
        inputs: torch.Tensor,
        true_labels: torch.Tensor,
        label_map: Dict[int, str],
    ) -> None:
        with torch.no_grad():
            outputs = model(inputs)

        if outputs.ndim != 3 or outputs.size(-1) < 6:
            print("\n        Detection summary unavailable: unexpected output shape.\n")
            return

        obj = outputs[:, :, 4:5]
        cls = outputs[:, :, 5:]
        scores = obj * cls

        # Best detection across boxes/classes for the first sample.
        first = scores[0]
        best_box_idx, best_class_idx = torch.nonzero(
            first == first.max(), as_tuple=True
        )
        box_i = int(best_box_idx[0].item())
        cls_i = int(best_class_idx[0].item())
        best_score = float(first[box_i, cls_i].item())

        box = outputs[0, box_i, :4].detach().cpu().tolist()
        true_idx = int(true_labels[0].item())

        print(
            f"\n"
            f"        Image target label:        {true_idx}\n"
            f"        Image target class name:   {label_map.get(true_idx, str(true_idx))}\n"
            f"        Top detection class:       {cls_i}\n"
            f"        Top detection class name:  {label_map.get(cls_i, str(cls_i))}\n"
            f"        Top detection score:       {best_score:.6f}\n"
            f"        Top detection box(cxcywh): [{box[0]:.2f}, {box[1]:.2f}, {box[2]:.2f}, {box[3]:.2f}]\n"
        )
