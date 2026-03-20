from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.modules.batchnorm as batchnorm

from .base import TaskAdapter
from target.spec import TargetSpec, TargetType


class DetectionObjectiveProvider(ABC):
    """Model-specific objective provider for detection tasks."""

    name = 'base'

    @abstractmethod
    def supports(self, model: nn.Module, context: Dict[str, Any]) -> bool:
        ...

    @abstractmethod
    def build_objective(
        self,
        model: nn.Module,
        inputs: torch.Tensor,
        gt_instances: List[Dict[str, torch.Tensor]],
        context: Dict[str, Any],
    ) -> torch.Tensor:
        ...


class YOLOXOfficialLossProvider(DetectionObjectiveProvider):
    """Use YOLOX official training loss by forwarding (inputs, labels)."""

    name = 'yolox_official'

    def supports(self, model: nn.Module, context: Dict[str, Any]) -> bool:
        base_model = model.module if isinstance(model, nn.DataParallel) else model
        return hasattr(base_model, 'head') and hasattr(getattr(base_model, 'head'), 'get_losses')

    def build_objective(
        self,
        model: nn.Module,
        inputs: torch.Tensor,
        gt_instances: List[Dict[str, torch.Tensor]],
        context: Dict[str, Any],
    ) -> torch.Tensor:
        labels = self._build_yolox_labels(inputs, gt_instances)
        with self._training_forward_mode(model):
            out = model(inputs, labels)

        if not isinstance(out, dict) or 'total_loss' not in out:
            raise ValueError("YOLOX official objective expects dict output with key 'total_loss'.")
        return out['total_loss']

    def _build_yolox_labels(
        self,
        inputs: torch.Tensor,
        gt_instances: List[Dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        if len(gt_instances) != inputs.size(0):
            raise ValueError('det_gt_instances batch size does not match inputs batch size.')

        max_gt = max((int(item['class_ids'].numel()) for item in gt_instances), default=0)
        if max_gt == 0:
            raise ValueError('YOLOX official loss requires at least one GT instance.')

        labels = inputs.new_zeros((inputs.size(0), max_gt, 5))
        for bidx, item in enumerate(gt_instances):
            class_ids = item['class_ids'].to(device=inputs.device, dtype=inputs.dtype)
            boxes_cxcywh = item['boxes_cxcywh'].to(device=inputs.device, dtype=inputs.dtype)
            n = int(class_ids.numel())
            if n == 0:
                continue
            labels[bidx, :n, 0] = class_ids
            labels[bidx, :n, 1:5] = boxes_cxcywh
        return labels

    @contextmanager
    def _training_forward_mode(self, model: nn.Module):
        modules = list(model.modules())
        training_flags = {id(mod): mod.training for mod in modules}
        try:
            model.train()
            for mod in modules:
                if isinstance(mod, batchnorm._BatchNorm):
                    mod.eval()
            yield
        finally:
            for mod in modules:
                mod.train(training_flags[id(mod)])


class DetectionTaskAdapter(TaskAdapter):
    """Task adapter for object detection models that output (B, N, 5 + C).

    The expected prediction layout is:
    - box coordinates: [:, :, :4]
    - objectness:      [:, :, 4]
    - class scores:    [:, :, 5:]

    The objective is a differentiable scalar built from per-box detection scores.
    """

    def __init__(
        self,
        objective_mode: str = 'gt_all_instances',
        objective_provider: str = 'auto',
        provider_strict: bool = False,
    ):
        self.objective_mode = objective_mode
        self.objective_provider = objective_provider
        self.provider_strict = provider_strict
        self._providers: Dict[str, DetectionObjectiveProvider] = {
            YOLOXOfficialLossProvider.name: YOLOXOfficialLossProvider(),
        }

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

    def _resolve_provider(
        self,
        model: nn.Module,
        context: Dict[str, Any],
    ) -> Optional[DetectionObjectiveProvider]:
        provider_name = str(context.get('det_objective_provider', self.objective_provider))
        if provider_name == 'none':
            return None

        if provider_name == 'auto':
            candidates = [self._providers[YOLOXOfficialLossProvider.name]]
        else:
            provider = self._providers.get(provider_name)
            if provider is None:
                raise ValueError(f"Unknown detection objective provider: '{provider_name}'.")
            candidates = [provider]

        for provider in candidates:
            if provider.supports(model, context):
                return provider
        return None

    @staticmethod
    def _cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
        out = boxes.clone()
        out[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.0
        out[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.0
        out[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.0
        out[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.0
        return out

    @staticmethod
    def _box_iou_xyxy_vec(boxes: torch.Tensor, box: torch.Tensor) -> torch.Tensor:
        x1 = torch.maximum(boxes[:, 0], box[0])
        y1 = torch.maximum(boxes[:, 1], box[1])
        x2 = torch.minimum(boxes[:, 2], box[2])
        y2 = torch.minimum(boxes[:, 3], box[3])

        inter_w = (x2 - x1).clamp_min(0.0)
        inter_h = (y2 - y1).clamp_min(0.0)
        inter = inter_w * inter_h

        area_a = (boxes[:, 2] - boxes[:, 0]).clamp_min(0.0) * (boxes[:, 3] - boxes[:, 1]).clamp_min(0.0)
        area_b = (box[2] - box[0]).clamp_min(0.0) * (box[3] - box[1]).clamp_min(0.0)
        union = area_a + area_b - inter
        return inter / union.clamp_min(1e-8)

    def _build_gt_all_instances_objective(
        self,
        outputs: torch.Tensor,
        gt_instances: List[Dict[str, torch.Tensor]],
        context: Dict[str, Any],
    ) -> torch.Tensor:
        eps = 1e-8
        iou_weight = float(context.get('det_iou_weight', 3.0))
        score_weight = float(context.get('det_score_weight', 1.0))

        obj = outputs[:, :, 4:5]
        cls = outputs[:, :, 5:]
        scores = obj * cls

        per_image_losses = []
        for bidx in range(outputs.size(0)):
            gt_item = gt_instances[bidx]
            gt_classes = gt_item['class_ids'].to(device=outputs.device, dtype=torch.long)
            gt_boxes = gt_item['boxes_xyxy'].to(device=outputs.device, dtype=outputs.dtype)
            if gt_classes.numel() == 0:
                raise ValueError('gt_all_instances objective requires at least one GT instance per image.')

            pred_boxes = self._cxcywh_to_xyxy(outputs[bidx, :, :4])
            pred_scores = scores[bidx]

            gt_losses = []
            for gidx in range(gt_classes.numel()):
                cls_id = int(gt_classes[gidx].item())
                cls_scores = pred_scores[:, cls_id].clamp_min(eps)
                ious = self._box_iou_xyxy_vec(pred_boxes, gt_boxes[gidx])
                logits = iou_weight * ious + score_weight * torch.log(cls_scores)
                weights = torch.softmax(logits, dim=0)
                covered_score = torch.sum(weights * cls_scores)
                gt_losses.append(-torch.log(covered_score.clamp_min(eps)))

            per_image_losses.append(torch.stack(gt_losses).mean())

        return torch.stack(per_image_losses).mean()

    def _build_gt_all_classes_objective(
        self,
        outputs: torch.Tensor,
        gt_instances: List[Dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        eps = 1e-8
        obj = outputs[:, :, 4:5]
        cls = outputs[:, :, 5:]
        scores = obj * cls

        per_image_losses = []
        for bidx in range(outputs.size(0)):
            gt_classes = gt_instances[bidx]['class_ids'].to(device=outputs.device, dtype=torch.long)
            unique_classes = torch.unique(gt_classes)
            if unique_classes.numel() == 0:
                raise ValueError('gt_all_classes objective requires at least one GT class per image.')

            cls_losses = []
            for cls_id in unique_classes:
                cls_scores = scores[bidx, :, int(cls_id.item())]
                cls_losses.append(-torch.log(cls_scores.max().clamp_min(eps)))
            per_image_losses.append(torch.stack(cls_losses).mean())

        return torch.stack(per_image_losses).mean()

    def _resolve_targets_from_gt(
        self,
        gt_instances: List[Dict[str, torch.Tensor]],
        true_labels: torch.Tensor,
    ) -> torch.Tensor:
        resolved = true_labels.new_full(true_labels.shape, -1)
        for bidx, item in enumerate(gt_instances):
            class_ids = item['class_ids']
            if class_ids.numel() > 0:
                resolved[bidx] = int(class_ids[0].item())
        return resolved

    def build_objective(
        self,
        model: nn.Module,
        inputs: torch.Tensor,
        true_labels: torch.Tensor,
        target_spec: TargetSpec,
        objective_context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        context = dict(objective_context or {})
        objective_mode = str(context.get('det_objective_mode', self.objective_mode))
        provider_strict = bool(context.get('det_provider_strict', self.provider_strict))

        if objective_mode == 'legacy_single_class':
            outputs = model(inputs)
            if outputs.ndim != 3 or outputs.size(-1) < 6:
                raise ValueError(
                    'DetectionTaskAdapter expects model outputs shaped as (B, N, 5 + C).'
                )
            resolved = self._resolve_target_classes(outputs, true_labels, target_spec)
            obj = outputs[:, :, 4:5]
            cls = outputs[:, :, 5:]
            scores = obj * cls
            gather_idx = resolved.view(-1, 1, 1).expand(-1, scores.size(1), 1)
            cls_scores = torch.gather(scores, dim=2, index=gather_idx).squeeze(-1)
            top_scores = cls_scores.max(dim=1).values
            loss = -torch.log(top_scores.clamp_min(1e-8)).mean()
            return loss, resolved

        gt_instances = context.get('det_gt_instances')
        if gt_instances is None:
            raise ValueError(
                "det_gt_instances is required for detection objective modes other than 'legacy_single_class'."
            )

        provider = self._resolve_provider(model, context)
        if provider is None and provider_strict and str(context.get('det_objective_provider', self.objective_provider)) != 'none':
            requested = str(context.get('det_objective_provider', self.objective_provider))
            raise ValueError(f"Requested objective provider '{requested}' is not supported by this model.")

        if provider is not None:
            loss = provider.build_objective(model, inputs, gt_instances, context)
            return loss, self._resolve_targets_from_gt(gt_instances, true_labels)

        outputs = model(inputs)
        if outputs.ndim != 3 or outputs.size(-1) < 6:
            raise ValueError(
                'DetectionTaskAdapter expects model outputs shaped as (B, N, 5 + C).'
            )

        if objective_mode == 'gt_all_instances':
            loss = self._build_gt_all_instances_objective(outputs, gt_instances, context)
        elif objective_mode == 'gt_all_classes':
            loss = self._build_gt_all_classes_objective(outputs, gt_instances)
        else:
            raise ValueError(
                f"Unknown detection objective mode: '{objective_mode}'. "
                "Choose 'gt_all_instances', 'gt_all_classes', or 'legacy_single_class'."
            )
        return loss, self._resolve_targets_from_gt(gt_instances, true_labels)

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
