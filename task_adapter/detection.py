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

    def build_objective_components(
        self,
        model: nn.Module,
        inputs: torch.Tensor,
        gt_instances: List[Dict[str, torch.Tensor]],
        context: Dict[str, Any],
    ) -> Dict[str, torch.Tensor]:
        return {'total': self.build_objective(model, inputs, gt_instances, context)}


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
        components = self.build_objective_components(model, inputs, gt_instances, context)
        return components['total']

    def build_objective_components(
        self,
        model: nn.Module,
        inputs: torch.Tensor,
        gt_instances: List[Dict[str, torch.Tensor]],
        context: Dict[str, Any],
    ) -> Dict[str, torch.Tensor]:
        labels = self._build_yolox_labels(inputs, gt_instances)
        with self._training_forward_mode(model):
            out = model(inputs, labels)

        if not isinstance(out, dict) or 'total_loss' not in out:
            raise ValueError("YOLOX official objective expects dict output with key 'total_loss'.")
        return {
            'total': out['total_loss'],
            'obj': out['conf_loss'],
            'cls': out['cls_loss'],
            'iou': out['iou_loss'],
        }

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

    def _classify_instances(
        self,
        pred_boxes: torch.Tensor,
        pred_cls_ids: torch.Tensor,
        pred_best_scores: torch.Tensor,
        gt_boxes: torch.Tensor,
        gt_classes: torch.Tensor,
        iou_threshold: float,
    ) -> Dict[str, Any]:
        """Split predictions into TP / FP-B / FP-A and derive FN indices.

        Matching is greedy by prediction confidence and mirrors the overlay policy.
        This routine is intentionally non-differentiable and only defines index sets.
        """
        num_preds = int(pred_boxes.size(0))
        num_gt = int(gt_boxes.size(0))

        if num_gt == 0:
            return {
                'tp_pairs': [],
                'fp_cls_pairs': [],
                'fp_loc_pred_indices': list(range(num_preds)),
                'fn_gt_indices': [],
            }

        order = torch.argsort(pred_best_scores.detach(), descending=True)
        matched_gt = torch.zeros((num_gt,), dtype=torch.bool, device=gt_boxes.device)

        tp_pairs: List[Tuple[int, int]] = []
        fp_cls_pairs: List[Tuple[int, int]] = []
        fp_loc_pred_indices: List[int] = []

        for pidx_t in order:
            pidx = int(pidx_t.item())
            pbox = pred_boxes[pidx]
            pcls = int(pred_cls_ids[pidx].item())

            unmatched = torch.nonzero(~matched_gt, as_tuple=False).squeeze(1)
            if unmatched.numel() == 0:
                fp_loc_pred_indices.append(pidx)
                continue

            same_cls = unmatched[gt_classes[unmatched] == pcls]
            if same_cls.numel() > 0:
                ious_same = torch.stack(
                    [self._box_iou_xyxy_vec(pbox.unsqueeze(0), gt_boxes[g]).squeeze(0) for g in same_cls],
                    dim=0,
                )
                best_local = int(torch.argmax(ious_same).item())
                best_iou = float(ious_same[best_local].item())
                best_gt = int(same_cls[best_local].item())
                if best_iou >= iou_threshold:
                    matched_gt[best_gt] = True
                    tp_pairs.append((pidx, best_gt))
                    continue

            ious_any = torch.stack(
                [self._box_iou_xyxy_vec(pbox.unsqueeze(0), gt_boxes[g]).squeeze(0) for g in unmatched],
                dim=0,
            )
            best_any_local = int(torch.argmax(ious_any).item())
            best_any_iou = float(ious_any[best_any_local].item())
            best_any_gt = int(unmatched[best_any_local].item())

            if best_any_iou >= iou_threshold and int(gt_classes[best_any_gt].item()) != pcls:
                fp_cls_pairs.append((pidx, best_any_gt))
            else:
                fp_loc_pred_indices.append(pidx)

        fn_gt_indices = torch.nonzero(~matched_gt, as_tuple=False).squeeze(1).tolist()
        return {
            'tp_pairs': tp_pairs,
            'fp_cls_pairs': fp_cls_pairs,
            'fp_loc_pred_indices': fp_loc_pred_indices,
            'fn_gt_indices': fn_gt_indices,
        }

    def _build_hotness_unified_components(
        self,
        outputs: torch.Tensor,
        gt_instances: List[Dict[str, torch.Tensor]],
        context: Dict[str, Any],
    ) -> Dict[str, torch.Tensor]:
        eps = 1e-8
        match_iou = float(context.get('det_match_iou_threshold', 0.5))
        fp_loc_score_power = float(context.get('det_fp_loc_score_power', 1.0))
        fp_cls_margin = float(context.get('det_fp_cls_margin', 0.1))

        w_tp = float(context.get('det_hotness_weight_tp', 0.0))
        w_fn = float(context.get('det_hotness_weight_fn', 1.0))
        w_fp_a = float(context.get('det_hotness_weight_fp_a', 1.0))
        w_fp_b = float(context.get('det_hotness_weight_fp_b', 1.0))
        gate_alpha = float(context.get('det_hotness_gate_alpha', 1.0))

        obj = outputs[:, :, 4:5]
        cls = outputs[:, :, 5:]
        pred_scores = obj * cls
        pred_best_scores, pred_best_cls_ids = pred_scores.max(dim=2)
        pred_boxes = self._cxcywh_to_xyxy(outputs[:, :, :4])

        tp_per_image: List[torch.Tensor] = []
        fn_per_image: List[torch.Tensor] = []
        fp_a_per_image: List[torch.Tensor] = []
        fp_b_per_image: List[torch.Tensor] = []

        for bidx in range(outputs.size(0)):
            gt_item = gt_instances[bidx]
            gt_classes = gt_item['class_ids'].to(device=outputs.device, dtype=torch.long)
            gt_boxes = gt_item['boxes_xyxy'].to(device=outputs.device, dtype=outputs.dtype)

            class_split = self._classify_instances(
                pred_boxes[bidx],
                pred_best_cls_ids[bidx],
                pred_best_scores[bidx],
                gt_boxes,
                gt_classes,
                iou_threshold=match_iou,
            )

            tp_losses = []
            for pidx, gt_idx in class_split['tp_pairs']:
                cls_id = int(gt_classes[gt_idx].item())
                s_gt = pred_scores[bidx, pidx, cls_id].clamp_min(eps)
                tp_losses.append(-torch.log(s_gt))
            tp_loss = torch.stack(tp_losses).mean() if tp_losses else self._zero_loss_like(outputs)

            fn_losses = []
            for gt_idx in class_split['fn_gt_indices']:
                cls_id = int(gt_classes[gt_idx].item())
                cls_scores = pred_scores[bidx, :, cls_id].clamp_min(eps)
                ious = self._box_iou_xyxy_vec(pred_boxes[bidx], gt_boxes[gt_idx])
                logits = 3.0 * ious + 1.0 * torch.log(cls_scores)
                weights = torch.softmax(logits, dim=0)
                covered_score = torch.sum(weights * cls_scores)
                fn_losses.append(-torch.log(covered_score.clamp_min(eps)))
            fn_loss = torch.stack(fn_losses).mean() if fn_losses else self._zero_loss_like(outputs)

            fp_a_losses = []
            for pidx in class_split['fp_loc_pred_indices']:
                fp_a_losses.append(torch.pow(pred_best_scores[bidx, pidx].clamp_min(eps), fp_loc_score_power))
            fp_a_loss = torch.stack(fp_a_losses).mean() if fp_a_losses else self._zero_loss_like(outputs)

            fp_b_losses = []
            for pidx, gt_idx in class_split['fp_cls_pairs']:
                gt_cls_id = int(gt_classes[gt_idx].item())
                gt_score = pred_scores[bidx, pidx, gt_cls_id]
                cls_row = pred_scores[bidx, pidx]
                wrong_mask = torch.ones_like(cls_row, dtype=torch.bool)
                wrong_mask[gt_cls_id] = False
                if wrong_mask.any():
                    wrong_score = cls_row[wrong_mask].max()
                    fp_b_losses.append(torch.clamp(wrong_score - gt_score + fp_cls_margin, min=0.0))
            fp_b_loss = torch.stack(fp_b_losses).mean() if fp_b_losses else self._zero_loss_like(outputs)

            tp_per_image.append(tp_loss)
            fn_per_image.append(fn_loss)
            fp_a_per_image.append(fp_a_loss)
            fp_b_per_image.append(fp_b_loss)

        tp = torch.stack(tp_per_image).mean()
        fn = torch.stack(fn_per_image).mean()
        fp_a = torch.stack(fp_a_per_image).mean()
        fp_b = torch.stack(fp_b_per_image).mean()

        gated_tp = torch.pow(tp.clamp_min(eps), gate_alpha)
        gated_fn = torch.pow(fn.clamp_min(eps), gate_alpha)
        gated_fp_a = torch.pow(fp_a.clamp_min(eps), gate_alpha)
        gated_fp_b = torch.pow(fp_b.clamp_min(eps), gate_alpha)

        total = (
            w_tp * gated_tp
            + w_fn * gated_fn
            + w_fp_a * gated_fp_a
            + w_fp_b * gated_fp_b
        )

        return {
            'tp': tp,
            'fn': fn,
            'fp_a': fp_a,
            'fp_b': fp_b,
            'tp_gated': gated_tp,
            'fn_gated': gated_fn,
            'fp_a_gated': gated_fp_a,
            'fp_b_gated': gated_fp_b,
            'total': total,
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

    @classmethod
    def _nms_indices_by_class(
        cls,
        boxes_xyxy: torch.Tensor,
        class_ids: torch.Tensor,
        scores: torch.Tensor,
        iou_threshold: float,
    ) -> torch.Tensor:
        if boxes_xyxy.numel() == 0:
            return torch.empty((0,), dtype=torch.long, device=boxes_xyxy.device)

        keep = []
        unique_classes = torch.unique(class_ids)
        for class_id in unique_classes:
            class_mask = class_ids == class_id
            class_indices = torch.nonzero(class_mask, as_tuple=False).squeeze(1)
            if class_indices.numel() == 0:
                continue

            class_scores = scores[class_indices]
            order = class_indices[torch.argsort(class_scores, descending=True)]

            while order.numel() > 0:
                current = int(order[0].item())
                keep.append(current)
                if order.numel() == 1:
                    break

                rest = order[1:]
                ious = cls._box_iou_xyxy_vec(boxes_xyxy[rest], boxes_xyxy[current])
                order = rest[ious <= float(iou_threshold)]

        if not keep:
            return torch.empty((0,), dtype=torch.long, device=boxes_xyxy.device)

        keep_tensor = torch.tensor(keep, dtype=torch.long, device=boxes_xyxy.device)
        keep_scores = scores[keep_tensor]
        return keep_tensor[torch.argsort(keep_scores, descending=True)]

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

    def _build_fp_loc_objective(
        self,
        outputs: torch.Tensor,
        gt_instances: List[Dict[str, torch.Tensor]],
        context: Dict[str, Any],
    ) -> torch.Tensor:
        """Penalize high-confidence predictions at locations with low IoU to any GT.

        This term targets false positives of type-2 (wrong location / no matching GT).
        """
        eps = 1e-8
        conf_threshold = float(context.get('det_conf_threshold', 0.3))
        nms_iou_threshold = float(context.get('det_nms_iou_threshold', 0.45))
        loc_iou_threshold = float(
            context.get(
                'det_fp_loc_iou_threshold',
                context.get('det_match_iou_threshold', 0.5),
            )
        )
        score_power = float(context.get('det_fp_loc_score_power', 1.0))

        obj = outputs[:, :, 4:5]
        cls = outputs[:, :, 5:]
        pred_scores = obj * cls
        scores, cls_ids = pred_scores.max(dim=2)

        per_image_losses = []
        for bidx in range(outputs.size(0)):
            pred_boxes = self._cxcywh_to_xyxy(outputs[bidx, :, :4])
            pred_cls_ids = cls_ids[bidx]
            pred_score = scores[bidx]

            conf_mask = pred_score >= conf_threshold
            if conf_mask.sum() == 0:
                per_image_losses.append(pred_score.sum() * 0.0)
                continue

            pred_boxes = pred_boxes[conf_mask]
            pred_cls_ids = pred_cls_ids[conf_mask]
            pred_score = pred_score[conf_mask]

            keep = self._nms_indices_by_class(
                pred_boxes,
                pred_cls_ids,
                pred_score,
                iou_threshold=nms_iou_threshold,
            )
            if keep.numel() == 0:
                per_image_losses.append(pred_score.sum() * 0.0)
                continue

            pred_boxes = pred_boxes[keep]
            pred_score = pred_score[keep]
            gt_boxes = gt_instances[bidx]['boxes_xyxy'].to(device=outputs.device, dtype=outputs.dtype)

            if gt_boxes.numel() == 0:
                fp_loc_scores = pred_score
            else:
                ious = [self._box_iou_xyxy_vec(pred_boxes, gt_boxes[gidx]) for gidx in range(gt_boxes.size(0))]
                max_iou = torch.stack(ious, dim=1).max(dim=1).values
                fp_loc_mask = max_iou < loc_iou_threshold
                fp_loc_scores = pred_score[fp_loc_mask]

            if fp_loc_scores.numel() == 0:
                per_image_losses.append(pred_score.sum() * 0.0)
                continue

            per_image_losses.append(torch.mean(torch.pow(fp_loc_scores.clamp_min(eps), score_power)))

        return torch.stack(per_image_losses).mean()

    @staticmethod
    def _zero_loss_like(outputs: torch.Tensor) -> torch.Tensor:
        return outputs.sum() * 0.0

    def _build_empty_gt_components(
        self,
        outputs: torch.Tensor,
        gt_instances: List[Dict[str, torch.Tensor]],
        context: Dict[str, Any],
    ) -> Dict[str, torch.Tensor]:
        zero = self._zero_loss_like(outputs)
        fp_loc = self._build_fp_loc_objective(outputs, gt_instances, context)
        return {
            'obj': zero,
            'cls': zero,
            'iou': zero,
            'fp_loc': fp_loc,
            'total': fp_loc,
        }

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
        fp_loc_weight = float(context.get('det_fp_loc_weight', 0.0))

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

        if objective_mode == 'hotness_unified':
            outputs = model(inputs)
            if outputs.ndim != 3 or outputs.size(-1) < 6:
                raise ValueError(
                    'DetectionTaskAdapter expects model outputs shaped as (B, N, 5 + C).'
                )
            components = self._build_hotness_unified_components(outputs, gt_instances, context)
            return components['total'], self._resolve_targets_from_gt(gt_instances, true_labels)

        det_has_gt = bool(context.get('det_has_gt', True))
        if not det_has_gt:
            outputs = model(inputs)
            if outputs.ndim != 3 or outputs.size(-1) < 6:
                raise ValueError(
                    'DetectionTaskAdapter expects model outputs shaped as (B, N, 5 + C).'
                )
            components = self._build_empty_gt_components(outputs, gt_instances, context)
            return components['total'], self._resolve_targets_from_gt(gt_instances, true_labels)

        provider = self._resolve_provider(model, context)
        if provider is None and provider_strict and str(context.get('det_objective_provider', self.objective_provider)) != 'none':
            requested = str(context.get('det_objective_provider', self.objective_provider))
            raise ValueError(f"Requested objective provider '{requested}' is not supported by this model.")

        if provider is not None:
            loss = provider.build_objective(model, inputs, gt_instances, context)
            if fp_loc_weight > 0.0:
                outputs_fp = model(inputs)
                if outputs_fp.ndim != 3 or outputs_fp.size(-1) < 6:
                    raise ValueError(
                        'DetectionTaskAdapter expects model outputs shaped as (B, N, 5 + C).'
                    )
                loss = loss + fp_loc_weight * self._build_fp_loc_objective(outputs_fp, gt_instances, context)
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
                "Choose 'gt_all_instances', 'gt_all_classes', 'hotness_unified', or 'legacy_single_class'."
            )
        if fp_loc_weight > 0.0:
            loss = loss + fp_loc_weight * self._build_fp_loc_objective(outputs, gt_instances, context)
        return loss, self._resolve_targets_from_gt(gt_instances, true_labels)

    def build_objective_components(
        self,
        model: nn.Module,
        inputs: torch.Tensor,
        true_labels: torch.Tensor,
        target_spec: TargetSpec,
        objective_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, torch.Tensor]:
        context = dict(objective_context or {})
        objective_mode = str(context.get('det_objective_mode', self.objective_mode))
        fp_loc_weight = float(context.get('det_fp_loc_weight', 0.0))

        if objective_mode == 'legacy_single_class':
            loss, _ = self.build_objective(
                model,
                inputs,
                true_labels,
                target_spec,
                objective_context=objective_context,
            )
            return {'total': loss}

        gt_instances = context.get('det_gt_instances')
        if gt_instances is None:
            raise ValueError(
                "det_gt_instances is required for detection objective modes other than 'legacy_single_class'."
            )

        det_has_gt = bool(context.get('det_has_gt', True))
        if not det_has_gt:
            outputs = model(inputs)
            if outputs.ndim != 3 or outputs.size(-1) < 6:
                raise ValueError(
                    'DetectionTaskAdapter expects model outputs shaped as (B, N, 5 + C).'
                )
            if objective_mode == 'hotness_unified':
                return self._build_hotness_unified_components(outputs, gt_instances, context)
            return self._build_empty_gt_components(outputs, gt_instances, context)

        if objective_mode == 'hotness_unified':
            outputs = model(inputs)
            if outputs.ndim != 3 or outputs.size(-1) < 6:
                raise ValueError(
                    'DetectionTaskAdapter expects model outputs shaped as (B, N, 5 + C).'
                )
            return self._build_hotness_unified_components(outputs, gt_instances, context)

        provider = self._resolve_provider(model, context)
        if provider is not None:
            components = dict(provider.build_objective_components(model, inputs, gt_instances, context))
            if fp_loc_weight > 0.0:
                outputs_fp = model(inputs)
                if outputs_fp.ndim != 3 or outputs_fp.size(-1) < 6:
                    raise ValueError(
                        'DetectionTaskAdapter expects model outputs shaped as (B, N, 5 + C).'
                    )
                fp_loc = self._build_fp_loc_objective(outputs_fp, gt_instances, context)
                components['fp_loc'] = fp_loc
                components['total'] = components['total'] + fp_loc_weight * fp_loc
            return components

        loss, _ = self.build_objective(
            model,
            inputs,
            true_labels,
            target_spec,
            objective_context=objective_context,
        )
        return {'total': loss}

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
