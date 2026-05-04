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


class YOLOXPerObjectProvider(DetectionObjectiveProvider):
    """Compute per-object (per-GT) losses using YOLOX SimOTA assignment.

    Returns a dict with keys 'tp', 'fp_b', 'fp_a', 'fn', each a list of
    scalar loss Tensors (backprop-able). FN uses SimOTA re-run on free anchors.
    None entries in 'fn' indicate unresolvable FN GTs (e.g. fully occluded).
    """

    name = 'yolox_per_object'

    def supports(self, model: nn.Module, context: Dict[str, Any]) -> bool:
        base_model = model.module if isinstance(model, nn.DataParallel) else model
        return hasattr(base_model, 'head') and hasattr(getattr(base_model, 'head'), 'get_per_object_losses')

    def build_objective(
        self,
        model: nn.Module,
        inputs: torch.Tensor,
        gt_instances: List[Dict[str, torch.Tensor]],
        context: Dict[str, Any],
    ) -> torch.Tensor:
        per_obj = self.build_per_object_losses(model, inputs, gt_instances, context)
        all_losses: List[torch.Tensor] = []
        for key in ('tp', 'fp_a', 'fp_b', 'fn'):
            for item in per_obj[key]:
                if item is not None:
                    all_losses.append(item)
        if not all_losses:
            from task_adapter.detection import DetectionTaskAdapter
            return inputs.sum() * 0.0
        return torch.stack(all_losses).mean()

    def build_per_object_losses(
        self,
        model: nn.Module,
        inputs: torch.Tensor,
        gt_instances: List[Dict[str, torch.Tensor]],
        context: Dict[str, Any],
    ) -> Dict[str, list]:
        """Return per-object loss dicts for Stage1 saliency computation.

        Structure:
            {
              'tp':   List[Tensor scalar],  one per TP GT (paired pred anchor)
              'fp_b': List[Tensor scalar],  one per FP-B pred (class-confusion)
              'fp_a': List[Tensor scalar],  one per FP-A pred (wrong location)
              'fn':   List[Optional[Tensor scalar]],  one per FN GT; None = unresolvable
              'classification': dict with 'tp_pairs', 'fp_cls_pairs', 'fp_loc_pred_indices', 'fn_gt_indices'
            }
        Only batch_idx=0 is supported (single-image inference).
        """
        match_iou = float(context.get('det_match_iou_threshold', 0.5))
        conf_threshold = float(context.get('det_conf_threshold', 0.3))
        nms_iou_threshold = float(context.get('det_nms_iou_threshold', 0.45))
        tau_fn = float(context.get('det_fn_tau', 0.1))

        labels = YOLOXOfficialLossProvider()._build_yolox_labels(inputs, gt_instances)

        # Run training-mode forward to get raw head outputs + shifts
        base_model = model.module if isinstance(model, nn.DataParallel) else model
        head = base_model.head

        # Collect raw forward outputs (training mode) and shifts
        outputs_list, origin_preds, x_shifts, y_shifts, expanded_strides = [], [], [], [], []
        with YOLOXOfficialLossProvider()._training_forward_mode(model):
            # We need to replicate the forward logic to get x_shifts etc.
            # Use head directly via _run_head_training
            for k, (cls_conv, reg_conv, stride_this_level, x) in enumerate(
                zip(head.cls_convs, head.reg_convs, head.strides, base_model.backbone(inputs) if hasattr(base_model, 'backbone') else self._get_fpn_features(base_model, inputs))
            ):
                x = head.stems[k](x)
                cls_feat = head.cls_convs[k](x)
                cls_output = head.cls_preds[k](cls_feat)
                reg_feat = head.reg_convs[k](x)
                reg_output = head.reg_preds[k](reg_feat)
                obj_output = head.obj_preds[k](reg_feat)

                output, grid = head.get_output_and_grid(
                    torch.cat([reg_output, obj_output, cls_output], 1),
                    k, stride_this_level, inputs.type(),
                )
                x_shifts.append(grid[:, :, 0])
                y_shifts.append(grid[:, :, 1])
                expanded_strides.append(
                    torch.zeros(1, grid.shape[1]).fill_(stride_this_level).type_as(inputs)
                )
                if head.use_l1:
                    bs, _, hsize, wsize = reg_output.shape
                    reg_out_l1 = reg_output.view(bs, 1, 4, hsize, wsize)
                    reg_out_l1 = reg_out_l1.permute(0, 1, 3, 4, 2).reshape(bs, -1, 4)
                    origin_preds.append(reg_out_l1.clone())
                outputs_list.append(output)

        raw_outputs = torch.cat(outputs_list, 1)  # (B, n_anchors, 5+C)

        # Normal SimOTA: get per-GT losses for TP/FP-B (assigned anchors)
        per_obj_result = head.get_per_object_losses(
            inputs, x_shifts, y_shifts, expanded_strides,
            labels, raw_outputs, origin_preds, dtype=inputs.dtype,
        )

        # Inference-mode forward for NMS-based FP-A/FP-B classification
        prev_training = model.training
        with torch.no_grad():
            model.eval()
            infer_outputs = model(inputs)
            model.train(prev_training)

        batch_idx = 0
        gt_item = gt_instances[batch_idx]
        gt_classes = gt_item['class_ids'].to(device=inputs.device, dtype=torch.long)
        gt_boxes   = gt_item['boxes_xyxy'].to(device=inputs.device, dtype=inputs.dtype)

        # NMS on inference outputs
        obj_s  = infer_outputs[batch_idx, :, 4:5]
        cls_s  = infer_outputs[batch_idx, :, 5:]
        scores_2d = (obj_s * cls_s)
        pred_best_scores, pred_best_cls_ids = scores_2d.max(dim=1)
        conf_mask = pred_best_scores >= conf_threshold

        pred_boxes_all = DetectionTaskAdapter._cxcywh_to_xyxy(infer_outputs[batch_idx, :, :4])
        if conf_mask.sum() > 0:
            kept_boxes  = pred_boxes_all[conf_mask]
            kept_cls    = pred_best_cls_ids[conf_mask]
            kept_scores = pred_best_scores[conf_mask]
            keep_idx    = DetectionTaskAdapter._nms_indices_by_class(
                kept_boxes, kept_cls, kept_scores, nms_iou_threshold
            )
            nms_boxes   = kept_boxes[keep_idx]
            nms_cls     = kept_cls[keep_idx]
            nms_scores  = kept_scores[keep_idx]
        else:
            nms_boxes  = pred_boxes_all.new_zeros((0, 4))
            nms_cls    = gt_classes.new_zeros((0,))
            nms_scores = pred_best_scores.new_zeros((0,))

        from task_adapter.detection import DetectionTaskAdapter as _DTA
        classification = _DTA._classify_instances(
            _DTA.__new__(_DTA),
            nms_boxes, nms_cls, nms_scores,
            gt_boxes, gt_classes,
            iou_threshold=match_iou,
        )

        result_tp:   List[torch.Tensor]          = []
        result_fp_b: List[torch.Tensor]          = []
        result_fp_a: List[torch.Tensor]          = []
        result_fn:   List[Optional[torch.Tensor]] = []

        # TP: per-GT loss from SimOTA result
        simota = per_obj_result[batch_idx]
        if simota is not None:
            per_gt_losses = simota['per_gt_losses']
            for pidx, gt_idx in classification['tp_pairs']:
                loss_k = per_gt_losses[gt_idx]
                result_tp.append(loss_k)

            # FP-B: use per-gt loss of the GT they overlap with
            for pidx, gt_idx in classification['fp_cls_pairs']:
                # Use the loss of the mis-matched GT anchor set as proxy signal
                loss_k = per_gt_losses[gt_idx]
                result_fp_b.append(loss_k)

            # FP-A: per-anchor total loss for NMS-surviving FP-A anchors
            # Map NMS box back to raw anchor index by closest box match
            fp_a_losses = self._compute_fp_a_losses(
                raw_outputs, classification['fp_loc_pred_indices'],
                nms_boxes, simota['fg_mask'], head,
                x_shifts, y_shifts, expanded_strides,
                origin_preds if head.use_l1 else None,
                batch_idx,
            )
            result_fp_a = fp_a_losses

            # FN: SimOTA re-run on free anchors
            fn_gt_inds = classification['fn_gt_indices']
            if fn_gt_inds:
                gt_boxes_cxcywh = gt_item['boxes_cxcywh'].to(device=inputs.device, dtype=inputs.dtype)
                fn_losses = head.compute_fn_losses(
                    batch_idx=batch_idx,
                    fn_gt_indices=fn_gt_inds,
                    gt_bboxes_per_image=gt_boxes_cxcywh[:len(gt_classes)],
                    gt_classes=gt_classes,
                    outputs=raw_outputs,
                    x_shifts=x_shifts,
                    y_shifts=y_shifts,
                    expanded_strides=expanded_strides,
                    origin_preds=origin_preds if head.use_l1 else None,
                    fg_mask=simota['fg_mask'],
                    tau_fn=tau_fn,
                )
                result_fn = fn_losses

        return {
            'tp':   result_tp,
            'fp_b': result_fp_b,
            'fp_a': result_fp_a,
            'fn':   result_fn,
            'classification': classification,
        }

    def _get_fpn_features(self, base_model: nn.Module, inputs: torch.Tensor):
        """Extract FPN features from a YOLOX model."""
        fpn_outs = base_model.neck(base_model.backbone(inputs)) if hasattr(base_model, 'neck') else base_model.backbone(inputs)
        return fpn_outs

    def _compute_fp_a_losses(
        self,
        raw_outputs: torch.Tensor,
        fp_a_nms_indices: List[int],
        nms_boxes: torch.Tensor,
        fg_mask: torch.Tensor,
        head,
        x_shifts, y_shifts, expanded_strides,
        origin_preds,
        batch_idx: int,
    ) -> List[torch.Tensor]:
        """Compute per-FP-A object loss.

        FP-A predictions are NMS survivors with no matching GT.
        For each NMS-surviving FP-A box, find the closest raw anchor and compute
        its detector loss with objectness target=0 (suppress it).
        """
        if not fp_a_nms_indices:
            return []

        x_shifts_cat         = torch.cat(x_shifts, 1)[0]
        y_shifts_cat         = torch.cat(y_shifts, 1)[0]
        expanded_strides_cat = torch.cat(expanded_strides, 1)[0]

        bbox_preds = raw_outputs[batch_idx, :, :4]   # (n_anchors, 4) decoded cxcywh
        obj_preds  = raw_outputs[batch_idx, :, 4:5]  # (n_anchors, 1)
        cls_preds  = raw_outputs[batch_idx, :, 5:]   # (n_anchors, n_cls)

        losses = []
        for nms_local_idx in fp_a_nms_indices:
            if nms_local_idx >= nms_boxes.shape[0]:
                continue
            fp_box_xyxy = nms_boxes[nms_local_idx]   # (4,) xyxy
            fp_cx = (fp_box_xyxy[0] + fp_box_xyxy[2]) / 2.0
            fp_cy = (fp_box_xyxy[1] + fp_box_xyxy[3]) / 2.0

            # Anchor centres
            anchor_cx = (x_shifts_cat + 0.5) * expanded_strides_cat
            anchor_cy = (y_shifts_cat + 0.5) * expanded_strides_cat
            dists = (anchor_cx - fp_cx) ** 2 + (anchor_cy - fp_cy) ** 2
            anchor_idx = int(dists.argmin().item())

            # Objectness suppression loss (target=0 means suppress)
            obj_target = torch.zeros(1, 1, device=raw_outputs.device, dtype=raw_outputs.dtype)
            loss_obj = head.bcewithlog_loss(
                obj_preds[anchor_idx:anchor_idx+1], obj_target
            ).sum()
            losses.append(loss_obj)

        return losses


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
            YOLOXPerObjectProvider.name: YOLOXPerObjectProvider(),
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
                "Choose 'gt_all_instances', 'gt_all_classes', or 'legacy_single_class'."
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
            return self._build_empty_gt_components(outputs, gt_instances, context)

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
