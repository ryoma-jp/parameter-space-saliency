"""Stage1 pipeline module.

Computes per-object filter-wise saliency for one image and saves
{run_dir}/{image_stem}/stage1_filter_saliency.pth.
"""

import json
import os

import cv2
import numpy as np
import torch

from model_adapter.factory import build_model_adapter
from task_adapter.detection import YOLOXPerObjectProvider
from parameter_saliency.saliency_model_backprop import compute_per_object_filter_saliency


def _build_model_spec(args) -> dict:
    if args.model_source == 'torchvision':
        return {'source': 'torchvision', 'name': args.model, 'pretrained': True}
    spec = {'source': 'custom_module', 'class_path': args.model_class_path}
    if args.model_weights_path:
        spec['weights_path'] = args.model_weights_path
    if args.model_import_root:
        spec['import_roots'] = args.model_import_root
    if args.model_kwargs_json:
        spec['model_kwargs'] = json.loads(args.model_kwargs_json)
    if args.preprocess_cfg_json:
        spec['preprocess'] = json.loads(args.preprocess_cfg_json)
    if args.state_dict_target_path:
        spec['state_dict_target_path'] = args.state_dict_target_path
    return spec


def _load_coco_gt_for_image(ann_path: str, basename: str, image_hw, preprocess_cfg: dict):
    with open(ann_path, 'r') as f:
        coco = json.load(f)

    image_info = next(
        (img for img in coco.get('images', []) if img.get('file_name') == basename),
        None,
    )
    if image_info is None:
        return [], [], {}, None

    sorted_categories = sorted(coco.get('categories', []), key=lambda cat: cat['id'])
    cat_id_to_cls = {cat['id']: idx for idx, cat in enumerate(sorted_categories)}
    cls_to_name = {idx: cat.get('name', str(cat['id'])) for idx, cat in enumerate(sorted_categories)}

    raw_w = float(image_info['width'])
    raw_h = float(image_info['height'])
    h, w = image_hw

    use_letterbox = bool(preprocess_cfg.get('letterbox', False))
    if use_letterbox:
        r = min(float(w) / raw_w, float(h) / raw_h)
        sx = sy = r
    else:
        sx = float(w) / raw_w
        sy = float(h) / raw_h

    gt_boxes, gt_classes = [], []
    image_id = image_info['id']
    for ann in coco.get('annotations', []):
        if ann.get('image_id') != image_id:
            continue
        if ann.get('iscrowd', 0) == 1:
            continue
        cat_id = ann.get('category_id')
        if cat_id not in cat_id_to_cls:
            continue
        x, y, bw, bh = ann['bbox']
        gt_boxes.append([x * sx, y * sy, (x + bw) * sx, (y + bh) * sy])
        gt_classes.append(cat_id_to_cls[cat_id])

    return gt_boxes, gt_classes, cls_to_name, {'sx': sx, 'sy': sy}


def _xyxy_to_cxcywh(boxes_xyxy: np.ndarray) -> np.ndarray:
    out = np.zeros_like(boxes_xyxy)
    out[:, 0] = (boxes_xyxy[:, 0] + boxes_xyxy[:, 2]) / 2.0
    out[:, 1] = (boxes_xyxy[:, 1] + boxes_xyxy[:, 3]) / 2.0
    out[:, 2] = np.clip(boxes_xyxy[:, 2] - boxes_xyxy[:, 0], 0.0, None)
    out[:, 3] = np.clip(boxes_xyxy[:, 3] - boxes_xyxy[:, 1], 0.0, None)
    return out


def _make_gt_instance(boxes_xyxy: np.ndarray, class_ids: np.ndarray) -> dict:
    boxes_xyxy = np.asarray(boxes_xyxy, dtype=np.float32).reshape(-1, 4)
    class_ids = np.asarray(class_ids, dtype=np.int64).reshape(-1)
    boxes_cxcywh = _xyxy_to_cxcywh(boxes_xyxy) if boxes_xyxy.shape[0] > 0 else np.zeros((0, 4), np.float32)
    return {
        'class_ids': torch.from_numpy(class_ids).to(dtype=torch.long),
        'boxes_xyxy': torch.from_numpy(boxes_xyxy),
        'boxes_cxcywh': torch.from_numpy(boxes_cxcywh),
    }


def _input_box_to_orig(box_input, scale_info):
    sx = scale_info['sx']
    sy = scale_info['sy']
    x1, y1, x2, y2 = box_input
    return [x1 / sx, y1 / sy, x2 / sx, y2 / sy]


def run_stage1_filter_saliency(args):
    if not args.run_dir:
        raise ValueError('--run_dir is required for pipeline_stage=stage1')
    if not args.image_path:
        raise ValueError('--image_path is required for pipeline_stage=stage1')
    if not args.det_annotations_json:
        raise ValueError('--det_annotations_json is required for pipeline_stage=stage1')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    preprocess_cfg = json.loads(args.preprocess_cfg_json) if args.preprocess_cfg_json else {}

    spec = _build_model_spec(args)
    adapter = build_model_adapter(spec)
    net = adapter.build_model().to(device)
    net.eval()

    raw_img = cv2.imread(args.image_path)
    if raw_img is None:
        raise FileNotFoundError(f'Image not found: {args.image_path}')
    orig_h, orig_w = raw_img.shape[:2]

    preprocess = adapter.get_preprocess()
    input_tensor = preprocess(raw_img).unsqueeze(0).to(device)
    model_h, model_w = int(input_tensor.shape[-2]), int(input_tensor.shape[-1])

    basename = os.path.basename(args.image_path)
    gt_boxes, gt_classes, _, scale_info = _load_coco_gt_for_image(
        args.det_annotations_json, basename, (model_h, model_w), preprocess_cfg
    )
    if scale_info is None:
        print(f'[SKIP] No COCO GT entry for {basename}')
        return

    stem = os.path.splitext(basename)[0]
    image_id = stem.lstrip('0') or '0'

    gt_instance = _make_gt_instance(
        np.array(gt_boxes, dtype=np.float32) if gt_boxes else np.zeros((0, 4), np.float32),
        np.array(gt_classes, dtype=np.int64) if gt_classes else np.zeros((0,), np.int64),
    )

    context = {
        'det_conf_threshold': args.det_conf_threshold,
        'det_nms_iou_threshold': args.det_nms_iou_threshold,
        'det_match_iou_threshold': args.det_match_iou_threshold,
        'det_fn_tau': args.det_fn_tau,
        'det_gt_instances': [gt_instance],
    }

    provider = YOLOXPerObjectProvider()
    if not provider.supports(net, context):
        raise RuntimeError('Model does not support YOLOXPerObjectProvider.')

    input_tensor.requires_grad_(True)
    per_obj = provider.build_per_object_losses(net, input_tensor, [gt_instance], context)
    saliencies = compute_per_object_filter_saliency(net, per_obj, create_graph=False, signed=False)

    with torch.no_grad():
        infer_out = net(input_tensor)
    pred_np = infer_out[0].detach().cpu().numpy()
    obj_s = pred_np[:, 4]
    cls_s = pred_np[:, 5:]
    cls_ids_np = np.argmax(cls_s, axis=1)
    cls_conf = cls_s[np.arange(cls_s.shape[0]), cls_ids_np]
    scores_np = obj_s * cls_conf

    pred_boxes_np = pred_np[:, :4].copy()
    pred_boxes_np[:, 0] = pred_np[:, 0] - pred_np[:, 2] / 2
    pred_boxes_np[:, 1] = pred_np[:, 1] - pred_np[:, 3] / 2
    pred_boxes_np[:, 2] = pred_np[:, 0] + pred_np[:, 2] / 2
    pred_boxes_np[:, 3] = pred_np[:, 1] + pred_np[:, 3] / 2

    conf_mask = scores_np >= args.det_conf_threshold
    nms_boxes_np = pred_boxes_np[conf_mask]
    nms_cls_np = cls_ids_np[conf_mask]
    nms_scores_np = scores_np[conf_mask]

    classification = per_obj.get('classification', {})
    tp_pairs = classification.get('tp_pairs', [])
    fp_b_pairs = classification.get('fp_cls_pairs', [])
    fp_a_inds = classification.get('fp_loc_pred_indices', [])
    fn_gt_inds = classification.get('fn_gt_indices', [])

    def _nms_box(local_idx):
        if local_idx < len(nms_boxes_np):
            return nms_boxes_np[local_idx].tolist()
        return [0.0, 0.0, 0.0, 0.0]

    def _nms_score(local_idx):
        if local_idx < len(nms_scores_np):
            return float(nms_scores_np[local_idx])
        return 0.0

    def _nms_cls(local_idx):
        if local_idx < len(nms_cls_np):
            return int(nms_cls_np[local_idx])
        return -1

    gt_boxes_arr = np.array(gt_boxes, dtype=np.float32) if gt_boxes else np.zeros((0, 4), np.float32)

    tp_records, fp_a_records, fp_b_records, fn_records = [], [], [], []

    for i, (pidx, gt_idx) in enumerate(tp_pairs):
        sal = saliencies['tp'][i]
        box_input = _nms_box(pidx)
        tp_records.append({
            'saliency': sal.detach().cpu() if sal is not None else None,
            'cls_id': int(gt_classes[gt_idx]) if gt_idx < len(gt_classes) else -1,
            'score': _nms_score(pidx),
            'box_input': box_input,
            'box_orig': _input_box_to_orig(box_input, scale_info),
        })

    for i, pidx in enumerate(fp_a_inds):
        sal = saliencies['fp_a'][i] if i < len(saliencies['fp_a']) else None
        box_input = _nms_box(pidx)
        fp_a_records.append({
            'saliency': sal.detach().cpu() if sal is not None else None,
            'cls_id': _nms_cls(pidx),
            'score': _nms_score(pidx),
            'box_input': box_input,
            'box_orig': _input_box_to_orig(box_input, scale_info),
        })

    for i, (pidx, _) in enumerate(fp_b_pairs):
        sal = saliencies['fp_b'][i] if i < len(saliencies['fp_b']) else None
        box_input = _nms_box(pidx)
        fp_b_records.append({
            'saliency': sal.detach().cpu() if sal is not None else None,
            'cls_id': _nms_cls(pidx),
            'score': _nms_score(pidx),
            'box_input': box_input,
            'box_orig': _input_box_to_orig(box_input, scale_info),
        })

    for i, gt_idx in enumerate(fn_gt_inds):
        sal = saliencies['fn'][i] if i < len(saliencies['fn']) else None
        box_input = gt_boxes_arr[gt_idx].tolist() if gt_idx < len(gt_boxes_arr) else [0.0, 0.0, 0.0, 0.0]
        fn_records.append({
            'saliency': sal.detach().cpu() if sal is not None else None,
            'cls_id': int(gt_classes[gt_idx]) if gt_idx < len(gt_classes) else -1,
            'box_input': box_input,
            'box_orig': _input_box_to_orig(box_input, scale_info),
        })

    output = {
        'image_id': image_id,
        'model_input_hw': [model_h, model_w],
        'original_hw': [orig_h, orig_w],
        'tp': tp_records,
        'fn': fn_records,
        'fp_a': fp_a_records,
        'fp_b': fp_b_records,
    }

    out_dir = os.path.join(args.run_dir, stem)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'stage1_filter_saliency.pth')
    torch.save(output, out_path)

    n_fn_skip = sum(1 for r in fn_records if r['saliency'] is None)
    print(
        f'[Stage1] {stem}: TP={len(tp_records)} FN={len(fn_records)}(skip={n_fn_skip}) '
        f'FP-A={len(fp_a_records)} FP-B={len(fp_b_records)} -> {out_path}'
    )
