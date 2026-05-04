"""Stage2 pipeline module.

Phase 2a: TP-based normalization stats.
Phase 2b: per-object z-score normalization and optional input-space projection.
"""

import glob
import json
import os
from functools import lru_cache
from typing import Optional

import cv2
import numpy as np
import torch
import torch.autograd as autograd
import torch.nn.functional as F
from matplotlib.patches import Rectangle

from model_adapter.factory import build_model_adapter
from task_adapter.detection import YOLOXPerObjectProvider
from parameter_saliency.saliency_model_backprop import compute_per_object_filter_saliency
from utils import show_heatmap_on_image


class WelfordStats:
    def __init__(self):
        self.n = 0
        self.mean: Optional[torch.Tensor] = None
        self.M2: Optional[torch.Tensor] = None

    def update(self, x: torch.Tensor):
        x = x.detach().float().cpu()
        if self.mean is None:
            self.mean = torch.zeros_like(x)
            self.M2 = torch.zeros_like(x)
        self.n += 1
        delta = x - self.mean
        self.mean.add_(delta / self.n)
        self.M2.add_(delta * (x - self.mean))

    def finalize(self):
        if self.n == 0:
            return None, None
        if self.n < 2:
            return self.mean, torch.ones_like(self.mean)
        return self.mean, (self.M2 / (self.n - 1)).sqrt()


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
    image_info = next((img for img in coco.get('images', []) if img.get('file_name') == basename), None)
    if image_info is None:
        return [], [], None
    sorted_cats = sorted(coco.get('categories', []), key=lambda c: c['id'])
    cat_id_to_cls = {c['id']: idx for idx, c in enumerate(sorted_cats)}
    h, w = image_hw
    raw_w = float(image_info['width'])
    raw_h = float(image_info['height'])
    if bool(preprocess_cfg.get('letterbox', False)):
        r = min(float(w) / raw_w, float(h) / raw_h)
        sx = sy = r
    else:
        sx = float(w) / raw_w
        sy = float(h) / raw_h
    gt_boxes, gt_classes = [], []
    for ann in coco.get('annotations', []):
        if ann.get('image_id') != image_info['id']:
            continue
        if ann.get('iscrowd', 0) == 1:
            continue
        cat_id = ann.get('category_id')
        if cat_id not in cat_id_to_cls:
            continue
        x, y, bw, bh = ann['bbox']
        gt_boxes.append([x * sx, y * sy, (x + bw) * sx, (y + bh) * sy])
        gt_classes.append(cat_id_to_cls[cat_id])
    return gt_boxes, gt_classes, {'sx': sx, 'sy': sy}


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


def _find_image(image_dir: str, stem: str) -> Optional[str]:
    for ext in ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'):
        p = os.path.join(image_dir, stem + ext)
        if os.path.isfile(p):
            return p
    return None


@lru_cache(maxsize=4)
def _load_class_names(ann_path: str) -> dict:
    if not ann_path or not os.path.isfile(ann_path):
        return {}
    with open(ann_path, 'r') as f:
        coco = json.load(f)
    sorted_cats = sorted(coco.get('categories', []), key=lambda c: c['id'])
    return {idx: cat.get('name', str(cat['id'])) for idx, cat in enumerate(sorted_cats)}


def run_phase_2a(run_dir: str) -> str:
    stage1_paths = sorted(glob.glob(os.path.join(run_dir, '*', 'stage1_filter_saliency.pth')))
    if not stage1_paths:
        raise FileNotFoundError(f'No stage1_filter_saliency.pth found under {run_dir}')

    welford = WelfordStats()
    n_tp_total = 0
    for path in stage1_paths:
        data = torch.load(path, map_location='cpu', weights_only=False)
        for rec in data.get('tp', []):
            sal = rec.get('saliency')
            if sal is not None:
                welford.update(sal)
                n_tp_total += 1

    mu, sigma = welford.finalize()
    if mu is None:
        raise ValueError('No TP saliencies found; cannot compute TP normalization stats.')

    stats = {'mean': mu, 'std': sigma, 'n_tp': n_tp_total, 'n_filters': mu.shape[0]}
    out_path = os.path.join(run_dir, 'tp_norm_stats.pth')
    torch.save(stats, out_path)
    print(f'[Phase 2a] TP stats: n_tp={n_tp_total}, n_filters={mu.shape[0]} -> {out_path}')
    return out_path


def _normalize(saliency: Optional[torch.Tensor], mu: torch.Tensor, sigma: torch.Tensor, eps: float):
    if saliency is None:
        return None
    return (saliency.float().cpu() - mu) / (sigma + eps)


def _compute_projection(saliency_with_graph, mu_tp, sigma_tp, input_tensor, top_f, boost_k, eps):
    try:
        s_hat_k = (saliency_with_graph - mu_tp.to(saliency_with_graph.device)) / (
            sigma_tp.to(saliency_with_graph.device) + eps
        )
        with torch.no_grad():
            k = min(top_f, s_hat_k.shape[0])
            top_indices = s_hat_k.detach().topk(k)[1]
            s_prime = s_hat_k.detach().clone()
            s_prime[top_indices] = s_prime[top_indices] * boost_k

        cos_dist = 1.0 - F.cosine_similarity(s_hat_k.unsqueeze(0), s_prime.unsqueeze(0), dim=1)
        (grad_x,) = autograd.grad(cos_dist, input_tensor, retain_graph=True)
        return grad_x.detach().abs().squeeze(0).sum(0).cpu()
    except Exception as exc:
        print(f'    [WARN] input projection failed: {exc}')
        return None


def _normalize_01(arr: np.ndarray) -> np.ndarray:
    arr_min = float(np.min(arr))
    arr_max = float(np.max(arr))
    denom = arr_max - arr_min
    if denom <= 1e-12:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - arr_min) / denom).astype(np.float32)


def _prepare_overlay_mask(proj_map: torch.Tensor) -> np.ndarray:
    arr = proj_map.float().numpy()
    clipped = arr.copy()
    hi = float(np.percentile(clipped, 99)) if clipped.size > 0 else 0.0
    lo = float(np.percentile(clipped, 90)) if clipped.size > 0 else 0.0
    if hi > lo:
        clipped[clipped > hi] = hi
        clipped[clipped < lo] = lo
    return _normalize_01(clipped)


def _project_map_to_original(
    proj_map: torch.Tensor,
    original_hw,
    model_input_hw,
    preprocess_cfg: dict,
) -> np.ndarray:
    orig_h, orig_w = original_hw
    in_h, in_w = model_input_hw
    arr = proj_map.float().numpy()
    if bool(preprocess_cfg.get('letterbox', False)):
        r = min(float(in_w) / float(orig_w), float(in_h) / float(orig_h))
        valid_w = max(1, min(in_w, int(round(orig_w * r))))
        valid_h = max(1, min(in_h, int(round(orig_h * r))))
        arr = arr[:valid_h, :valid_w]
    return cv2.resize(arr, (int(orig_w), int(orig_h)), interpolation=cv2.INTER_LINEAR)


def _iter_overlay_items(data: dict):
    color_map = {
        'tp': 'lime',
        'fn': 'yellow',
        'fp_a': 'red',
        'fp_b': 'orange',
    }
    for key in ('tp', 'fn', 'fp_a', 'fp_b'):
        for item in data.get(key, []):
            yield key, color_map[key], item


def _save_detection_boxes_png(
    raw_img_bgr: np.ndarray,
    data: dict,
    class_names: dict,
    out_path: str,
):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        image_rgb = cv2.cvtColor(raw_img_bgr, cv2.COLOR_BGR2RGB)
        fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
        ax.imshow(image_rgb)
        for key, color, item in _iter_overlay_items(data):
            x1, y1, x2, y2 = item['box_orig']
            cls_id = int(item.get('cls_id', -1))
            cls_name = class_names.get(cls_id, str(cls_id)) if cls_id >= 0 else 'unknown'
            rect = Rectangle(
                (x1, y1),
                max(1.0, x2 - x1),
                max(1.0, y2 - y1),
                fill=False,
                edgecolor=color,
                linewidth=2.0,
            )
            ax.add_patch(rect)
            ax.text(
                x1,
                max(0.0, y1 - 2.0),
                cls_name,
                color=color,
                fontsize=8,
                fontweight='bold',
                ha='left',
                va='bottom',
            )
        ax.axis('off')
        fig.savefig(out_path, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
    except Exception as exc:
        print(f'    [WARN] detection boxes PNG save failed: {exc}')


def _save_heatmap_png(
    proj_map: torch.Tensor,
    out_path: str,
    raw_img_bgr: np.ndarray,
    data: dict,
    preprocess_cfg: dict,
    class_names: dict,
):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        orig_h, orig_w = raw_img_bgr.shape[:2]
        model_input_hw = tuple(data.get('model_input_hw') or proj_map.shape)
        proj_orig = _project_map_to_original(proj_map, (orig_h, orig_w), model_input_hw, preprocess_cfg)
        heatmap_mask = _prepare_overlay_mask(torch.from_numpy(proj_orig))

        image_rgb = cv2.cvtColor(raw_img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        heatmap_superimposed = show_heatmap_on_image(image_rgb, heatmap_mask)

        fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
        ax.imshow(heatmap_superimposed)
        for key, color, item in _iter_overlay_items(data):
            x1, y1, x2, y2 = item['box_orig']
            cls_id = int(item.get('cls_id', -1))
            cls_name = class_names.get(cls_id, str(cls_id)) if cls_id >= 0 else 'unknown'
            rect = Rectangle(
                (x1, y1),
                max(1.0, x2 - x1),
                max(1.0, y2 - y1),
                fill=False,
                edgecolor=color,
                linewidth=2.0,
            )
            ax.add_patch(rect)
            ax.text(
                x1,
                max(0.0, y1 - 2.0),
                cls_name,
                color=color,
                fontsize=8,
                fontweight='bold',
                ha='left',
                va='bottom',
            )
        ax.axis('off')
        fig.savefig(out_path, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
    except Exception as exc:
        print(f'    [WARN] heatmap PNG save failed: {exc}')


def _run_phase_2b_image(stage1_path, tp_stats, args, net, adapter, provider, preprocess_cfg, device, do_projection, do_viz):
    data = torch.load(stage1_path, map_location='cpu', weights_only=False)
    img_dir = os.path.dirname(stage1_path)
    stem = os.path.basename(img_dir)

    mu_tp = tp_stats['mean']
    sigma_tp = tp_stats['std']
    eps = args.stage2_norm_eps

    sals_with_graph = {}
    input_tensor = None
    raw_img = None
    class_names = _load_class_names(args.det_annotations_json) if args.det_annotations_json else {}

    if do_viz or do_projection:
        img_path = _find_image(args.pipeline_image_dir, stem)
        if img_path is None:
            print(f'  [WARN] image not found for {stem} in {args.pipeline_image_dir}; skipping visualization')
            do_viz = False
            do_projection = False

    if do_viz or do_projection:
        raw_img = cv2.imread(img_path)
        _save_detection_boxes_png(
            raw_img, data, class_names,
            os.path.join(img_dir, 'detection_boxes.png'),
        )

    if do_projection:
        input_tensor = adapter.get_preprocess()(raw_img).unsqueeze(0).to(device)
        input_tensor.requires_grad_(True)

        model_h, model_w = int(input_tensor.shape[-2]), int(input_tensor.shape[-1])
        basename = os.path.basename(img_path)
        if args.det_annotations_json:
            gt_boxes, gt_classes, _ = _load_coco_gt_for_image(
                args.det_annotations_json, basename, (model_h, model_w), preprocess_cfg
            )
        else:
            gt_boxes, gt_classes = [], []

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
        per_obj = provider.build_per_object_losses(net, input_tensor, [gt_instance], context)
        _sals = compute_per_object_filter_saliency(net, per_obj, create_graph=True, signed=False)
        for typ in ('tp', 'fp_a', 'fp_b', 'fn'):
            sals_with_graph[typ] = _sals.get(typ, [])

    def _build_records(type_name: str, need_proj: bool):
        out_records = []
        src_records = data.get(type_name, [])
        graph_sals = sals_with_graph.get(type_name, [])
        for i, rec in enumerate(src_records):
            out_rec = {k: v for k, v in rec.items() if k != 'saliency'}
            out_rec['saliency_norm'] = _normalize(rec.get('saliency'), mu_tp, sigma_tp, eps)
            proj_map = None
            if need_proj and do_projection and i < len(graph_sals) and graph_sals[i] is not None:
                proj_map = _compute_projection(
                    graph_sals[i], mu_tp, sigma_tp, input_tensor,
                    args.stage2_top_f, args.stage2_boost_k, eps,
                )
                if proj_map is not None:
                    _save_heatmap_png(
                        proj_map,
                        os.path.join(img_dir, f'proj_{type_name}_{i:03d}.png'),
                        raw_img,
                        data,
                        preprocess_cfg,
                        class_names,
                    )
            out_rec['input_projection'] = proj_map
            out_records.append(out_rec)
        return out_records

    output = {
        'image_id': data.get('image_id', stem),
        'model_input_hw': data.get('model_input_hw'),
        'original_hw': data.get('original_hw'),
        'tp': _build_records('tp', True),
        'fn': _build_records('fn', True),
        'fp_a': _build_records('fp_a', True),
        'fp_b': _build_records('fp_b', True),
    }

    out_path = os.path.join(img_dir, 'stage2_normalized_saliency.pth')
    torch.save(output, out_path)
    print(
        f'[Phase 2b] {stem}: TP={len(output["tp"])} FN={len(output["fn"])} '
        f'FP-A={len(output["fp_a"])} FP-B={len(output["fp_b"])} -> {out_path}'
    )


def run_phase_2b(run_dir: str, tp_stats: dict, args):
    stage1_paths = sorted(glob.glob(os.path.join(run_dir, '*', 'stage1_filter_saliency.pth')))
    if not stage1_paths:
        raise FileNotFoundError(f'No stage1_filter_saliency.pth found under {run_dir}')

    do_viz = bool(args.pipeline_image_dir)
    do_projection = do_viz and (not args.stage2_no_projection)
    if do_projection:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        preprocess_cfg = json.loads(args.preprocess_cfg_json) if args.preprocess_cfg_json else {}
        adapter = build_model_adapter(_build_model_spec(args))
        net = adapter.build_model().to(device)
        net.eval()
        provider = YOLOXPerObjectProvider()
    else:
        device, preprocess_cfg, adapter, net, provider = 'cpu', {}, None, None, None

    print(f'[Phase 2b] Processing {len(stage1_paths)} images (projection={do_projection}, viz={do_viz})')
    for path in stage1_paths:
        _run_phase_2b_image(path, tp_stats, args, net, adapter, provider, preprocess_cfg, device, do_projection, do_viz)


def run_stage2_normalize_project(args):
    if not args.run_dir:
        raise ValueError('--run_dir is required for pipeline_stage=stage2')

    stats_path = os.path.join(args.run_dir, 'tp_norm_stats.pth')
    if args.stage2_phase in ('all', '2a'):
        run_phase_2a(args.run_dir)

    if args.stage2_phase in ('all', '2b'):
        if not os.path.isfile(stats_path):
            raise FileNotFoundError(f'TP norm stats not found at {stats_path}. Run stage2_phase=2a first.')
        tp_stats = torch.load(stats_path, map_location='cpu', weights_only=False)
        run_phase_2b(args.run_dir, tp_stats, args)
