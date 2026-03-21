import yaml
import json
import urllib
import torch
import torch.backends.cudnn as cudnn
import torchvision
import argparse
import os
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import cv2
import warnings

from utils import show_heatmap_on_image, test_and_find_incorrectly_classified, transform_raw_image
from parameter_saliency.saliency_model_backprop import SaliencyModel, find_testset_saliency
from model_adapter.factory import build_model_adapter
from task_adapter.classification import ClassificationTaskAdapter
from task_adapter.detection import DetectionTaskAdapter
from target.spec import TargetSpec, TargetType

parser = argparse.ArgumentParser(description='Parameter-Space and Input-Space Saliency')

# ----- Model -----
parser.add_argument('--model', default='resnet50', type=str,
                    help='torchvision model name (used when --model_source torchvision)')
parser.add_argument('--model_source', default='torchvision',
                    choices=['torchvision', 'custom_module'],
                    help='source of the model')
parser.add_argument('--model_class_path', default=None, type=str,
                    help='fully-qualified class path for custom_module, e.g. mypkg.models.MyNet')
parser.add_argument('--model_weights_path', default=None, type=str,
                    help='path to weights checkpoint for custom_module')
parser.add_argument('--model_import_root', action='append', default=[],
                    help='extra import root to prepend before resolving --model_class_path; repeatable')
parser.add_argument('--model_kwargs_json', default=None, type=str,
                    help='JSON object with keyword args passed to the custom model constructor/factory')
parser.add_argument('--preprocess_cfg_json', default=None, type=str,
                    help='JSON object overriding preprocessing, e.g. {"resize":[416,416],"crop":null}')
parser.add_argument('--state_dict_target_path', default=None, type=str,
                    help='optional dotted attribute path under the constructed model to receive load_state_dict')
parser.add_argument('--export_model_pth', default=None, type=str,
                    help='export the loaded model weights to a .pth checkpoint and continue execution')
parser.add_argument('--task', default='classification', choices=['classification', 'detection'],
                    help='task adapter to use for objective construction')

# ----- Dataset -----
parser.add_argument('--data_to_use', default='ImageNet', type=str,
                    help='which dataset to use (currently only ImageNet)')
parser.add_argument('--imagenet_val_path', default='<insert-ImageNet-val-path-here>',
                    type=str, help='ImageNet validation set path')

# ----- Target -----
parser.add_argument('--target_type', default='true_label',
                    choices=['true_label', 'predicted_top1', 'specified_class'],
                    help='what to use as the saliency target')
parser.add_argument('--target_class_id', default=None, type=int,
                    help='class id when --target_type specified_class')

# ----- Label map -----
parser.add_argument('--label_map_path', default=None, type=str,
                    help='path to YAML label map {int: str}; '
                         'if omitted, ImageNet labels are downloaded for torchvision models')

# ----- Output -----
parser.add_argument('--figure_folder_name', default='input_space_saliency', type=str,
                    help='subdirectory under output_root for input-space figures')
parser.add_argument('--output_root', default='figures', type=str,
                    help='root directory to save output figures')

# ----- Saliency options -----
parser.add_argument('--signed', action='store_true', help='Use signed saliency')
parser.add_argument('--logit', action='store_true',
                    help='[deprecated, not implemented] Use logits rather than cross-entropy')
parser.add_argument('--logit_difference', action='store_true',
                    help='[deprecated, not implemented] Use logit difference as loss')

# ----- Input-space saliency (boosting) -----
parser.add_argument('--boost_factor', default=100.0, type=float,
                    help='boost factor for salient filters')
parser.add_argument('--k_salient', default=10, type=int,
                    help='number of top salient filters to boost')
parser.add_argument('--compare_random', action='store_true',
                    help='boost k random filters for comparison')

# ----- SmoothGrad-like options -----
parser.add_argument('--noise_iters', default=1, type=int,
                    help='number of noise iterations to average')
parser.add_argument('--noise_percent', default=0, type=float,
                    help='std of the noise')

# ----- Reference image -----
parser.add_argument('--image_path',
                    default='raw_images/great_white_shark_mispred_as_killer_whale.jpeg',
                    type=str, help='path to a raw image file')
parser.add_argument('--image_target_label', default=None, type=int,
                    help='ground-truth class index (0-based) for the raw image')
parser.add_argument('--reference_id', default=None, type=int,
                    help='index of image in the validation set')
parser.add_argument('--det_annotations_json', default=None, type=str,
                    help='COCO annotations JSON path for TP/FP/FN overlay in detection task')
parser.add_argument('--det_conf_threshold', default=0.3, type=float,
                    help='confidence threshold for detection overlay')
parser.add_argument('--det_nms_iou_threshold', default=0.45, type=float,
                    help='NMS IoU threshold for detection overlay')
parser.add_argument('--det_match_iou_threshold', default=0.5, type=float,
                    help='IoU threshold for TP/FP matching against GT')
parser.add_argument('--det_objective_mode',
                    default='gt_all_instances',
                    choices=['gt_all_instances', 'gt_all_classes', 'legacy_single_class'],
                    help='objective mode for detection saliency loss')
parser.add_argument('--det_objective_provider',
                    default='auto',
                    choices=['auto', 'none', 'yolox_official'],
                    help='optional model-specific objective provider')
parser.add_argument('--det_provider_strict', action='store_true',
                    help='error if requested detection objective provider is unavailable')
parser.add_argument('--det_iou_weight', default=3.0, type=float,
                    help='weight of IoU term in gt_all_instances objective')
parser.add_argument('--det_score_weight', default=1.0, type=float,
                    help='weight of class-score term in gt_all_instances objective')
parser.add_argument('--input_saliency_method',
                default='auto',
                choices=['auto', 'matching', 'direct_loss'],
                help='how to compute input-space saliency gradients: '
                    'matching (original PSS), direct_loss (dL/dx), or auto '
                    '(classification=matching, detection=direct_loss)')

def _cache_key(args) -> str:
    """Derive a filesystem-safe cache key from model arguments."""
    if args.model_source == 'torchvision':
        return args.model
    return args.model_class_path.split('.')[-1]


def _get_preprocess_cfg(args) -> dict:
    if not hasattr(args, '_cached_preprocess_cfg'):
        cfg = {}
        if args.preprocess_cfg_json:
            cfg = json.loads(args.preprocess_cfg_json)
        args._cached_preprocess_cfg = cfg
    return args._cached_preprocess_cfg


def _load_label_map(args) -> dict:
    """Load or download a label map {int -> str}."""
    if args.label_map_path:
        with open(args.label_map_path) as f:
            return yaml.load(f, Loader=yaml.Loader)
    if args.model_source == 'torchvision':
        label_url = urllib.request.urlopen(
            'https://gist.githubusercontent.com/yrevar/942d3a0ac09ec9e5eb3a'
            '/raw/238f720ff059c1f82f368259d1ca4ffa5dd8f9f5/imagenet1000_clsidx_to_labels.txt'
        )
        raw = ''.join(f.decode('utf-8') for f in label_url)
        return yaml.load(raw, Loader=yaml.Loader)
    return {}


def _build_model_spec(args) -> dict:
    """Build a model spec dict from CLI args."""
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


def _resolve_input_saliency_method(args) -> str:
    method = str(args.input_saliency_method)
    if method == 'auto':
        return 'direct_loss' if args.task == 'detection' else 'matching'
    return method


def _reference_save_name(args) -> str:
    save_name = (str(args.reference_id) if args.reference_id is not None
                 else args.image_path.split('/')[-1].split('.')[0])
    save_name += '_' + _cache_key(args)
    return save_name


def _prepare_gradient_arrays(grads: torch.Tensor) -> dict:
    raw_grad = grads[0].detach().cpu().numpy()
    abs_map = np.abs(raw_grad).max(axis=0)

    clipped_map = abs_map.copy()
    hi = np.percentile(clipped_map, 99)
    lo = np.percentile(clipped_map, 90)
    clipped_map[clipped_map > hi] = hi
    clipped_map[clipped_map < lo] = lo

    denom = np.max(clipped_map) - np.min(clipped_map)
    if denom <= 1e-12:
        normalized_map = np.zeros_like(clipped_map)
    else:
        normalized_map = (clipped_map - np.min(clipped_map)) / denom

    heatmap_mask = np.ones_like(normalized_map) - normalized_map
    heatmap_mask = cv2.GaussianBlur(heatmap_mask, (3, 3), 0)

    return {
        'raw_grad': raw_grad,
        'abs_map': abs_map,
        'normalized_map': normalized_map,
        'heatmap_mask': heatmap_mask,
        'percentile_low': float(lo),
        'percentile_high': float(hi),
    }


def _save_gradient_visualization(
    heatmap_mask: np.ndarray,
    out_path: str,
    reference_image,
    inv_transform_test,
    detection_overlay=None,
):
    reference_image_to_compare = inv_transform_test(reference_image[0].cpu()).permute(1, 2, 0)
    heatmap_superimposed = show_heatmap_on_image(
        reference_image_to_compare.detach().cpu().numpy(), heatmap_mask
    )
    fig, ax = plt.subplots()
    ax.imshow(heatmap_superimposed)

    if detection_overlay is not None:
        color_map = {'tp': 'lime', 'fp': 'red', 'fn': 'yellow'}
        class_names = detection_overlay.get('class_names', {})
        for key in ('tp', 'fp', 'fn'):
            for item in detection_overlay.get(key, []):
                x1, y1, x2, y2 = item['box']
                cls_id = int(item.get('cls_id', -1))
                cls_name = class_names.get(cls_id, str(cls_id)) if cls_id >= 0 else 'unknown'
                rect = Rectangle(
                    (x1, y1),
                    max(1.0, x2 - x1),
                    max(1.0, y2 - y1),
                    fill=False,
                    edgecolor=color_map[key],
                    linewidth=2.0,
                )
                ax.add_patch(rect)
                ax.text(
                    x1,
                    max(0.0, y1 - 2.0),
                    cls_name,
                    color=color_map[key],
                    fontsize=8,
                    fontweight='bold',
                    ha='left',
                    va='bottom',
                )

    ax.axis('off')
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)


def _summarize_distribution(values: np.ndarray) -> dict:
    if values.size == 0:
        return {
            'count': 0,
            'mean': None,
            'std': None,
            'min': None,
            'max': None,
            'q25': None,
            'median': None,
            'q75': None,
            'q90': None,
            'q95': None,
        }

    values = values.astype(np.float64)
    return {
        'count': int(values.size),
        'mean': float(np.mean(values)),
        'std': float(np.std(values)),
        'min': float(np.min(values)),
        'max': float(np.max(values)),
        'q25': float(np.percentile(values, 25)),
        'median': float(np.percentile(values, 50)),
        'q75': float(np.percentile(values, 75)),
        'q90': float(np.percentile(values, 90)),
        'q95': float(np.percentile(values, 95)),
    }


def _clip_box_to_map(box, map_hw):
    x1, y1, x2, y2 = box
    h, w = map_hw
    x1 = int(max(0, min(w - 1, np.floor(x1))))
    y1 = int(max(0, min(h - 1, np.floor(y1))))
    x2 = int(max(0, min(w, np.ceil(x2))))
    y2 = int(max(0, min(h, np.ceil(y2))))
    return x1, y1, x2, y2


def _build_overlay_group_stats(normalized_map: np.ndarray, detection_overlay: dict) -> dict:
    out = {
        'overlay_counts': {
            'tp': int(len(detection_overlay.get('tp', []))),
            'fp': int(len(detection_overlay.get('fp', []))),
            'fn': int(len(detection_overlay.get('fn', []))),
        },
        'groups': {},
    }

    for key in ('tp', 'fp', 'fn'):
        items = detection_overlay.get(key, [])
        box_means = []
        box_medians = []
        pixel_values = []

        for item in items:
            x1, y1, x2, y2 = _clip_box_to_map(item['box'], normalized_map.shape)
            if x2 <= x1 or y2 <= y1:
                continue

            patch = normalized_map[y1:y2, x1:x2]
            if patch.size == 0:
                continue

            box_means.append(float(np.mean(patch)))
            box_medians.append(float(np.median(patch)))
            pixel_values.append(patch.reshape(-1))

        pixel_values = np.concatenate(pixel_values) if pixel_values else np.array([], dtype=np.float64)
        out['groups'][key] = {
            'box_mean_distribution': _summarize_distribution(np.array(box_means, dtype=np.float64)),
            'box_median_distribution': _summarize_distribution(np.array(box_medians, dtype=np.float64)),
            'pixel_distribution': _summarize_distribution(pixel_values),
        }

    return out


def _save_component_gradient_exports(
    component_gradients: dict,
    component_losses: dict,
    args,
    reference_image,
    inv_transform_test,
    detection_overlay=None,
):
    if not component_gradients:
        return

    save_name = _reference_save_name(args)
    root_dir = os.path.join(args.output_root, 'loss_component_saliency')
    raw_dir = os.path.join(root_dir, 'raw_gradients')
    map_dir = os.path.join(root_dir, 'maps')
    image_dir = os.path.join(root_dir, 'images')
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(map_dir, exist_ok=True)
    os.makedirs(image_dir, exist_ok=True)

    metadata = {
        'image_path': args.image_path,
        'reference_id': args.reference_id,
        'model_class_path': args.model_class_path,
        'objective_provider': args.det_objective_provider,
        'objective_mode': args.det_objective_mode,
        'input_saliency_method': args.input_saliency_method,
        'noise_iters': args.noise_iters,
        'noise_percent': args.noise_percent,
        'aggregation': 'max_abs_over_channels',
        'normalization': 'clip_to_p90_p99_then_minmax',
        'normalized_map_overlay_statistics': {},
        'components': {},
    }

    for component_name, grads in component_gradients.items():
        arrays = _prepare_gradient_arrays(grads)
        np.save(os.path.join(raw_dir, f'{save_name}_{component_name}_raw_grad.npy'), arrays['raw_grad'])
        np.save(os.path.join(map_dir, f'{save_name}_{component_name}_abs_map.npy'), arrays['abs_map'])
        np.save(os.path.join(map_dir, f'{save_name}_{component_name}_normalized_map.npy'), arrays['normalized_map'])

        image_path = os.path.join(image_dir, f'input_saliency_heatmap_{save_name}_{component_name}.png')
        _save_gradient_visualization(
            arrays['heatmap_mask'],
            image_path,
            reference_image,
            inv_transform_test,
            detection_overlay=detection_overlay,
        )

        metadata['components'][component_name] = {
            'loss_value': float(component_losses.get(component_name, float('nan'))),
            'percentile_low': arrays['percentile_low'],
            'percentile_high': arrays['percentile_high'],
            'raw_gradient_path': os.path.join('raw_gradients', f'{save_name}_{component_name}_raw_grad.npy'),
            'abs_map_path': os.path.join('maps', f'{save_name}_{component_name}_abs_map.npy'),
            'normalized_map_path': os.path.join('maps', f'{save_name}_{component_name}_normalized_map.npy'),
            'image_path': os.path.join('images', f'input_saliency_heatmap_{save_name}_{component_name}.png'),
        }

        if detection_overlay is not None:
            metadata['normalized_map_overlay_statistics'][component_name] = _build_overlay_group_stats(
                arrays['normalized_map'], detection_overlay
            )

    metadata_path = os.path.join(root_dir, f'{save_name}_metadata.json')
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f'Loss component saliency saved under {root_dir}')


def _resolve_detection_annotations_path(args) -> str:
    if args.det_annotations_json:
        return args.det_annotations_json

    candidates = [
        os.path.join('raw_images', 'coco2017', 'annotations', 'instances_val2017.json'),
        os.path.join('externals', 'YOLOX', 'datasets', 'COCO', 'annotations', 'instances_val2017.json'),
    ]
    for cand in candidates:
        if os.path.isfile(cand):
            return cand
    return None


def _cxcywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    out = boxes.copy()
    out[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.0
    out[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.0
    out[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.0
    out[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.0
    return out


def _box_iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return float(inter / union)


def _nms_by_class(boxes_xyxy: np.ndarray, classes: np.ndarray, scores: np.ndarray, iou_thr: float) -> np.ndarray:
    if boxes_xyxy.shape[0] == 0:
        return np.array([], dtype=np.int64)

    keep = []
    for cls_id in np.unique(classes):
        cls_inds = np.where(classes == cls_id)[0]
        if cls_inds.size == 0:
            continue
        cls_boxes = torch.from_numpy(boxes_xyxy[cls_inds]).float()
        cls_scores = torch.from_numpy(scores[cls_inds]).float()
        kept_local = torchvision.ops.nms(cls_boxes, cls_scores, iou_thr).cpu().numpy()
        keep.extend(cls_inds[kept_local].tolist())

    keep = np.array(keep, dtype=np.int64)
    keep = keep[np.argsort(scores[keep])[::-1]]
    return keep


def _load_coco_gt_for_image(args, image_hw):
    ann_path = _resolve_detection_annotations_path(args)
    if ann_path is None:
        print('Detection overlay: COCO annotations JSON not found; FN overlay is skipped.')
        return [], [], {}

    if args.image_path is None:
        return [], [], {}

    with open(ann_path, 'r') as f:
        coco = json.load(f)

    basename = os.path.basename(args.image_path)
    image_info = None
    for img in coco.get('images', []):
        if img.get('file_name') == basename:
            image_info = img
            break

    if image_info is None:
        print(f'Detection overlay: no GT entry found for image {basename}; FN overlay is skipped.')
        return [], [], {}

    sorted_categories = sorted(coco.get('categories', []), key=lambda cat: cat['id'])
    cat_id_to_cls = {cat['id']: idx for idx, cat in enumerate(sorted_categories)}
    cls_to_name = {idx: cat.get('name', str(cat['id'])) for idx, cat in enumerate(sorted_categories)}

    raw_w = float(image_info['width'])
    raw_h = float(image_info['height'])
    h, w = image_hw
    preprocess_cfg = _get_preprocess_cfg(args)
    use_letterbox = bool(preprocess_cfg.get('letterbox', False))

    if use_letterbox:
        r = min(float(w) / raw_w, float(h) / raw_h)
        sx = r
        sy = r
    else:
        sx = float(w) / raw_w
        sy = float(h) / raw_h

    gt_boxes = []
    gt_classes = []
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
        x1 = x * sx
        y1 = y * sy
        x2 = (x + bw) * sx
        y2 = (y + bh) * sy
        gt_boxes.append([x1, y1, x2, y2])
        gt_classes.append(cat_id_to_cls[cat_id])

    return gt_boxes, gt_classes, cls_to_name


def _xyxy_to_cxcywh(boxes_xyxy: np.ndarray) -> np.ndarray:
    out = np.zeros_like(boxes_xyxy)
    out[:, 0] = (boxes_xyxy[:, 0] + boxes_xyxy[:, 2]) / 2.0
    out[:, 1] = (boxes_xyxy[:, 1] + boxes_xyxy[:, 3]) / 2.0
    out[:, 2] = np.clip(boxes_xyxy[:, 2] - boxes_xyxy[:, 0], a_min=0.0, a_max=None)
    out[:, 3] = np.clip(boxes_xyxy[:, 3] - boxes_xyxy[:, 1], a_min=0.0, a_max=None)
    return out


def _build_detection_objective_context(args, reference_image):
    if args.task != 'detection':
        return None

    context = {
        'det_objective_mode': args.det_objective_mode,
        'det_objective_provider': args.det_objective_provider,
        'det_provider_strict': args.det_provider_strict,
        'det_iou_weight': args.det_iou_weight,
        'det_score_weight': args.det_score_weight,
    }

    if args.det_objective_mode == 'legacy_single_class':
        return context

    if args.reference_id is not None:
        raise NotImplementedError(
            'Detection objective modes based on GT instances currently require --image_path; '
            '--reference_id is not supported yet.'
        )

    h = int(reference_image.shape[-2])
    w = int(reference_image.shape[-1])
    gt_boxes, gt_classes, _ = _load_coco_gt_for_image(args, image_hw=(h, w))
    if len(gt_boxes) == 0:
        raise ValueError(
            'No COCO GT instances were found for the selected image. '
            'Provide --det_annotations_json and ensure --image_path matches file_name in annotations.'
        )

    boxes_xyxy = np.array(gt_boxes, dtype=np.float32)
    boxes_cxcywh = _xyxy_to_cxcywh(boxes_xyxy)
    context['det_gt_instances'] = [
        {
            'class_ids': torch.tensor(gt_classes, dtype=torch.long),
            'boxes_xyxy': torch.from_numpy(boxes_xyxy),
            'boxes_cxcywh': torch.from_numpy(boxes_cxcywh),
        }
    ]
    return context


def _build_detection_overlay(reference_image, net, args):
    if args.task != 'detection' or args.reference_id is not None:
        return None

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    with torch.no_grad():
        outputs = net(reference_image.to(device))

    if outputs.ndim != 3 or outputs.size(0) == 0 or outputs.size(-1) < 6:
        print('Detection overlay: unexpected model output shape; overlay is skipped.')
        return None

    pred = outputs[0].detach().cpu().numpy()
    boxes_xyxy = _cxcywh_to_xyxy(pred[:, :4])
    obj = pred[:, 4]
    cls_scores = pred[:, 5:]
    cls_ids = np.argmax(cls_scores, axis=1)
    cls_conf = cls_scores[np.arange(cls_scores.shape[0]), cls_ids]
    scores = obj * cls_conf

    conf_mask = scores >= float(args.det_conf_threshold)
    boxes_xyxy = boxes_xyxy[conf_mask]
    cls_ids = cls_ids[conf_mask]
    scores = scores[conf_mask]

    if boxes_xyxy.shape[0] == 0:
        return {'tp': [], 'fp': [], 'fn': [], 'class_names': {}}

    keep = _nms_by_class(boxes_xyxy, cls_ids, scores, float(args.det_nms_iou_threshold))
    boxes_xyxy = boxes_xyxy[keep]
    cls_ids = cls_ids[keep]
    scores = scores[keep]

    h = int(reference_image.shape[-2])
    w = int(reference_image.shape[-1])
    boxes_xyxy[:, [0, 2]] = np.clip(boxes_xyxy[:, [0, 2]], 0, w - 1)
    boxes_xyxy[:, [1, 3]] = np.clip(boxes_xyxy[:, [1, 3]], 0, h - 1)

    gt_boxes, gt_classes, class_names = _load_coco_gt_for_image(args, image_hw=(h, w))
    if len(gt_boxes) == 0:
        # GT が無い場合は全予測をFPとして描画して注意喚起する。
        return {
            'tp': [],
            'fp': [
                {'box': boxes_xyxy[i].tolist(), 'cls_id': int(cls_ids[i])}
                for i in range(boxes_xyxy.shape[0])
            ],
            'fn': [],
            'class_names': class_names,
        }

    gt_boxes = np.array(gt_boxes, dtype=np.float32)
    gt_classes = np.array(gt_classes, dtype=np.int64)

    pred_order = np.argsort(scores)[::-1]
    matched_gt = np.zeros(len(gt_boxes), dtype=bool)
    tp_boxes = []
    fp_boxes = []

    iou_thr = float(args.det_match_iou_threshold)
    for pidx in pred_order:
        pbox = boxes_xyxy[pidx]
        pcls = cls_ids[pidx]

        candidate = np.where((gt_classes == pcls) & (~matched_gt))[0]
        if candidate.size == 0:
            fp_boxes.append({'box': pbox.tolist(), 'cls_id': int(pcls)})
            continue

        ious = np.array([_box_iou_xyxy(pbox, gt_boxes[g]) for g in candidate], dtype=np.float32)
        best_local = int(np.argmax(ious))
        best_iou = float(ious[best_local])
        best_gt = int(candidate[best_local])

        if best_iou >= iou_thr:
            matched_gt[best_gt] = True
            tp_boxes.append({'box': pbox.tolist(), 'cls_id': int(pcls)})
        else:
            fp_boxes.append({'box': pbox.tolist(), 'cls_id': int(pcls)})

    fn_indices = np.where(~matched_gt)[0]
    fn_boxes = [
        {'box': gt_boxes[g].tolist(), 'cls_id': int(gt_classes[g])}
        for g in fn_indices
    ]
    return {'tp': tp_boxes, 'fp': fp_boxes, 'fn': fn_boxes, 'class_names': class_names}


def _reference_output_key(args) -> str:
    return str(args.reference_id) if args.reference_id is not None else os.path.splitext(os.path.basename(args.image_path))[0]


def _reference_file_name(args, testset) -> str:
    if args.reference_id is None:
        return os.path.basename(args.image_path)

    if testset is not None:
        samples = getattr(testset, 'samples', None)
        if samples is not None and 0 <= int(args.reference_id) < len(samples):
            sample_path = samples[int(args.reference_id)][0]
            return os.path.basename(sample_path)

    return _reference_output_key(args)


def _register_feature_hooks(model, layer_mapping):
    feature_cache = {}
    handles = []
    module_dict = dict(model.named_modules())

    missing = [module_name for module_name in layer_mapping.values() if module_name not in module_dict]
    if missing:
        raise KeyError(
            'Failed to find feature layers in model.named_modules(): {}'.format(', '.join(missing))
        )

    for logical_name, module_name in layer_mapping.items():
        module = module_dict[module_name]

        def _hook(_, __, output, logical_name_=logical_name, module_name_=module_name):
            tensor = output[0] if isinstance(output, (list, tuple)) else output
            if tensor is None or not hasattr(tensor, 'detach'):
                return
            feature_cache[logical_name_] = {
                'module_name': module_name_,
                'tensor': tensor.detach().cpu(),
            }

        handles.append(module.register_forward_hook(_hook))

    return handles, feature_cache


def _save_feature_arrays(
    per_image_dir,
    image_id_key,
    file_name,
    feature_cache,
    model_input_shape,
):
    npy_dir = os.path.join(per_image_dir, 'npy')
    os.makedirs(npy_dir, exist_ok=True)

    manifest = {
        'image_id': image_id_key,
        'file_name': file_name,
        'model_input_shape': list(model_input_shape),
        'layers': [],
    }

    for logical_name in sorted(feature_cache.keys()):
        item = feature_cache[logical_name]
        tensor = item['tensor']
        if tensor.ndim == 4 and tensor.shape[0] == 1:
            tensor = tensor[0]

        array = tensor.float().numpy()
        file_base = 'feat_{}.npy'.format(logical_name)
        file_path = os.path.join(npy_dir, file_base)
        np.save(file_path, array)

        manifest['layers'].append(
            {
                'logical_name': logical_name,
                'module_name': item['module_name'],
                'tensor_path': os.path.join('npy', file_base),
                'shape': list(array.shape),
                'dtype': str(array.dtype),
            }
        )

    manifest_path = os.path.join(per_image_dir, 'feature_manifest.json')
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


def _export_intermediate_features(reference_image, net, adapter, args, testset):
    export_model = net.module if isinstance(net, torch.nn.DataParallel) else net
    layer_mapping = adapter.get_feature_export_layers(export_model)
    if not layer_mapping:
        print('Intermediate feature export skipped: no eligible layers were found.')
        return

    image_id_key = _reference_output_key(args)
    per_image_dir = os.path.join(args.output_root, image_id_key)
    file_name = _reference_file_name(args, testset)

    hook_handles, feature_cache = _register_feature_hooks(export_model, layer_mapping)
    try:
        with torch.no_grad():
            export_model(reference_image.to(next(export_model.parameters()).device))
    finally:
        for handle in hook_handles:
            handle.remove()

    _save_feature_arrays(
        per_image_dir=per_image_dir,
        image_id_key=image_id_key,
        file_name=file_name,
        feature_cache=feature_cache,
        model_input_shape=tuple(reference_image.shape),
    )
    print(f'Intermediate features saved to {per_image_dir}')


def _export_model_checkpoint(model, args) -> None:
    """Export the loaded model as a checkpoint compatible with CustomModuleAdapter."""
    if not args.export_model_pth:
        return

    export_dir = os.path.dirname(args.export_model_pth)
    if export_dir:
        os.makedirs(export_dir, exist_ok=True)

    checkpoint = {
        'state_dict': model.state_dict(),
        'model_source': args.model_source,
        'model_spec': _build_model_spec(args),
    }
    if args.model_source == 'custom_module':
        checkpoint['model_class_path'] = args.model_class_path

    torch.save(checkpoint, args.export_model_pth)
    print(f'Exported loaded model checkpoint to {args.export_model_pth}')


def save_gradients(grads_to_save, args, reference_image, inv_transform_test, detection_overlay=None):
    arrays = _prepare_gradient_arrays(grads_to_save)
    save_path = os.path.join(args.output_root, args.figure_folder_name)
    os.makedirs(save_path, exist_ok=True)
    save_name = _reference_save_name(args)
    plt.axis('off')
    out_path = os.path.join(save_path, f'input_saliency_heatmap_{save_name}.png')
    _save_gradient_visualization(
        arrays['heatmap_mask'],
        out_path,
        reference_image,
        inv_transform_test,
        detection_overlay=detection_overlay,
    )
    print(f'Input space saliency saved to {out_path}\n')
    return


def compute_detection_component_gradients(
    reference_inputs,
    reference_targets,
    net,
    args,
    task_adapter,
    target_spec,
    objective_context=None,
):
    if args.task != 'detection' or not hasattr(task_adapter, 'build_objective_components'):
        return {}, {}

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    reference_inputs = reference_inputs.to(device)
    reference_targets = reference_targets.to(device)

    component_samples = {}
    component_losses = {}
    for _ in range(args.noise_iters):
        perturbed_inputs = reference_inputs.detach().clone()
        perturbed_inputs = (
            (1 - args.noise_percent) * perturbed_inputs
            + args.noise_percent * torch.randn_like(perturbed_inputs)
        )
        perturbed_inputs.requires_grad_()

        components = task_adapter.build_objective_components(
            net,
            perturbed_inputs,
            reference_targets,
            target_spec,
            objective_context=objective_context,
        )
        for component_name, loss_tensor in components.items():
            net.zero_grad()
            if perturbed_inputs.grad is not None:
                perturbed_inputs.grad.zero_()
            loss_tensor.backward(retain_graph=(component_name != list(components.keys())[-1]))
            component_samples.setdefault(component_name, []).append(perturbed_inputs.grad.detach().cpu().clone())
            component_losses[component_name] = float(loss_tensor.detach().cpu().item())

    averaged = {
        component_name: torch.stack(samples).mean(0)
        for component_name, samples in component_samples.items()
    }
    return averaged, component_losses

def compute_input_space_saliency(
    reference_inputs, reference_targets, net, args,
    task_adapter, target_spec,
    testset_mean_stat=None, testset_std_stat=None,
    inv_transform_test=None, readable_labels=None,
    objective_context=None,
):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    input_saliency_method = _resolve_input_saliency_method(args)
    saliency_mode = 'std'
    if testset_mean_stat is None or testset_std_stat is None:
        saliency_mode = 'naive'
        print('Testset saliency statistics are unavailable; falling back to naive saliency mode.')

    # Log prediction summary via TaskAdapter
    task_adapter.summarize_prediction(
        net,
        reference_inputs.to(device),
        reference_targets.to(device),
        readable_labels or {},
    )

    filter_saliency_model = SaliencyModel(
        net, task_adapter,
        device=device, mode=saliency_mode,
        aggregation='filter_wise', signed=args.signed,
    )
    reference_inputs  = reference_inputs.to(device)
    reference_targets = reference_targets.to(device)

    grad_samples = []
    for _ in range(args.noise_iters):
        perturbed_inputs = reference_inputs.detach().clone()
        perturbed_inputs = (
            (1 - args.noise_percent) * perturbed_inputs
            + args.noise_percent * torch.randn_like(perturbed_inputs)
        )
        perturbed_inputs.requires_grad_()

        if input_saliency_method == 'direct_loss':
            net.zero_grad()
            loss, _ = task_adapter.build_objective(
                net,
                perturbed_inputs,
                reference_targets,
                target_spec,
                objective_context=objective_context,
            )
            loss.backward()
        else:
            filter_saliency = filter_saliency_model(
                perturbed_inputs, reference_targets, target_spec,
                testset_mean_abs_grad=testset_mean_stat,
                testset_std_abs_grad=testset_std_stat,
                objective_context=objective_context,
            ).to(device)

            if args.compare_random:
                sorted_filters = torch.randperm(filter_saliency.size(0)).cpu().numpy()
            else:
                sorted_filters = torch.argsort(filter_saliency, descending=True).cpu().numpy()

            filter_saliency_boosted = filter_saliency.detach().clone()
            filter_saliency_boosted[sorted_filters[:args.k_salient]] *= args.boost_factor

            matching_criterion = torch.nn.CosineSimilarity()
            matching_loss = matching_criterion(
                filter_saliency[None, :], filter_saliency_boosted[None, :]
            )
            matching_loss.backward()

        grad_samples.append(perturbed_inputs.grad.detach().cpu())

    # Keep the filter saliency profile output for downstream plotting.
    final_inputs = reference_inputs.detach().clone().requires_grad_(True)
    filter_saliency = filter_saliency_model(
        final_inputs, reference_targets, target_spec,
        testset_mean_abs_grad=testset_mean_stat,
        testset_std_abs_grad=testset_std_stat,
        objective_context=objective_context,
    ).to(device)

    grads_to_save = torch.stack(grad_samples).mean(0)
    return grads_to_save, filter_saliency


def sort_filters_layer_wise(filter_profile, layer_to_filter_id, filter_std = None):
    layer_sorted_profile = []
    means = []
    stds = []
    for layer in layer_to_filter_id:
        layer_inds = layer_to_filter_id[layer]
        layer_sorted_profile.append(np.sort(filter_profile[layer_inds])[::-1])
        means.append(np.ones_like(filter_profile[layer_inds])*np.mean(filter_profile[layer_inds]))
        if filter_std is not None:
            stds.append(filter_std[layer_inds][np.argsort(filter_profile[layer_inds])[::-1]])
    layer_sorted_profile = np.concatenate(layer_sorted_profile)
    sal_means = np.concatenate(means)
    if filter_std is not None:
        sal_stds = np.concatenate(stds)
        return layer_sorted_profile, sal_means, sal_stds
    else:
        return layer_sorted_profile, sal_means

if __name__ == '__main__':

    torch.manual_seed(40)
    np.random.seed(40)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(40)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args   = parser.parse_args()

    if args.logit or args.logit_difference:
        raise NotImplementedError('--logit and --logit_difference are not yet implemented.')

    os.makedirs(args.output_root, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Model                                                                #
    # ------------------------------------------------------------------ #
    print('==> Building model..')
    adapter = build_model_adapter(_build_model_spec(args))
    net     = adapter.build_model()
    _export_model_checkpoint(net, args)

    # Saliency units: derived BEFORE DataParallel to keep layer names clean
    layer_to_filter_id = adapter.iter_saliency_units(net)
    total_filters = sum(len(v) for v in layer_to_filter_id.values())
    print(f'Total filters: {total_filters}')
    print(f'Total layers:  {len(layer_to_filter_id)}')

    transform_test     = adapter.get_preprocess()
    inv_transform_test = adapter.get_inv_preprocess()

    net = net.to(device)
    net.eval()
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark    = False
        cudnn.deterministic = True

    # ------------------------------------------------------------------ #
    # Task & target                                                        #
    # ------------------------------------------------------------------ #
    if args.task == 'classification':
        task_adapter = ClassificationTaskAdapter()
    else:
        task_adapter = DetectionTaskAdapter(
            objective_mode=args.det_objective_mode,
            objective_provider=args.det_objective_provider,
            provider_strict=args.det_provider_strict,
        )
    target_spec  = TargetSpec.from_args(args.target_type, args.target_class_id)

    # ------------------------------------------------------------------ #
    # Dataset                                                              #
    # ------------------------------------------------------------------ #
    print('==> Preparing data..')
    testset = None
    if args.data_to_use == 'ImageNet':
        images_path = args.imagenet_val_path
    else:
        raise NotImplementedError(f'data_to_use={args.data_to_use!r} is not supported.')

    if images_path != '<insert-ImageNet-val-path-here>':
        testset = torchvision.datasets.ImageFolder(images_path, transform=transform_test)
    else:
        print(
            '\n  ImageNet validation set path is not specified.\n'
            '  The code will only work with --image_path and --image_target_label.\n'
            '  --reference_id requires the validation set path.\n'
        )

    # ------------------------------------------------------------------ #
    # Label map                                                            #
    # ------------------------------------------------------------------ #
    readable_labels = _load_label_map(args)

    # ------------------------------------------------------------------ #
    # Cache paths                                                          #
    # ------------------------------------------------------------------ #
    ck = _cache_key(args)
    model_helpers_root_path = os.path.join('helper_objects', ck)
    os.makedirs(model_helpers_root_path, exist_ok=True)

    # Include target_type in stats filename so different targets use separate caches
    target_suffix = '' if args.target_type == 'true_label' else f'_{args.target_type}'

    # ------------------------------------------------------------------ #
    # Inference cache                                                      #
    # ------------------------------------------------------------------ #
    inference_file = os.path.join(
        model_helpers_root_path,
        f'ImageNet_val_inference_results_{ck}.pth',
    )
    if os.path.isfile(inference_file):
        inf_results           = torch.load(inference_file)
        incorrect_id          = inf_results['incorrect_id']
        incorrect_predictions = inf_results['incorrect_predictions']
        correct_id            = inf_results['correct_id']
    elif testset is not None:
        warnings.warn('Computing inference; check cache filenames if unintended.')
        incorrect_id, incorrect_predictions, correct_id = \
            test_and_find_incorrectly_classified(net, testset)
        torch.save(
            {'incorrect_id': incorrect_id,
             'incorrect_predictions': incorrect_predictions,
             'correct_id': correct_id},
            inference_file,
        )

    # ------------------------------------------------------------------ #
    # Testset saliency statistics cache                                    #
    # ------------------------------------------------------------------ #
    filter_stats_file = os.path.join(
        model_helpers_root_path,
        f'ImageNet_val_saliency_stat_{ck}{target_suffix}_filter_wise.pth',
    )
    if args.task == 'detection' and args.det_objective_mode != 'legacy_single_class':
        filter_testset_mean_abs_grad = None
        filter_testset_std_abs_grad  = None
        print('Skipping testset saliency statistics for detection objective mode requiring per-image GT context.')
    elif os.path.isfile(filter_stats_file):
        filter_stats                 = torch.load(filter_stats_file)
        filter_testset_mean_abs_grad = filter_stats['mean']
        filter_testset_std_abs_grad  = filter_stats['std']
    elif testset is not None:
        warnings.warn('Computing testset stats; check cache filenames if unintended.')
        filter_testset_mean_abs_grad, filter_testset_std_abs_grad = find_testset_saliency(
            net, testset, 'filter_wise', task_adapter, target_spec, signed=args.signed,
        )
        torch.save(
            {'mean': filter_testset_mean_abs_grad, 'std': filter_testset_std_abs_grad},
            filter_stats_file,
        )
    else:
        filter_testset_mean_abs_grad = None
        filter_testset_std_abs_grad  = None

    # ------------------------------------------------------------------ #
    # Reference image                                                      #
    # ------------------------------------------------------------------ #
    if args.reference_id is None:
        print(f'\n  Using image {args.image_path} with target label {args.image_target_label}\n')
        reference_image  = transform_raw_image(
            args.image_path, preprocess=transform_test
        ).unsqueeze(0)
        if args.image_target_label is None:
            if args.target_type == TargetType.SPECIFIED_CLASS.value:
                fallback_target = int(args.target_class_id)
            else:
                fallback_target = 0
                print('No --image_target_label was provided; using 0 as fallback target label.')
        else:
            fallback_target = int(args.image_target_label)
        reference_target = torch.tensor(fallback_target).unsqueeze(0)
    else:
        print(f'\n  Using {args.reference_id}-th image from the validation set.\n')
        reference_image, reference_target = testset.__getitem__(args.reference_id)
        reference_target = torch.tensor(reference_target).unsqueeze(0)
        reference_image.unsqueeze_(0)

    objective_context = _build_detection_objective_context(args, reference_image)

    # ------------------------------------------------------------------ #
    # Compute saliency                                                     #
    # ------------------------------------------------------------------ #
    grads_to_save, filter_saliency = compute_input_space_saliency(
        reference_image, reference_target, net, args,
        task_adapter, target_spec,
        filter_testset_mean_abs_grad, filter_testset_std_abs_grad,
        inv_transform_test, readable_labels,
        objective_context=objective_context,
    )
    component_gradients, component_losses = compute_detection_component_gradients(
        reference_image,
        reference_target,
        net,
        args,
        task_adapter,
        target_spec,
        objective_context=objective_context,
    )

    layer_sorted_profile, _ = sort_filters_layer_wise(
        filter_saliency.detach().cpu().numpy(), layer_to_filter_id,
    )

    # ------------------------------------------------------------------ #
    # Save results                                                         #
    # ------------------------------------------------------------------ #
    input_tensor_path = os.path.join(args.output_root, 'input_tensor.npy')
    np.save(input_tensor_path, reference_image.cpu().numpy()[0, :3])  # Save the original input tensor (without gradients) for reference
    print(f'Input tensor saved to {input_tensor_path}')

    detection_overlay = _build_detection_overlay(reference_image, net, args)
    _export_intermediate_features(reference_image, net, adapter, args, testset)
    save_gradients(
        grads_to_save,
        args,
        reference_image,
        inv_transform_test,
        detection_overlay=detection_overlay,
    )
    _save_component_gradient_exports(
        component_gradients,
        component_losses,
        args,
        reference_image,
        inv_transform_test,
        detection_overlay=detection_overlay,
    )

    fig, ax = plt.subplots(1, 1, figsize=(15, 4))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    pal = sns.color_palette('colorblind')
    ax.plot(layer_sorted_profile, label='Sorted filter saliency', c=pal.as_hex()[0])
    ax.legend()
    ax.get_legend().get_frame().set_alpha(0.0)
    ax.set_xlabel('Filter ID')
    ax.set_ylabel('Saliency')

    save_name = (str(args.reference_id) if args.reference_id is not None
                 else args.image_path.split('/')[-1].split('.')[0])
    save_name += '_' + ck
    out_path = os.path.join(args.output_root, f'filter_saliency_{save_name}.png')
    fig.savefig(out_path)
    print(f'Filter saliency saved to {out_path}')
#Run this: python3 parameter_and_input_saliency.py --model resnet50 --image_path raw_images/great_white_shark_mispred_as_killer_whale.jpeg --image_target_label 2
