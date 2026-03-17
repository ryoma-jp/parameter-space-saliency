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

def _cache_key(args) -> str:
    """Derive a filesystem-safe cache key from model arguments."""
    if args.model_source == 'torchvision':
        return args.model
    return args.model_class_path.split('.')[-1]


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
    grads_to_save, _ = grads_to_save.max(dim=1)
    grads_to_save = grads_to_save[0].detach().cpu().numpy()
    grads_to_save = np.abs(grads_to_save)
    # grads_to_save[grads_to_save < 0] = 0.0

    #Percentile thresholding
    grads_to_save[grads_to_save > np.percentile(grads_to_save, 99)] = np.percentile(grads_to_save, 99)
    grads_to_save[grads_to_save < np.percentile(grads_to_save, 90)] = np.percentile(grads_to_save, 90)

    save_path = os.path.join(args.output_root, args.figure_folder_name)
    os.makedirs(save_path, exist_ok=True)
    ck = _cache_key(args)
    save_name = (str(args.reference_id) if args.reference_id is not None
                 else args.image_path.split('/')[-1].split('.')[0])
    save_name += '_' + ck
    plt.axis('off')

    grads_to_save = (grads_to_save - np.min(grads_to_save)) / (np.max(grads_to_save) - np.min(grads_to_save))

    reference_image_to_compare = inv_transform_test(reference_image[0].cpu()).permute(1, 2, 0)
    gradients_heatmap = np.ones_like(grads_to_save) - grads_to_save
    gradients_heatmap = cv2.GaussianBlur(gradients_heatmap, (3, 3), 0)

    heatmap_superimposed = show_heatmap_on_image(
        reference_image_to_compare.detach().cpu().numpy(), gradients_heatmap
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
    out_path = os.path.join(save_path, f'input_saliency_heatmap_{save_name}.png')
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f'Input space saliency saved to {out_path}\n')
    return

def compute_input_space_saliency(
    reference_inputs, reference_targets, net, args,
    task_adapter, target_spec,
    testset_mean_stat=None, testset_std_stat=None,
    inv_transform_test=None, readable_labels=None,
):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
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

        filter_saliency = filter_saliency_model(
            perturbed_inputs, reference_targets, target_spec,
            testset_mean_abs_grad=testset_mean_stat,
            testset_std_abs_grad=testset_std_stat,
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
        task_adapter = DetectionTaskAdapter()
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
    if os.path.isfile(filter_stats_file):
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

    # ------------------------------------------------------------------ #
    # Compute saliency                                                     #
    # ------------------------------------------------------------------ #
    grads_to_save, filter_saliency = compute_input_space_saliency(
        reference_image, reference_target, net, args,
        task_adapter, target_spec,
        filter_testset_mean_abs_grad, filter_testset_std_abs_grad,
        inv_transform_test, readable_labels,
    )

    layer_sorted_profile, _ = sort_filters_layer_wise(
        filter_saliency.detach().cpu().numpy(), layer_to_filter_id,
    )

    # ------------------------------------------------------------------ #
    # Save results                                                         #
    # ------------------------------------------------------------------ #
    detection_overlay = _build_detection_overlay(reference_image, net, args)
    save_gradients(
        grads_to_save,
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
