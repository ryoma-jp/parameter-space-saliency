#!/usr/bin/env python3
"""Generate a per-object feature catalog from detection overlay catalogs.

Reads each ``detection_overlay_catalog.json`` produced by
``parameter_and_input_saliency.py``, crops the corresponding original image
at each box region (in original-image space), computes image features, and
writes a flat ``object_catalog.csv`` (and optionally ``object_catalog.json``).

Usage (inside Docker):
    docker compose run --rm -u $(id -u):$(id -g) \\
        -e HOME=/work \\
        -e PYTHONPATH=/work/externals/YOLOX:/work \\
        pss \\
        python3 tools/detection_feature_analysis/run_catalog.py \\
            --results_root results/yolox_tiny_custom_model_auto \\
            --image_dir raw_images/coco2017/val2017 \\
            --output_csv results/yolox_tiny_custom_model_auto/object_catalog.csv

Output CSV columns
------------------
Identification:
  image_id              – COCO image ID string (e.g. "000000000139")
  result_type           – tp / fp_cls / fp_loc / fn
  cls_id                – Predicted class ID (for fn: GT class ID)
  class_name            – Human-readable class name
  matched_gt_cls_id     – (fp_cls only) GT class ID of the IoU-matched GT box
  matched_gt_class_name – (fp_cls only) GT class name
  score                 – objectness × class_score; NaN for fn

Box geometry (original-image space, pixels):
  box_orig_x1/y1/x2/y2 – Box corners in original-image space.
                         Used as the crop region for feature extraction.
  box_input_x1/y1/x2/y2 – Box corners in model-input space (e.g. 416×416).

Derived geometry:
  box_area_orig_px      – Box area in original pixels² = (x2-x1)*(y2-y1)
  box_relative_area     – box_area / (orig_H * orig_W); 0–1
  box_aspect_ratio      – width / height
  box_width_ratio       – box width / image width
  box_height_ratio      – box height / image height

Image features (computed from the original-image crop):
  luminance_mean/std/median – Greyscale (Rec.601) pixel statistics
  rms_contrast          – sqrt(E[I²] - E[I]²)
  saturation_mean/std   – HSV S-channel (0–255)
  hue_std               – Circular std of HSV H-channel (radians)
  colorfulness          – Hasler & Süsstrunk (2003)
  noise_sigma           – MAD/0.6745 of Laplacian residual
  total_variation       – Anisotropic L1-TV / pixel count
  sharpness_laplacian   – Variance of Laplacian response
  edge_density          – Fraction of Canny edge pixels

Overlap:
  max_gt_iou            – Max IoU between this box and all GT boxes
                         (both in original-image space).
                         Indicates occlusion / crowd context.
"""

import argparse
import csv
import json
import math
import os
import sys

import cv2
import numpy as np

# Ensure sibling module is importable regardless of working directory.
sys.path.insert(0, os.path.dirname(__file__))
from image_features import compute_image_features  # noqa: E402


RESULT_TYPES = ('tp', 'fp_cls', 'fp_loc', 'fn')

CSV_COLUMNS = [
    # --- identification ---
    'image_id',
    'result_type',
    'cls_id',
    'class_name',
    'matched_gt_cls_id',
    'matched_gt_class_name',
    'score',
    # --- box: original-image space ---
    'box_orig_x1',
    'box_orig_y1',
    'box_orig_x2',
    'box_orig_y2',
    # --- box: model-input space ---
    'box_input_x1',
    'box_input_y1',
    'box_input_x2',
    'box_input_y2',
    # --- derived geometry ---
    'box_area_orig_px',
    'box_relative_area',
    'box_aspect_ratio',
    'box_width_ratio',
    'box_height_ratio',
    # --- image features ---
    'luminance_mean',
    'luminance_std',
    'luminance_median',
    'rms_contrast',
    'saturation_mean',
    'saturation_std',
    'hue_std',
    'colorfulness',
    'noise_sigma',
    'total_variation',
    'sharpness_laplacian',
    'edge_density',
    # --- overlap ---
    'max_gt_iou',
]


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _box_iou_xyxy(a, b) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def _max_gt_iou(box_orig, gt_orig_boxes) -> float:
    """Return the maximum IoU between *box_orig* and all GT boxes."""
    if not gt_orig_boxes:
        return float('nan')
    return max(_box_iou_xyxy(box_orig, g) for g in gt_orig_boxes)


def _clip_box_to_image(box, orig_hw):
    """Clip box to image bounds and return integer pixel coordinates."""
    h, w = orig_hw
    x1 = int(max(0, min(math.floor(box[0]), w - 1)))
    y1 = int(max(0, min(math.floor(box[1]), h - 1)))
    x2 = int(max(0, min(math.ceil(box[2]), w)))
    y2 = int(max(0, min(math.ceil(box[3]), h)))
    return x1, y1, x2, y2


# ---------------------------------------------------------------------------
# Per-catalog processing
# ---------------------------------------------------------------------------

def process_catalog(catalog_path: str, image_dir: str) -> list:
    """Process one ``detection_overlay_catalog.json`` and return a list of rows.

    Each row is a dict ready to be written as a CSV record.
    """
    with open(catalog_path, encoding='utf-8') as f:
        catalog = json.load(f)

    image_id = catalog['image_id']
    file_name = catalog['file_name']
    class_names = catalog.get('class_names', {})    # {str(int): name}
    original_hw = catalog['original_hw']             # [H, W]
    orig_h, orig_w = int(original_hw[0]), int(original_hw[1])

    image_path = os.path.join(image_dir, file_name)
    if not os.path.isfile(image_path):
        print(f'    WARNING: image not found: {image_path}; skipping.')
        return []

    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print(f'    WARNING: failed to read image: {image_path}; skipping.')
        return []

    # Collect GT boxes in original-image space for max_gt_iou computation.
    gt_orig_boxes = [item['box_orig'] for item in catalog.get('gt', [])]

    rows = []
    for result_type in RESULT_TYPES:
        for item in catalog.get(result_type, []):
            box_orig = item['box_orig']    # [x1, y1, x2, y2] in original-image space
            box_input = item['box_input']  # [x1, y1, x2, y2] in model-input space
            cls_id = int(item['cls_id'])
            class_name = class_names.get(str(cls_id), str(cls_id))

            raw_score = item.get('score')
            score = float(raw_score) if raw_score is not None else float('nan')

            raw_matched = item.get('matched_gt_cls_id')
            if raw_matched is not None:
                matched_gt_cls_id = int(raw_matched)
                matched_gt_class_name = class_names.get(str(matched_gt_cls_id), str(matched_gt_cls_id))
            else:
                matched_gt_cls_id = float('nan')
                matched_gt_class_name = float('nan')

            # ---- geometry ------------------------------------------------
            x1, y1, x2, y2 = [float(v) for v in box_orig]
            bw = max(0.0, x2 - x1)
            bh = max(0.0, y2 - y1)
            box_area = bw * bh
            image_area = float(orig_h * orig_w)

            # ---- image features from original-image crop -----------------
            cx1, cy1, cx2, cy2 = _clip_box_to_image(box_orig, (orig_h, orig_w))
            if cx2 > cx1 and cy2 > cy1:
                crop = img_bgr[cy1:cy2, cx1:cx2]
                feats = compute_image_features(crop)
            else:
                feats = compute_image_features(None)

            row = {
                # identification
                'image_id':               image_id,
                'result_type':            result_type,
                'cls_id':                 cls_id,
                'class_name':             class_name,
                'matched_gt_cls_id':      matched_gt_cls_id,
                'matched_gt_class_name':  matched_gt_class_name,
                'score':                  score,
                # box: original-image space
                'box_orig_x1':            round(x1, 3),
                'box_orig_y1':            round(y1, 3),
                'box_orig_x2':            round(x2, 3),
                'box_orig_y2':            round(y2, 3),
                # box: model-input space
                'box_input_x1':           round(float(box_input[0]), 3),
                'box_input_y1':           round(float(box_input[1]), 3),
                'box_input_x2':           round(float(box_input[2]), 3),
                'box_input_y2':           round(float(box_input[3]), 3),
                # derived geometry
                'box_area_orig_px':       round(box_area, 2),
                'box_relative_area':      round(box_area / image_area, 6) if image_area > 0 else float('nan'),
                'box_aspect_ratio':       round(bw / bh, 4) if bh > 0 else float('nan'),
                'box_width_ratio':        round(bw / orig_w, 6) if orig_w > 0 else float('nan'),
                'box_height_ratio':       round(bh / orig_h, 6) if orig_h > 0 else float('nan'),
                # image features
                **feats,
                # overlap
                'max_gt_iou': (
                    round(_max_gt_iou(box_orig, gt_orig_boxes), 4)
                    if not math.isnan(_max_gt_iou(box_orig, gt_orig_boxes))
                    else float('nan')
                ),
            }
            rows.append(row)

    return rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description=(
            'Build a per-object feature catalog from detection_overlay_catalog.json files.'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument(
        '--results_root', required=True,
        help='Root directory containing per-image result subdirectories '
             '(each with a detection_overlay_catalog.json).',
    )
    ap.add_argument(
        '--image_dir', required=True,
        help='Directory containing original image files (e.g. val2017/).',
    )
    ap.add_argument(
        '--output_csv', required=True,
        help='Output CSV file path.',
    )
    ap.add_argument(
        '--output_json', default=None,
        help='Optional output JSON file path (array of row dicts).',
    )
    args = ap.parse_args()

    entries = sorted(
        e for e in os.listdir(args.results_root)
        if os.path.isdir(os.path.join(args.results_root, e))
    )
    print(f'Scanning {len(entries)} directories under {args.results_root} ...')

    all_rows = []
    n_catalogs = 0
    for entry in entries:
        catalog_path = os.path.join(args.results_root, entry, 'detection_overlay_catalog.json')
        if not os.path.isfile(catalog_path):
            continue
        n_catalogs += 1
        rows = process_catalog(catalog_path, args.image_dir)
        all_rows.extend(rows)
        n_tp  = sum(1 for r in rows if r['result_type'] == 'tp')
        n_fpc = sum(1 for r in rows if r['result_type'] == 'fp_cls')
        n_fpl = sum(1 for r in rows if r['result_type'] == 'fp_loc')
        n_fn  = sum(1 for r in rows if r['result_type'] == 'fn')
        print(f'  {entry}: {len(rows)} objects  (tp={n_tp}, fp_cls={n_fpc}, fp_loc={n_fpl}, fn={n_fn})')

    print(f'\nProcessed {n_catalogs} catalogs  →  {len(all_rows)} total objects')

    out_dir = os.path.dirname(os.path.abspath(args.output_csv))
    os.makedirs(out_dir, exist_ok=True)

    with open(args.output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(all_rows)
    print(f'CSV saved to: {args.output_csv}')

    if args.output_json:
        with open(args.output_json, 'w', encoding='utf-8') as f:
            json.dump(all_rows, f, ensure_ascii=False, indent=2, default=str)
        print(f'JSON saved to: {args.output_json}')


if __name__ == '__main__':
    main()
