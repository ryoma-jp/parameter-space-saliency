"""Image-region feature computation for detection analysis.

Each feature operates on a BGR uint8 crop (H x W x 3 numpy array).
All values are scalars; NaN is returned for degenerate crops.

Feature reference
-----------------
luminance_mean/std/median  : Weighted greyscale (Rec.601) statistics.
rms_contrast               : RMS = sqrt(E[I²] - E[I]²).  Low → low local contrast.
saturation_mean/std        : OpenCV HSV S-channel (0–255).
hue_std                    : Circular std of HSV H-channel (radians).
                             sqrt(-2 ln R) where R = |mean of unit vectors|.
colorfulness               : Hasler & Süsstrunk (2003).
                             sqrt(σ_rg² + σ_yb²) + 0.3 sqrt(μ_rg² + μ_yb²).
noise_sigma                : MAD/0.6745 of Laplacian residual.
                             Estimates sensor/texture noise.
total_variation            : Anisotropic L1-TV normalised by pixel count.
                             TV(I)/N = (Σ|Δₓ| + Σ|Δᵧ|) / N.
sharpness_laplacian        : Variance of Laplacian response.  Low → blurry.
edge_density               : Fraction of Canny edge pixels.
"""

import cv2
import numpy as np


_FEATURE_KEYS = [
    'luminance_mean', 'luminance_std', 'luminance_median',
    'rms_contrast',
    'saturation_mean', 'saturation_std',
    'hue_std',
    'colorfulness',
    'noise_sigma',
    'total_variation',
    'sharpness_laplacian',
    'edge_density',
]


def _nan_features() -> dict:
    return {k: float('nan') for k in _FEATURE_KEYS}


def compute_image_features(crop_bgr: np.ndarray) -> dict:
    """Compute all image features from a BGR uint8 crop.

    Args:
        crop_bgr: H x W x 3 numpy uint8 array in BGR channel order,
                  as returned by cv2.imread or array slicing.
                  Pass None or an empty array to get NaN-filled output.

    Returns:
        Dict mapping feature name to float scalar.
        All values are NaN when the crop is degenerate (None / empty).
    """
    if crop_bgr is None:
        return _nan_features()
    if crop_bgr.ndim != 3 or crop_bgr.shape[2] != 3:
        return _nan_features()
    h, w = crop_bgr.shape[:2]
    if h < 1 or w < 1:
        return _nan_features()

    # ------------------------------------------------------------------ #
    # Prepare colour-space representations                                 #
    # ------------------------------------------------------------------ #
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY).astype(np.float64)
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)   # H∈[0,179], S,V∈[0,255]
    gray_u8 = np.clip(gray, 0, 255).astype(np.uint8)

    # ------------------------------------------------------------------ #
    # Luminance                                                            #
    # ------------------------------------------------------------------ #
    lum_mean = float(np.mean(gray))
    lum_std = float(np.std(gray))
    lum_median = float(np.median(gray))

    # ------------------------------------------------------------------ #
    # RMS contrast                                                         #
    # ------------------------------------------------------------------ #
    rms_contrast = float(np.sqrt(max(0.0, np.mean(gray ** 2) - lum_mean ** 2)))

    # ------------------------------------------------------------------ #
    # Saturation (OpenCV HSV, S ∈ [0, 255])                               #
    # ------------------------------------------------------------------ #
    s_ch = hsv[:, :, 1].astype(np.float64)
    sat_mean = float(np.mean(s_ch))
    sat_std = float(np.std(s_ch))

    # ------------------------------------------------------------------ #
    # Hue circular standard deviation                                      #
    # OpenCV H ∈ [0, 179] represents 0–360°, so multiply by π/90 to get  #
    # radians over the full circle.                                        #
    # Circular std = sqrt(-2 ln R) where R = |mean unit vector|.          #
    # ------------------------------------------------------------------ #
    h_ch = hsv[:, :, 0].astype(np.float64)
    angle_rad = h_ch * (np.pi / 90.0)
    sin_m = float(np.mean(np.sin(angle_rad)))
    cos_m = float(np.mean(np.cos(angle_rad)))
    R = np.sqrt(sin_m ** 2 + cos_m ** 2)
    hue_std = float(np.sqrt(max(0.0, -2.0 * np.log(R + 1e-12))))

    # ------------------------------------------------------------------ #
    # Colorfulness – Hasler & Süsstrunk (2003)                            #
    # Values in [0, 255] range.                                            #
    # ------------------------------------------------------------------ #
    R_ch = crop_bgr[:, :, 2].astype(np.float64)
    G_ch = crop_bgr[:, :, 1].astype(np.float64)
    B_ch = crop_bgr[:, :, 0].astype(np.float64)
    rg = R_ch - G_ch
    yb = 0.5 * (R_ch + G_ch) - B_ch
    colorfulness = float(
        np.sqrt(np.std(rg) ** 2 + np.std(yb) ** 2)
        + 0.3 * np.sqrt(np.mean(rg) ** 2 + np.mean(yb) ** 2)
    )

    # ------------------------------------------------------------------ #
    # Noise sigma – MAD of Laplacian residual (Donoho & Johnstone)        #
    # ------------------------------------------------------------------ #
    lap = cv2.Laplacian(gray_u8, cv2.CV_64F)
    lap_median = float(np.median(lap))
    noise_sigma = float(np.median(np.abs(lap - lap_median)) / 0.6745)

    # ------------------------------------------------------------------ #
    # Total Variation – anisotropic L1, normalised by pixel count          #
    # Units: pixel-value difference per pixel.                             #
    # ------------------------------------------------------------------ #
    dv = np.diff(gray, axis=0)   # vertical differences   (H-1, W)
    dh = np.diff(gray, axis=1)   # horizontal differences (H, W-1)
    total_variation = float((np.sum(np.abs(dv)) + np.sum(np.abs(dh))) / float(gray.size))

    # ------------------------------------------------------------------ #
    # Sharpness – Laplacian variance                                       #
    # Higher → sharper; lower → blurrier.                                  #
    # ------------------------------------------------------------------ #
    sharpness_laplacian = float(np.var(lap))

    # ------------------------------------------------------------------ #
    # Edge density – fraction of Canny edge pixels                         #
    # ------------------------------------------------------------------ #
    edges = cv2.Canny(gray_u8, 50, 150)
    edge_density = float(np.mean(edges > 0))

    return {
        'luminance_mean':      round(lum_mean,           4),
        'luminance_std':       round(lum_std,            4),
        'luminance_median':    round(lum_median,         4),
        'rms_contrast':        round(rms_contrast,       4),
        'saturation_mean':     round(sat_mean,           4),
        'saturation_std':      round(sat_std,            4),
        'hue_std':             round(hue_std,            4),
        'colorfulness':        round(colorfulness,       4),
        'noise_sigma':         round(noise_sigma,        4),
        'total_variation':     round(total_variation,    4),
        'sharpness_laplacian': round(sharpness_laplacian, 4),
        'edge_density':        round(edge_density,       4),
    }
