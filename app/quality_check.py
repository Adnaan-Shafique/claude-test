"""
Classical (non-learned) edge-layer image quality check for PM acceptance photos.

Implements the MM-IQA framework exactly as specified in:
  Aglin, Muchiri, Nkundineza. "A Lightweight Multi-Metric No-Reference Image
  Quality Assessment Framework for UAV Imaging." arXiv:2604.13112.
Equation numbers referenced in comments below (Eq. 1-15) match the paper.

Three additions on top of the paper (documented at each site, not from the paper):
  - EXIF-orientation correction before any cue is computed.
  - A hard minimum-resolution floor, independent of the paper's LowRes% cue.
  - A color-histogram-entropy occlusion/uniformity cue, folded into the same
    weighted-fusion scheme with its own configurable weight.

No neural network, no training data. OpenCV + NumPy for the cues, PIL for EXIF.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image, ImageOps

ImageInput = Union[str, Path, np.ndarray, Image.Image]


@dataclass
class QualityConfig:
    # --- Hard resolution floor (NOT from the paper) ---
    min_width: int = 640
    min_height: int = 480

    # --- Eq. (3), (12): Laplacian variance calibration reference ---
    laplacian_ref: float = 1000.0

    # --- Eq. (4), (12): Tenengrad energy calibration reference ---
    tenengrad_ref: float = 6000.0

    # --- Eq. (5)-(6): Canny edge density ---
    canny_low: int = 100
    canny_high: int = 200
    edge_density_ref_blur: float = 0.05   # used inside Blur%/LowRes% (a_Edge, b_Edge), Eq. (12)-(13)
    edge_density_ref_quality: float = 0.2  # used in final q_edge normalization, Eq. (14)

    # --- Eq. (7): FFT log-energy ---
    fft_ref_lowres: float = 8.0    # used inside LowRes% (b_FFT), Eq. (13)
    fft_ref_quality: float = 9.0   # used in final q_fft normalization, Eq. (14)

    # --- Eq. (8)-(9): noise (median-residual RMS) ---
    noise_ref: float = 15.0  # Eq. (14)

    # --- Eq. (10): exposure clipping thresholds (8-bit gray levels) ---
    underexposed_level: int = 30
    overexposed_level: int = 225

    # --- Eq. (11): dark-channel haze proxy ---
    haze_patch_size: int = 15
    haze_ref: float = 100.0  # Eq. (14)

    # --- Eq. (15): fusion weights, in the paper's order
    # {blur, low-resolution, noise, underexposed, overexposed, haze, edge density, FFT energy} ---
    weight_blur: float = 0.30
    weight_lowres: float = 0.20
    weight_noise: float = 0.15
    weight_underexposed: float = 0.08
    weight_overexposed: float = 0.07
    weight_haze: float = 0.05
    weight_edge_density: float = 0.10
    weight_fft_energy: float = 0.05

    # --- Occlusion / uniformity cue (NOT from the paper) ---
    # Normalized color-histogram entropy in bits; natural photos comfortably
    # exceed this, a finger-over-lens or blank-wall frame falls well below it.
    entropy_ref_bits: float = 5.0
    weight_occlusion: float = 0.10

    # --- Pass/fail and failure-reason reporting (NOT from the paper; the
    # paper only ranks images, it does not define a pass/fail cutoff) ---
    pass_threshold: float = 65.0
    concern_threshold: float = 0.5
    max_reasons: int = 3


CUE_MESSAGES: Dict[str, str] = {
    "resolution_too_low": "Image resolution is too low. Use a higher-resolution camera setting.",
    "blur": "Photo is blurry. Hold the camera steady and retake.",
    "lowres": "Photo lacks fine detail (looks low-resolution). Move closer and refocus.",
    "noise": "Photo is noisy/grainy. Retake in better lighting.",
    "underexposed": "Photo is too dark. Add light or use flash and retake.",
    "overexposed": "Photo is too bright/washed out. Reduce glare and retake.",
    "haze": "Photo appears hazy or foggy. Clean the lens and retake.",
    "edge_density": "Photo lacks sharp detail. Get closer and make sure it's in focus.",
    "fft_energy": "Photo lacks fine detail. Get closer and make sure it's in focus.",
    "occlusion": "Camera may be obstructed (e.g. finger over the lens) or the frame is too uniform. Check the lens and retake.",
}


@dataclass
class CueResult:
    name: str
    raw: float
    normalized: float  # q_i in [0,1], higher = better quality contribution
    weight: float       # normalized weight actually used in the fusion sum


@dataclass
class Intermediates:
    grayscale: np.ndarray
    laplacian_response: np.ndarray
    tenengrad_magnitude: np.ndarray
    canny_edges: np.ndarray
    fft_magnitude_shifted: np.ndarray
    noise_residual: np.ndarray
    exposure_histogram: np.ndarray
    dark_channel_map: np.ndarray


@dataclass
class QualityResult:
    score: float
    passed: bool
    threshold: float
    resolution_ok: bool
    width: int
    height: int
    cues: Dict[str, CueResult]
    failure_reasons: List[str]
    retake_instructions: List[str]
    intermediates: Optional[Intermediates] = None
    roi_box_used: Optional[Tuple[int, int, int, int]] = None

    def to_dict(self) -> dict:
        return {
            "score": self.score,
            "passed": self.passed,
            "threshold": self.threshold,
            "resolution_ok": self.resolution_ok,
            "width": self.width,
            "height": self.height,
            "cues": {
                name: {"raw": c.raw, "normalized": c.normalized, "weight": c.weight}
                for name, c in self.cues.items()
            },
            "failure_reasons": self.failure_reasons,
            "retake_instructions": self.retake_instructions,
            "roi_box_used": self.roi_box_used,
        }


def _load_image_bgr(image: ImageInput) -> np.ndarray:
    if isinstance(image, np.ndarray):
        arr = image
        if arr.ndim == 2:
            arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
        return arr

    pil_img = image if isinstance(image, Image.Image) else Image.open(image)
    pil_img = ImageOps.exif_transpose(pil_img)  # correct camera-app mis-rotation
    return cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)


def load_image_bgr(image: ImageInput) -> np.ndarray:
    """Public entry point for the same EXIF-corrected BGR loading assess_quality()
    uses internally — for callers (e.g. run_quality_batch_foreground.py) that need
    the array once and reuse it for both segmentation and scoring."""
    return _load_image_bgr(image)


def _clip01(x: float) -> float:
    return float(min(1.0, max(0.0, x)))


def _clip_box_to_image(box: Tuple[int, int, int, int], width: int, height: int) -> Optional[Tuple[int, int, int, int]]:
    # Clamps an (x1, y1, x2, y2) ROI to the image bounds. Returns None if the
    # box doesn't overlap the image at all or collapses to zero area after
    # clamping (caller should fall back to the whole frame in that case).
    x1, y1, x2, y2 = box
    x1 = max(0, min(int(x1), width))
    y1 = max(0, min(int(y1), height))
    x2 = max(0, min(int(x2), width))
    y2 = max(0, min(int(y2), height))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _to_grayscale(image_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)


def _laplacian_variance(gray: np.ndarray):
    # Eq. (1)-(3): 3x3 four-neighbour stencil [[0,1,0],[1,-4,1],[0,1,0]] is
    # exactly OpenCV's Laplacian kernel for ksize=1.
    lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=1)
    lv = float(np.var(lap, ddof=1))  # sample variance, 1/(MN-1) as in Eq. (3)
    return lv, lap


def _tenengrad(gray: np.ndarray):
    # Eq. (4): mean squared Sobel gradient magnitude.
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag_sq = gx ** 2 + gy ** 2
    t = float(np.mean(mag_sq))
    return t, mag_sq


def _edge_density(gray: np.ndarray, low: int, high: int):
    # Eq. (5)-(6): fixed 1:2 hysteresis Canny thresholds.
    edges = cv2.Canny(gray, low, high)
    ed = float(np.count_nonzero(edges) / edges.size)
    return ed, edges


def _fft_energy(gray: np.ndarray):
    # Eq. (7): mean log-compressed FFT magnitude over the full spectrum.
    spectrum = np.fft.fft2(gray.astype(np.float64))
    mag = np.abs(spectrum)
    f = float(np.mean(np.log1p(mag)))
    mag_shifted = np.fft.fftshift(mag)  # for visualization only
    return f, mag_shifted


def _noise_estimate(gray: np.ndarray):
    # Eq. (8)-(9): RMS residual between the image and a 3x3 median blur.
    median = cv2.medianBlur(gray, 3)
    diff = gray.astype(np.float64) - median.astype(np.float64)
    n = float(np.sqrt(np.mean(diff ** 2)))
    return n, diff


def _exposure_balance(gray: np.ndarray, t_u: int, t_o: int):
    # Eq. (10): tail mass of the normalized histogram below T_u / above T_o.
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    p = hist / hist.sum()
    u_pct = float(100.0 * p[:t_u].sum())
    o_pct = float(100.0 * p[t_o + 1:].sum())
    return u_pct, o_pct, hist


def _haze_proxy(image_bgr: np.ndarray, patch_size: int):
    # Eq. (11): dark-channel prior, s x s rectangular erosion.
    b, g, r = cv2.split(image_bgr.astype(np.float64))
    dark = np.minimum(np.minimum(r, g), b)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch_size, patch_size))
    eroded = cv2.erode(dark, kernel)
    h = float(np.mean(eroded))
    return h, eroded


def _color_entropy_bits(image_bgr: np.ndarray) -> float:
    # NOT from the paper. Per-channel Shannon entropy of the color histogram,
    # averaged across channels; a near-uniform frame (obstructed lens, blank
    # wall) collapses toward 0 bits, natural scenes sit well above entropy_ref_bits.
    entropies = []
    for ch in range(3):
        hist = cv2.calcHist([image_bgr], [ch], None, [256], [0, 256]).flatten()
        p = hist / hist.sum()
        p_nonzero = p[p > 0]
        entropies.append(float(-np.sum(p_nonzero * np.log2(p_nonzero))))
    return float(np.mean(entropies))


def _blur_percentage(lv: float, t: float, ed: float, cfg: QualityConfig) -> float:
    # Eq. (12)
    a_lap = _clip01((cfg.laplacian_ref - lv) / cfg.laplacian_ref)
    a_ten = _clip01((cfg.tenengrad_ref - t) / cfg.tenengrad_ref)
    a_edge = _clip01((cfg.edge_density_ref_blur - ed) / cfg.edge_density_ref_blur)
    return 100.0 * (a_lap + a_ten + a_edge) / 3.0


def _lowres_percentage(ed: float, f: float, cfg: QualityConfig) -> float:
    # Eq. (13)
    b_edge = _clip01((cfg.edge_density_ref_blur - ed) / cfg.edge_density_ref_blur)
    b_fft = _clip01((cfg.fft_ref_lowres - f) / cfg.fft_ref_lowres)
    return 100.0 * (b_edge + b_fft) / 2.0


def assess_quality(
    image: ImageInput,
    config: Optional[QualityConfig] = None,
    return_intermediates: bool = False,
    roi_box: Optional[Tuple[int, int, int, int]] = None,
) -> QualityResult:
    """Run the full MM-IQA cue pipeline plus resolution/occlusion checks on one image.

    roi_box: optional (x1, y1, x2, y2) pixel box (e.g. from a detector or a
    foreground segmenter). When given, every content cue (blur, lowres,
    noise, exposure, haze, edge density, FFT energy, occlusion) is computed
    on the cropped region instead of the whole frame — this is what fixes
    sky/background-dominated shots swamping the whole-frame averages (see
    README.md §5). The resolution floor still checks the *original* frame's
    width/height, since that's about camera capture settings, not framing.
    An invalid/degenerate box (fully outside the frame, zero area) silently
    falls back to the whole frame rather than erroring.
    """
    cfg = config or QualityConfig()

    image_bgr_full = _load_image_bgr(image)
    height, width = image_bgr_full.shape[:2]
    resolution_ok = width >= cfg.min_width and height >= cfg.min_height

    roi_box_used = None
    image_bgr = image_bgr_full
    if roi_box is not None:
        clipped = _clip_box_to_image(roi_box, width, height)
        if clipped is not None:
            x1, y1, x2, y2 = clipped
            image_bgr = image_bgr_full[y1:y2, x1:x2]
            roi_box_used = clipped

    gray = _to_grayscale(image_bgr)

    lv, lap = _laplacian_variance(gray)
    t, ten_mag = _tenengrad(gray)
    ed, edges = _edge_density(gray, cfg.canny_low, cfg.canny_high)
    f, fft_mag = _fft_energy(gray)
    n, noise_resid = _noise_estimate(gray)
    u_pct, o_pct, hist = _exposure_balance(gray, cfg.underexposed_level, cfg.overexposed_level)
    h, dark_map = _haze_proxy(image_bgr, cfg.haze_patch_size)
    entropy_bits = _color_entropy_bits(image_bgr)

    blur_pct = _blur_percentage(lv, t, ed, cfg)
    lowres_pct = _lowres_percentage(ed, f, cfg)

    # Eq. (14): normalize every cue to a quality contribution q_i in [0,1].
    q = {
        "blur": 1.0 - blur_pct / 100.0,
        "lowres": 1.0 - lowres_pct / 100.0,
        "noise": 1.0 - min(n / cfg.noise_ref, 1.0),
        "underexposed": 1.0 - u_pct / 100.0,
        "overexposed": 1.0 - o_pct / 100.0,
        "haze": 1.0 - min(h / cfg.haze_ref, 1.0),
        "edge_density": min(ed / cfg.edge_density_ref_quality, 1.0),
        "fft_energy": min(f / cfg.fft_ref_quality, 1.0),
        "occlusion": min(entropy_bits / cfg.entropy_ref_bits, 1.0),
    }
    raw = {
        "blur": blur_pct,
        "lowres": lowres_pct,
        "noise": n,
        "underexposed": u_pct,
        "overexposed": o_pct,
        "haze": h,
        "edge_density": ed,
        "fft_energy": f,
        "occlusion": entropy_bits,
    }

    # Eq. (15) weight vector, extended with the occlusion cue's own weight and
    # renormalized to sum to 1 (weight_occlusion=0 reproduces the paper exactly).
    weights_raw = {
        "blur": cfg.weight_blur,
        "lowres": cfg.weight_lowres,
        "noise": cfg.weight_noise,
        "underexposed": cfg.weight_underexposed,
        "overexposed": cfg.weight_overexposed,
        "haze": cfg.weight_haze,
        "edge_density": cfg.weight_edge_density,
        "fft_energy": cfg.weight_fft_energy,
        "occlusion": cfg.weight_occlusion,
    }
    total_w = sum(weights_raw.values())
    weights = {k: v / total_w for k, v in weights_raw.items()}

    score = _clip01(sum(weights[k] * q[k] for k in q)) * 100.0

    cues = {k: CueResult(name=k, raw=raw[k], normalized=q[k], weight=weights[k]) for k in q}

    passed = resolution_ok and score >= cfg.pass_threshold

    reasons: List[str] = []
    instructions: List[str] = []
    if not resolution_ok:
        reasons.append("resolution_too_low")
        instructions.append(CUE_MESSAGES["resolution_too_low"])

    if score < cfg.pass_threshold:
        contributions = sorted(
            ((k, weights[k] * (1.0 - q[k])) for k in q if q[k] < cfg.concern_threshold),
            key=lambda kv: kv[1],
            reverse=True,
        )
        for name, _ in contributions[: cfg.max_reasons]:
            if name not in reasons:
                reasons.append(name)
            msg = CUE_MESSAGES[name]
            if msg not in instructions:
                instructions.append(msg)

    intermediates = None
    if return_intermediates:
        intermediates = Intermediates(
            grayscale=gray,
            laplacian_response=lap,
            tenengrad_magnitude=ten_mag,
            canny_edges=edges,
            fft_magnitude_shifted=fft_mag,
            noise_residual=noise_resid,
            exposure_histogram=hist,
            dark_channel_map=dark_map,
        )

    return QualityResult(
        score=score,
        passed=passed,
        threshold=cfg.pass_threshold,
        resolution_ok=resolution_ok,
        width=width,
        height=height,
        cues=cues,
        failure_reasons=reasons,
        retake_instructions=instructions,
        intermediates=intermediates,
        roi_box_used=roi_box_used,
    )
