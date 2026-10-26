"""Read the number on a marker, then check it against the scene's exhibit list (SPEC 4.4).

Two parts:
  * `constrained_read`: pure logic, no OCR needed. Uses the list of marker numbers
    that exist in the scene to accept, correct ("snap") or reject a raw read.
  * `read_marker`: crop -> preprocess -> EasyOCR (digits only) -> constrained_read.
"""

from __future__ import annotations

import cv2
import numpy as np

from .markers import Box, MarkerParams, crop_box, yellow_mask

# Digit pairs that OCR often mixes up (R-A3-4, rule 3).
CONFUSABLE = {frozenset(p) for p in [("0", "8"), ("1", "7"), ("3", "8"),
                                      ("5", "6"), ("6", "8"), ("4", "9")]}
MIN_SNAP_CONF = 0.40


def _one_edit_apart(a: str, b: str) -> bool:
    """True if a and b differ by exactly one changed, added or removed character."""
    if a == b:
        return False
    if len(a) == len(b):
        return sum(x != y for x, y in zip(a, b)) == 1
    if abs(len(a) - len(b)) != 1:
        return False
    short, long_ = (a, b) if len(a) < len(b) else (b, a)
    return any(long_[:i] + long_[i + 1:] == short for i in range(len(long_)))


def _is_confusable_swap(a: str, b: str) -> bool:
    if len(a) != len(b):
        return False
    diffs = [(x, y) for x, y in zip(a, b) if x != y]
    return len(diffs) == 1 and frozenset(diffs[0]) in CONFUSABLE


def constrained_read(raw: str, conf: float, valid: set[int],
                     min_conf: float = MIN_SNAP_CONF) -> tuple[int | None, str]:
    """Return (marker_no or None, status) with status in accepted / snapped / rejected."""
    raw = (raw or "").strip()
    if not raw:
        return None, "rejected"
    valid_str = {str(n): n for n in valid}

    if raw in valid_str:                                   # rule 1
        return valid_str[raw], "accepted"
    if conf < min_conf:
        return None, "rejected"

    near = [s for s in valid_str if _one_edit_apart(raw, s)]   # rule 2
    if len(near) == 1:
        return valid_str[near[0]], "snapped"
    if len(near) > 1:                                      # rule 3
        confusable = [s for s in near if _is_confusable_swap(raw, s)]
        if len(confusable) == 1:
            return valid_str[confusable[0]], "snapped"
    return None, "rejected"                                # rule 4


def preprocess_crop(crop_bgr: np.ndarray, params: MarkerParams = MarkerParams(),
                    min_height: int = 96) -> np.ndarray:
    """Black digits on a white background, large enough for OCR (R-A3-2)."""
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    dark = hsv[:, :, 2] < params.ink_v_max

    contours, _ = cv2.findContours(yellow_mask(hsv, params), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    region = np.zeros(dark.shape, np.uint8)
    if contours:
        cv2.drawContours(region, [max(contours, key=cv2.contourArea)], -1, 255, cv2.FILLED)
        ink = dark & (region > 0)
    else:
        ink = dark                                          # no yellow found: use all dark pixels

    out = np.full(dark.shape, 255, np.uint8)
    out[ink] = 0
    scale = max(1.0, min_height / out.shape[0])
    if scale > 1.0:
        out = cv2.resize(out, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    return cv2.copyMakeBorder(out, 16, 16, 16, 16, cv2.BORDER_CONSTANT, value=255)


_READER = None


def get_reader():
    """EasyOCR reader, created once. The first call downloads model files."""
    global _READER
    if _READER is None:
        import easyocr

        _READER = easyocr.Reader(["en"], gpu=False, verbose=False)
    return _READER


def read_digits(prepped: np.ndarray, reader=None) -> tuple[str, float]:
    """(joined digits left to right, mean confidence) (R-A3-3)."""
    reader = reader or get_reader()
    pieces = reader.readtext(prepped, allowlist="0123456789", detail=1, paragraph=False)
    if not pieces:
        return "", 0.0
    pieces = sorted(pieces, key=lambda r: min(pt[0] for pt in r[0]))
    text = "".join(p[1] for p in pieces if p[1].isdigit())
    conf = float(np.mean([p[2] for p in pieces]))
    return text, conf


def read_marker(img_bgr: np.ndarray, box: Box, valid: set[int], reader=None,
                params: MarkerParams = MarkerParams()) -> dict:
    """Full A3 step for one marker box. Keys match the `markers` contract columns."""
    prepped = preprocess_crop(crop_box(img_bgr, box), params)
    raw, conf = read_digits(prepped, reader)
    marker_no, status = constrained_read(raw, conf, valid)
    return {"raw_ocr": raw, "ocr_conf": round(conf, 3),
            "marker_no": marker_no, "status": status}
