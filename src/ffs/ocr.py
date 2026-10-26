"""Read the number on a marker, then check it against the scene's exhibit list (SPEC 4.4).

Two parts:
  * `constrained_read`: pure logic, no OCR needed. Uses the list of marker numbers
    that exist in the scene to accept, correct ("snap") or reject a raw read.
  * `read_marker`: crop -> preprocess -> EasyOCR (digits only) -> constrained_read.
"""

from __future__ import annotations

import cv2
import numpy as np

from .markers import Box, MarkerParams, crop_box, ink_pixels, ink_threshold, yellow_mask

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

    if len(raw) < 2:                                       # R-A3-5: never snap one digit
        return None, "rejected"

    near = [s for s in valid_str if _one_edit_apart(raw, s)]   # rule 2
    if len(near) == 1:
        return valid_str[near[0]], "snapped"
    if len(near) > 1:                                      # rule 3
        confusable = [s for s in near if _is_confusable_swap(raw, s)]
        if len(confusable) == 1:
            return valid_str[confusable[0]], "snapped"
    return None, "rejected"                                # rule 4


MAX_HEIGHT = 220          # R-A3-9
MIN_PIECE_FRAC = 0.30     # ink pieces shorter than this share of the tallest piece are dropped


def drop_short_pieces(ink: np.ndarray) -> np.ndarray:
    """Keep only ink pieces about as tall as the digits; the printed ruler strip is thin and short."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8))
    if n <= 1:
        return ink
    heights = stats[:, cv2.CC_STAT_HEIGHT]
    keep = heights >= MIN_PIECE_FRAC * heights[1:].max()
    keep[0] = False                                         # label 0 is the background
    return keep[labels]


def preprocess_crop(crop_bgr: np.ndarray, params: MarkerParams = MarkerParams(),
                    min_height: int = 140, wide: bool = False) -> np.ndarray:
    """Black digits on a white background, large enough for OCR (R-A3-2).

    tight (wide=False): dark pixels inside the yellow region.
    wide: the yellow region is grown by 12% of the crop size first, so digits that touch
    or cross the marker's edge are kept.
    """
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2]

    mask = yellow_mask(hsv, params)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        region = np.zeros(v.shape, np.uint8)
        cv2.drawContours(region, [max(contours, key=cv2.contourArea)], -1, 255, cv2.FILLED)
        ink = ink_pixels(v, region > 0, mask > 0, params)   # same rule as the finder (R-A2-8)
        if wide:
            k = max(3, int(0.12 * min(v.shape)) | 1)
            grown = cv2.dilate(region, np.ones((k, k), np.uint8)) > 0
            # judge darkness against the body, but accept ink anywhere in the grown region
            body_v = v[(mask > 0) & (region > 0)]
            ink = (v < ink_threshold(float(np.median(body_v)), params)) & grown
    else:                                                   # no yellow found: judge against the whole crop
        ink = v < ink_threshold(float(np.median(v)), params)

    ink = drop_short_pieces(ink)                            # R-A3-9: no ruler strip, no specks
    out = np.full(v.shape, 255, np.uint8)
    out[ink] = 0
    ys, xs = np.where(ink)
    if len(ys) > 0:                                         # D-012: crop to the ink box + margin
        m = max(2, int(0.30 * (ys.max() - ys.min() + 1)))
        out = out[max(0, ys.min() - m):ys.max() + m + 1, max(0, xs.min() - m):xs.max() + m + 1]
    scale = max(1.0, min_height / out.shape[0])
    if scale > 1.0:
        out = cv2.resize(out, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    elif out.shape[0] > MAX_HEIGHT:                         # R-A3-9: EasyOCR misreads very large images
        out = cv2.resize(out, None, fx=MAX_HEIGHT / out.shape[0], fy=MAX_HEIGHT / out.shape[0],
                         interpolation=cv2.INTER_AREA)
    return cv2.copyMakeBorder(out, 16, 16, 16, 16, cv2.BORDER_CONSTANT, value=255)


_RANK = {"accepted": 0, "snapped": 1, "rejected": 2}


def pick_best(reads: list[tuple[str, float]], valid: set[int]) -> dict:
    """Constrain each (raw, conf) read; keep the best status, ties by confidence (R-A3-6)."""
    cands = []
    for raw, conf in reads:
        no, status = constrained_read(raw, conf, valid)
        cands.append({"raw": raw, "conf": conf, "marker_no": no, "status": status})
    return min(cands, key=lambda c: (_RANK[c["status"]], -c["conf"]))


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


RETRY_ANGLES = (30, -30, 60, -60)
RETRY_BELOW_CONF = 0.60


def rotate_on_white(img: np.ndarray, angle: float) -> np.ndarray:
    """Turn a black-on-white image, growing the canvas so no corner is cut off."""
    h, w = img.shape[:2]
    turn = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    cos, sin = abs(turn[0, 0]), abs(turn[0, 1])
    new_w, new_h = int(h * sin + w * cos), int(h * cos + w * sin)
    turn[0, 2] += new_w / 2 - w / 2
    turn[1, 2] += new_h / 2 - h / 2
    return cv2.warpAffine(img, turn, (new_w, new_h), flags=cv2.INTER_LINEAR, borderValue=255)


def read_marker(img_bgr: np.ndarray, box: Box, valid: set[int], reader=None,
                params: MarkerParams = MarkerParams()) -> dict:
    """Full A3 step for one marker box. Keys match the `markers` contract columns."""
    crop = crop_box(img_bgr, box)
    preps = [preprocess_crop(crop, params, wide=w) for w in (False, True)]
    reads = [read_digits(p, reader) for p in preps]
    best = pick_best(reads, valid)
    if best["status"] == "rejected" or best["conf"] < RETRY_BELOW_CONF:   # R-A3-7: tilted markers
        reads += [read_digits(rotate_on_white(p, a), reader) for p in preps for a in RETRY_ANGLES]
        best = pick_best(reads, valid)
    return {"raw_ocr": best["raw"], "ocr_conf": round(best["conf"], 3),
            "marker_no": best["marker_no"], "status": best["status"]}
