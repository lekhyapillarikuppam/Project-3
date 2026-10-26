"""Find yellow numbered markers in a photo by colour + ink (SPEC 4.3).

Idea in plain words: a marker is a yellow blob with dark printed digits on it.
We (1) keep yellow pixels, (2) keep blobs of a sensible size and solidity, and
(3) keep only blobs that contain dark ink, which rejects plain yellow objects.
We do not insist on a triangle, because a tent-shaped placard looks different
from different angles (decision D-007).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class MarkerParams:
    """All tunable thresholds. Log any change made after looking at real photos (R-A2-7)."""

    work_side: int = 1600          # longest side of the working copy, pixels
    h_lo: int = 18                 # yellow hue range, OpenCV scale 0-179
    h_hi: int = 38
    s_min: int = 110
    v_min: int = 110
    close_kernel: int = 5          # morphological close to join broken yellow regions
    min_area_frac: float = 1e-4    # blob area as a fraction of the image
    max_area_frac: float = 0.15
    min_solidity: float = 0.70     # blob area / convex hull area
    min_aspect: float = 0.4        # bounding box width / height
    max_aspect: float = 2.5
    ink_v_max: int = 90            # "dark" means brightness below this
    min_ink_frac: float = 0.02     # dark pixels inside the blob / blob area
    max_ink_frac: float = 0.45
    merge_iou: float = 0.30


@dataclass
class Box:
    x: int
    y: int
    w: int
    h: int
    score: float


def iou(a: Box, b: Box) -> float:
    ix = max(0, min(a.x + a.w, b.x + b.w) - max(a.x, b.x))
    iy = max(0, min(a.y + a.h, b.y + b.h) - max(a.y, b.y))
    inter = ix * iy
    union = a.w * a.h + b.w * b.h - inter
    return inter / union if union else 0.0


def yellow_mask(hsv: np.ndarray, p: MarkerParams) -> np.ndarray:
    mask = cv2.inRange(hsv, (p.h_lo, p.s_min, p.v_min), (p.h_hi, 255, 255))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (p.close_kernel, p.close_kernel))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


def find_markers(img_bgr: np.ndarray, params: MarkerParams = MarkerParams()) -> list[Box]:
    """Boxes in ORIGINAL pixel coordinates, best first (R-A2-1 to R-A2-6)."""
    p = params
    H, W = img_bgr.shape[:2]
    scale = min(1.0, p.work_side / max(H, W))
    small = cv2.resize(img_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) \
        if scale < 1.0 else img_bgr

    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    mask = yellow_mask(hsv, p)
    dark = hsv[:, :, 2] < p.ink_v_max
    img_area = small.shape[0] * small.shape[1]

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    found: list[Box] = []
    for c in contours:
        area = cv2.contourArea(c)
        if not (p.min_area_frac <= area / img_area <= p.max_area_frac):
            continue
        hull_area = cv2.contourArea(cv2.convexHull(c))
        if hull_area <= 0:
            continue
        solidity = area / hull_area
        if solidity < p.min_solidity:
            continue
        x, y, w, h = cv2.boundingRect(c)
        if not (p.min_aspect <= w / h <= p.max_aspect):
            continue

        # Ink test: dark pixels INSIDE the blob outline (holes such as digits count).
        filled = np.zeros((h, w), np.uint8)
        cv2.drawContours(filled, [c - np.array([[x, y]])], -1, 255, thickness=cv2.FILLED)
        blob_px = max(int((filled > 0).sum()), 1)
        ink_frac = float((dark[y:y + h, x:x + w] & (filled > 0)).sum()) / blob_px
        if not (p.min_ink_frac <= ink_frac <= p.max_ink_frac):
            continue

        score = solidity * (0.5 + 0.5 * min(1.0, ink_frac / 0.08))
        found.append(Box(int(x / scale), int(y / scale), int(round(w / scale)),
                         int(round(h / scale)), float(score)))

    # Merge overlaps, keeping the higher score (R-A2-5).
    found.sort(key=lambda b: b.score, reverse=True)
    kept: list[Box] = []
    for b in found:
        if all(iou(b, k) <= p.merge_iou for k in kept):
            kept.append(b)
    return kept


def crop_box(img_bgr: np.ndarray, box: Box, pad: float = 0.10) -> np.ndarray:
    """Crop the box grown by `pad` on each side, clipped to the image (R-A3-1)."""
    H, W = img_bgr.shape[:2]
    dx, dy = int(box.w * pad), int(box.h * pad)
    x0, y0 = max(0, box.x - dx), max(0, box.y - dy)
    x1, y1 = min(W, box.x + box.w + dx), min(H, box.y + box.h + dy)
    return img_bgr[y0:y1, x0:x1]


def draw_overlay(img_bgr: np.ndarray, boxes: list[Box], labels: list[str] | None = None,
                 max_side: int = 1400) -> np.ndarray:
    """Copy of the photo, downscaled, with boxes and optional labels drawn on it."""
    H, W = img_bgr.shape[:2]
    s = min(1.0, max_side / max(H, W))
    out = cv2.resize(img_bgr, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else img_bgr.copy()
    for i, b in enumerate(boxes):
        p0, p1 = (int(b.x * s), int(b.y * s)), (int((b.x + b.w) * s), int((b.y + b.h) * s))
        cv2.rectangle(out, p0, p1, (0, 0, 255), 3)
        text = labels[i] if labels else f"{i}"
        cv2.putText(out, text, (p0[0], max(20, p0[1] - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (0, 0, 255), 2, cv2.LINE_AA)
    return out
