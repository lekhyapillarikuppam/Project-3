"""Synthetic test images: a cluttered background and tent-like yellow markers with a number."""

import cv2
import numpy as np

from .markers import Box

YELLOW_BGR = (0, 215, 255)  # a warm yellow


def cluttered_background(h, w, seed=0):
    rng = np.random.default_rng(seed)
    img = np.full((h, w, 3), 120, np.uint8)
    for _ in range(12):  # blue, green, red and grey blocks: no yellow
        x, y = int(rng.integers(0, w - 80)), int(rng.integers(0, h - 80))
        color = [(200, 60, 40), (60, 160, 60), (50, 50, 190), (90, 90, 90)][int(rng.integers(0, 4))]
        cv2.rectangle(img, (x, y), (x + int(rng.integers(40, 160)), y + int(rng.integers(40, 160))),
                      color, -1)
    noise = rng.normal(0, 6, img.shape)
    return np.clip(img + noise, 0, 255).astype(np.uint8)


def hsv_bgr(h, s, v):
    """One BGR colour from OpenCV-scale HSV, to build photos with a known brightness."""
    return tuple(int(c) for c in cv2.cvtColor(np.uint8([[[h, s, v]]]), cv2.COLOR_HSV2BGR)[0, 0])


def dark_background(h, w, seed=0, factor=0.25):
    """The cluttered background, darkened to mimic an underexposed photo."""
    return (cluttered_background(h, w, seed) * factor).astype(np.uint8)


def draw_marker(img, x, y, w, h, text, body_bgr=YELLOW_BGR, ink_bgr=(0, 0, 0)):
    """A tent-like yellow trapezoid with a black number that fits inside. Returns the true box."""
    pts = np.array([[x + w * 0.15, y], [x + w * 0.85, y], [x + w, y + h], [x, y + h]], np.int32)
    cv2.fillPoly(img, [pts], body_bgr)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = h / 55
    thick = max(2, h // 18)
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    if tw > 0.62 * w:                       # two-digit numbers: shrink to stay inside the marker
        scale *= 0.62 * w / tw
        thick = max(2, int(thick * 0.62 * w / tw))
        (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    org = (x + (w - tw) // 2, y + int(h * 0.55) + th // 2)
    cv2.putText(img, text, org, font, scale, ink_bgr, thick, cv2.LINE_AA)
    return Box(x, y, w, h, 1.0)
