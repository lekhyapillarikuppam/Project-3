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


def draw_marker(img, x, y, w, h, text):
    """A tent-like yellow trapezoid with a black number. Returns the true box."""
    pts = np.array([[x + w * 0.15, y], [x + w * 0.85, y], [x + w, y + h], [x, y + h]], np.int32)
    cv2.fillPoly(img, [pts], YELLOW_BGR)
    cv2.putText(img, text, (x + int(w * 0.22), y + int(h * 0.78)), cv2.FONT_HERSHEY_SIMPLEX,
                h / 55, (0, 0, 0), max(2, h // 18), cv2.LINE_AA)
    return Box(x, y, w, h, 1.0)
