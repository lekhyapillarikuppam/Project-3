"""Acceptance tests AT-A3-1 to AT-A3-10 (SPEC 4.4)."""

import numpy as np
import pytest

import cv2

from ffs.ocr import constrained_read, pick_best, preprocess_crop
from ffs.synthetic import cluttered_background, draw_marker


def test_at_a3_1_exact_read_is_accepted():
    assert constrained_read("7", 0.9, {1, 7, 9}) == (7, "accepted")


def test_at_a3_2_snaps_to_the_one_valid_neighbour():
    assert constrained_read("18", 0.8, {16, 7}) == (16, "snapped")


def test_at_a3_3_low_confidence_is_rejected():
    assert constrained_read("18", 0.2, {16, 7}) == (None, "rejected")


def test_at_a3_4_confusable_digit_breaks_a_tie():
    assert constrained_read("18", 0.8, {16, 19}) == (16, "snapped")


def test_at_a3_5_empty_read_is_rejected():
    assert constrained_read("", 0.9, {1}) == (None, "rejected")


def test_at_a3_6_extra_digit_snaps_to_shorter_number():
    assert constrained_read("123", 0.8, {12, 7}) == (12, "snapped")


def test_at_a3_7_one_digit_read_is_never_snapped():
    assert constrained_read("1", 1.0, {3, 7, 9, 12, 16}) == (None, "rejected")


def test_at_a3_8_one_digit_exact_match_is_accepted():
    assert constrained_read("1", 1.0, {1, 7}) == (1, "accepted")


def test_at_a3_9_better_status_wins_over_confidence():
    valid = {12, 16}
    reads = [("1", 1.0), ("12", 0.8)]
    assert pick_best(reads, valid) == {"raw": "12", "conf": 0.8, "marker_no": 12, "status": "accepted"}


def test_at_a3_10_two_digit_marker_keeps_both_digits():
    img = cluttered_background(500, 500)
    box = draw_marker(img, 100, 100, 150, 190, "12")
    crop = img[box.y:box.y + box.h, box.x:box.x + box.w]
    for wide in (False, True):
        out = preprocess_crop(crop, wide=wide)
        n, _ = cv2.connectedComponents((out == 0).astype(np.uint8))
        assert n - 1 == 2, f"wide={wide}: expected 2 digit blobs"


def test_at_a3_11_digits_fill_the_preprocessed_image():
    img = cluttered_background(500, 500)
    box = draw_marker(img, 100, 100, 150, 190, "7")
    crop = img[box.y:box.y + box.h, box.x:box.x + box.w]
    out = preprocess_crop(crop)[16:-16, 16:-16]          # drop the white border
    rows = np.where((out == 0).any(axis=1))[0]
    assert (rows.max() - rows.min() + 1) / out.shape[0] >= 0.4
    assert rows.max() - rows.min() + 1 >= 80


def test_unknown_number_far_from_everything_is_rejected():
    assert constrained_read("55", 0.9, {1, 7, 9}) == (None, "rejected")


def test_untied_unconfusable_ties_are_rejected():
    assert constrained_read("20", 0.9, {21, 22}) == (None, "rejected")


def test_preprocess_gives_black_digits_on_white():
    img = cluttered_background(400, 400)
    box = draw_marker(img, 100, 100, 120, 150, "7")
    crop = img[box.y:box.y + box.h, box.x:box.x + box.w]
    out = preprocess_crop(crop)
    assert out.dtype == np.uint8 and out.ndim == 2
    assert out.shape[0] >= 96
    assert (out == 255).mean() > 0.7      # mostly white background
    assert (out == 0).sum() > 0           # some black ink


def test_real_ocr_reads_a_synthetic_digit():
    pytest.importorskip("easyocr")        # runs only after `pip install -r requirements-ml.txt`
    from ffs.ocr import read_marker
    from ffs.markers import find_markers

    img = cluttered_background(900, 1200)
    draw_marker(img, 500, 300, 120, 150, "7")
    box = find_markers(img)[0]
    result = read_marker(img, box, valid={3, 7, 9})
    assert result["marker_no"] == 7
