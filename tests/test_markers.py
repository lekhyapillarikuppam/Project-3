"""Acceptance tests AT-A2-1 to AT-A2-6 (SPEC 4.3), on synthetic images."""

import cv2
import numpy as np

from ffs.markers import MarkerParams, find_markers, iou, yellow_mask
from ffs.synthetic import (YELLOW_BGR, cluttered_background, dark_background, draw_marker,
                           hsv_bgr)

def test_at_a2_1_finds_marker_with_digit():
    img = cluttered_background(900, 1200)
    truth = draw_marker(img, 500, 300, 120, 150, "7")
    boxes = find_markers(img)
    assert len(boxes) == 1
    assert iou(boxes[0], truth) >= 0.5


def test_at_a2_2_plain_yellow_rectangle_is_rejected():
    img = cluttered_background(900, 1200)
    cv2.rectangle(img, (500, 300), (620, 450), YELLOW_BGR, -1)
    assert find_markers(img) == []


def test_at_a2_3_no_yellow_gives_no_boxes():
    assert find_markers(cluttered_background(900, 1200, seed=3)) == []


def test_at_a2_4_boxes_are_in_original_coordinates():
    img = cluttered_background(3200, 4000)  # larger than work_side, so it gets downscaled
    truth = draw_marker(img, 2000, 1500, 220, 260, "12")
    boxes = find_markers(img)
    assert len(boxes) == 1
    assert iou(boxes[0], truth) >= 0.5


def test_two_markers_both_found():
    img = cluttered_background(900, 1200)
    draw_marker(img, 200, 200, 110, 140, "3")
    draw_marker(img, 800, 500, 110, 140, "9")
    assert len(find_markers(img)) == 2


def test_at_a2_5_dark_photo_marker_is_found():
    img = dark_background(900, 1200)
    body = hsv_bgr(19, 250, 100)           # measured on th_0006: body V about 98
    ink = hsv_bgr(19, 250, 20)
    truth = draw_marker(img, 500, 300, 120, 150, "14", body_bgr=body, ink_bgr=ink)
    boxes = find_markers(img)
    assert len(boxes) == 1
    assert iou(boxes[0], truth) >= 0.5


def test_at_a2_6_marker_with_non_compact_outline_is_found():
    img = cluttered_background(900, 1200)
    draw_marker(img, 500, 300, 120, 150, "7")
    cv2.line(img, (560, 450), (710, 515), YELLOW_BGR, 14)   # yellow tail, as if touching another object
    # the test only means something if the blob really is below the old 0.70 limit
    mask = yellow_mask(cv2.cvtColor(img, cv2.COLOR_BGR2HSV), MarkerParams())
    c = max(cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0], key=cv2.contourArea)
    solidity = cv2.contourArea(c) / cv2.contourArea(cv2.convexHull(c))
    assert 0.60 <= solidity < 0.70
    assert len(find_markers(img)) == 1
