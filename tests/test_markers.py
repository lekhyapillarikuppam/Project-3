"""Acceptance tests AT-A2-1 to AT-A2-4 (SPEC 4.3), on synthetic images."""

import cv2

from ffs.markers import find_markers, iou
from ffs.synthetic import YELLOW_BGR, cluttered_background, draw_marker

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
