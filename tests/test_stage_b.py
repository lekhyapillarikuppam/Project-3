"""Acceptance tests AT-B-1 to AT-B-6 (SPEC section 5): Stage B helpers, no GPU needed."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("stage_b_run", ROOT / "kaggle" / "stage_b_run.py")
sb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sb)

sys.path.insert(0, str(ROOT / "scripts"))
import package_stage_b as pk  # noqa: E402


def test_at_b_1_tiles_cover_the_photo_and_overlap():
    tiles = sb.make_tiles(1000, 800)
    assert tiles[0] == (0, 0, 0, 1000, 800) and len(tiles) == 5
    assert min(t[1] for t in tiles[1:]) == 0 and max(t[3] for t in tiles[1:]) == 1000
    assert min(t[2] for t in tiles[1:]) == 0 and max(t[4] for t in tiles[1:]) == 800
    left, right = tiles[1], tiles[2]
    assert right[1] < left[3]                       # neighbouring tiles overlap


def test_at_b_2_decode_maps_box_back_to_original_pixels():
    scores = np.array([[0.9], [0.02]])
    boxes = np.array([[0.5, 0.25, 0.2, 0.1], [0.5, 0.5, 0.1, 0.1]])
    d = sb.decode_boxes(scores, boxes, side=1000, off_x=100, off_y=50, max_x=2000, max_y=2000, scale=2.0)
    assert len(d[0]) == 1                           # the 0.02 box is under the cut-off
    s, x, y, w, h = d[0][0]
    assert s == pytest.approx(0.9)
    assert (x, y, w, h) == pytest.approx((2 * (400 + 100), 2 * (200 + 50), 2 * 200, 2 * 100))


def test_at_b_3_boxes_in_the_padding_are_dropped_and_clamped():
    scores = np.array([[0.8], [0.8]])
    boxes = np.array([[0.5, 0.95, 0.2, 0.05],       # inside the padded bottom strip of a 1000x500 tile
                      [0.95, 0.25, 0.2, 0.2]])      # sticks out of the right edge
    d = sb.decode_boxes(scores, boxes, side=1000, off_x=0, off_y=0, max_x=1000, max_y=500)
    assert len(d[0]) == 1
    _, x, y, w, h = d[0][0]
    assert x + w <= 1000.0001


def test_at_b_4_top_k_limits_boxes_per_query():
    scores = np.linspace(0.9, 0.1, 100).reshape(-1, 1)
    boxes = np.tile([0.5, 0.5, 0.1, 0.1], (100, 1))
    d = sb.decode_boxes(scores, boxes, side=100, off_x=0, off_y=0, max_x=100, max_y=100)
    assert len(d[0]) == sb.TOP_K


def test_at_b_5_prompts_never_say_blood_and_cover_five_categories():
    text = " ".join(p for ps in list(sb.CLIP_PROMPTS.values()) + list(sb.OWL_QUERIES.values()) for p in ps).lower()
    assert "blood" not in text
    assert set(sb.CLIP_PROMPTS) == set(sb.OWL_QUERIES)
    assert len(sb.flat_queries(sb.OWL_QUERIES)) == sum(len(v) for v in sb.OWL_QUERIES.values())


def test_at_b_6_test_package_needs_the_freeze_tag_and_shrink_keeps_original_size(tmp_path):
    assert pk.protocol_frozen(["protocol-v1"]) and not pk.protocol_frozen([])
    from PIL import Image
    src = tmp_path / "a.jpg"
    Image.new("RGB", (4000, 3000), (200, 10, 10)).save(src)
    w, h = pk.shrink(src, tmp_path / "out" / "a.jpg")
    assert (w, h) == (4000, 3000)
    assert max(Image.open(tmp_path / "out" / "a.jpg").size) == pk.MAX_SIDE
