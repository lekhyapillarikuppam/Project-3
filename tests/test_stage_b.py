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


# ---- AT-B-7 and AT-B-8: photo-level decision rule and box merge rule (src/ffs/stage_b_eval.py) --------
import pandas as pd  # noqa: E402

sys.path.insert(0, str(ROOT / "src"))
from ffs import stage_b_eval as ev  # noqa: E402


def test_at_b_7_decision_rule():
    # CLIP: category with the highest mean logit gets the largest softmax share; shares sum to 1 per photo
    rows = [("s", "p", c, f"q{i}", v) for c, vals in {"blade": [20, 22, 24], "ammunition": [10, 10, 10]}.items()
            for i, v in enumerate(vals)]
    out = ev.clip_photo_scores(pd.DataFrame(rows, columns=["scene_id", "photo_id", "category", "prompt", "score"]))
    s = out.set_index("category").score
    assert s.sum() == pytest.approx(1.0) and s["blade"] > 0.99
    # threshold is inclusive
    d = ev.decide(pd.DataFrame({"category": ["blade", "blade", "ammunition"], "score": [0.3, 0.29, 0.1]}),
                  {"blade": 0.3, "ammunition": 0.2})
    assert list(d.pred) == [1, 0, 0]
    # OWLv2: best box over the tiles in use; no box at all means score 0
    owl = pd.DataFrame({"scene_id": ["s"] * 2, "photo_id": ["p"] * 2, "category": ["blade"] * 2,
                        "score": [0.2, 0.6], "tile": [0, 3]})
    photos = pd.DataFrame({"scene_id": ["s"], "photo_id": ["p"]})
    full = ev.owl_photo_scores(owl, photos, "full").set_index("category").score
    both = ev.owl_photo_scores(owl, photos, "all").set_index("category").score
    assert full["blade"] == 0.2 and both["blade"] == 0.6 and both["ammunition"] == 0.0


def test_at_b_7_untuned_category_gets_median_threshold():
    n = 10
    cats = ["blade", "ammunition", "bottle_glass", "paper_document", "red_brown_stain"]
    df = pd.DataFrame({"category": [c for c in cats for _ in range(n)],
                       "score": [0.9, 0.1] * (n // 2) * 5, "y": [1, 0] * (n // 2) * 5})
    df.loc[df.category == "bottle_glass", "y"] = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]       # only 1 positive
    th, flagged = ev.choose_thresholds(df, [0.2, 0.5, 0.8])
    assert flagged == ["bottle_glass"]
    others = [th[c] for c in cats if c != "bottle_glass"]
    assert th["bottle_glass"] == pytest.approx(float(np.median(others)))


def test_at_b_8_nms_merge():
    boxes = pd.DataFrame({"x": [0, 5, 500], "y": [0, 5, 500], "w": [100, 100, 50], "h": [100, 100, 50],
                          "score": [0.4, 0.7, 0.3]})
    kept = ev.nms(boxes)
    assert list(kept.score) == [0.7, 0.3]                 # the overlapping weaker box is dropped, the far box stays
    assert ev.iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0 and ev.iou((0, 0, 10, 10), (20, 20, 5, 5)) == 0.0
    assert kept.score.max() == boxes.score.max()           # merging never changes the photo score
