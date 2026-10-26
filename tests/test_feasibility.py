"""AT-F-1 (SPEC 8.2): the feasibility metrics match a hand-computed example."""

import math

import pandas as pd

from ffs.feasibility import compute_metrics, number_set, verdict


def tables():
    truth = pd.DataFrame({
        "scene_id": ["s1", "s1", "s2"], "photo_id": ["p1", "p2", "p3"],
        "n_markers_visible": [2, 0, 1], "true_marker_nos": ["3;7", "", "5"]})
    photos = pd.DataFrame({
        "scene_id": ["s1", "s1", "s2"], "photo_id": ["p1", "p2", "p3"],
        "n_boxes": [3, 0, 0], "read_nos": ["3;9", "", ""]})
    markers = pd.DataFrame({
        "scene_id": ["s1", "s1", "s1"], "photo_id": ["p1", "p1", "p1"],
        "raw_ocr": ["3", "9", "1"]})
    return truth, photos, markers


def test_hand_computed_metrics():
    m = compute_metrics(*tables())
    assert m["n_photos"] == 3 and m["n_true_markers"] == 3
    assert math.isclose(m["finder_hit_rate"], 2 / 3)      # min(3,2)+min(0,0)+min(0,1) over 3
    assert math.isclose(m["finder_fp_per_photo"], 1 / 3)  # one extra box in p1, over 3 photos
    assert math.isclose(m["read_accuracy"], 1 / 3)        # {3,7}&{3,9} -> 1 of 3 true numbers
    assert math.isclose(m["raw_read_accuracy"], 1 / 3)    # raw {3,9,1} -> 1 of 3
    assert not m["pass_finder"] and not m["pass_read"]
    assert verdict(m).startswith("FAIL (finder)")


def test_unlabelled_rows_are_ignored():
    truth, photos, markers = tables()
    truth.loc[2, "n_markers_visible"] = None               # not labelled yet
    assert compute_metrics(truth, photos, markers)["n_photos"] == 2


def test_number_set_handles_blanks_and_floats():
    assert number_set("") == set() and number_set(None) == set()
    assert number_set("3; 7.0,12") == {"3", "7", "12"}
