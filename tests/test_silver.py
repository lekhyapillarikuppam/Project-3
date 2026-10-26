"""Acceptance tests AT-S-1 to AT-S-6 (SPEC 4.5): silver labels."""

import pandas as pd

from ffs import silver
from ffs.schemas import validate

CATEGORIES = ["blade", "bottle_glass", "firearm"]


def exhibit_cats(rows):
    """rows: (scene, marker_no, category[, source, confirmed])."""
    out = [(r + ("rule", 1))[:5] for r in rows]
    return pd.DataFrame(out, columns=["scene_id", "marker_no", "category", "source", "confirmed"])


def markers(rows):
    """rows: (scene, photo, marker_no, status)."""
    return pd.DataFrame([{"scene_id": s, "photo_id": p, "marker_idx": i, "x": 0, "y": 0, "w": 10, "h": 10,
                          "method": "color_shape", "det_score": 0.9, "raw_ocr": str(n or ""), "ocr_conf": 1.0,
                          "marker_no": n if n is not None else "", "status": st}
                         for i, (s, p, n, st) in enumerate(rows)])


def photos(*ids):
    return pd.DataFrame({"scene_id": ["s1"] * len(ids), "photo_id": list(ids)})


def row(df, photo, cat):
    return df[(df.photo_id == photo) & (df.category == cat)].iloc[0]


def test_at_s_1_accepted_marker_sets_its_category_only():
    out = silver.build_silver(photos("p1"), markers([("s1", "p1", 7, "accepted")]),
                              exhibit_cats([("s1", 7, "blade")]), CATEGORIES)
    assert row(out, "p1", "blade").present == 1 and row(out, "p1", "blade").marker_nos == "7"
    assert out[out.category != "blade"].present.sum() == 0


def test_at_s_2_rejected_box_adds_no_label():
    out = silver.build_silver(photos("p1"), markers([("s1", "p1", 7, "rejected")]),
                              exhibit_cats([("s1", 7, "blade")]), CATEGORIES)
    assert out.present.sum() == 0


def test_at_s_3_repeated_number_sets_all_its_categories():
    out = silver.build_silver(photos("p1"), markers([("s1", "p1", 15, "snapped")]),
                              exhibit_cats([("s1", 15, "blade"), ("s1", 15, "bottle_glass")]), CATEGORIES)
    assert row(out, "p1", "blade").present == 1 and row(out, "p1", "bottle_glass").present == 1
    assert row(out, "p1", "firearm").present == 0


def test_at_s_4_full_matrix_that_passes_the_schema():
    ph = photos("p1", "p2", "p3")
    out = silver.build_silver(ph, markers([("s1", "p1", 7, "accepted")]),
                              exhibit_cats([("s1", 7, "blade")]), CATEGORIES)
    assert len(out) == len(ph) * len(CATEGORIES)
    validate(out, "silver")


def test_at_s_5_number_not_in_the_scene_list_is_ignored():
    out = silver.build_silver(photos("p1"), markers([("s1", "p1", 9, "accepted")]),
                              exhibit_cats([("s1", 7, "blade")]), CATEGORIES)
    assert out.present.sum() == 0


def test_at_s_6_uncertain_photos_come_from_rejected_boxes():
    m = markers([("s1", "p2", 7, "rejected")])                          # p1 has no markers at all
    out = silver.build_silver(photos("p1", "p2"), m, exhibit_cats([("s1", 7, "blade")]), CATEGORIES)
    assert out[out.photo_id == "p1"].present.sum() == 0
    unc = silver.uncertain_photos(m)
    assert ("s1", "p2") in unc and ("s1", "p1") not in unc


def test_unconfirmed_llm_rows_do_not_count():
    cats = exhibit_cats([("s1", 7, "blade", "llm", 0)])
    out = silver.build_silver(photos("p1"), markers([("s1", "p1", 7, "accepted")]), cats, CATEGORIES)
    assert out.present.sum() == 0
    cats = exhibit_cats([("s1", 7, "blade", "llm", 1)])                 # once the user confirms it, it counts
    out = silver.build_silver(photos("p1"), markers([("s1", "p1", 7, "accepted")]), cats, CATEGORIES)
    assert row(out, "p1", "blade").present == 1
