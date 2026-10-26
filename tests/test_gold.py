"""Acceptance tests AT-G-1 to AT-G-6 (SPEC 4.5b): gold labels."""

import pandas as pd
import pytest

from ffs import gold

CATS = ["blade", "bottle_glass"]


def tables():
    photos = pd.DataFrame({"scene_id": ["th_0001", "th_0001", "th_0002", "th_0003"],
                           "photo_id": ["a", "b", "c", "d"],
                           "relpath": [f"x/{p}.jpg" for p in "abcd"]})
    split = pd.DataFrame({"scene_id": ["th_0001", "th_0002", "th_0003"], "split": ["test", "test", "dev"]})
    exhibits = pd.DataFrame({"scene_id": ["th_0001", "th_0001", "th_0003"], "marker_no": [7, 3, 9],
                             "description": ["knife", "bottle", "paper"]})
    silver = pd.DataFrame([(s, p, c, v) for (s, p), vals in
                           {("th_0001", "a"): (1, 0), ("th_0001", "b"): (0, 0), ("th_0002", "c"): (0, 1)}.items()
                           for c, v in zip(CATS, vals)],
                          columns=["scene_id", "photo_id", "category", "present"])
    silver["marker_nos"] = ""
    return photos, split, exhibits, silver


def export_rows(labels):
    return pd.DataFrame([(s, p, c, v, "2026-10-27T10:00:00Z") for (s, p), vals in labels.items()
                         for c, v in zip(CATS, vals)],
                        columns=["scene_id", "photo_id", "category", "present", "labelled_at"])


def test_at_g_1_tool_data_has_test_photos_only_and_no_silver():
    photos, split, exhibits, _ = tables()
    d = gold.build_tool_data(photos, split, exhibits, CATS)
    assert [(p["scene"], p["photo"]) for p in d["photos"]] == [("th_0001", "a"), ("th_0001", "b"), ("th_0002", "c")]
    assert d["cats"] == CATS and "silver" not in str(d).lower()
    assert list(d["exhibits"]) == ["th_0001"]               # th_0003 is dev: not included
    assert d["exhibits"]["th_0001"] == [[3, "bottle"], [7, "knife"]]


def test_at_g_2_pass2_sample_is_deterministic_and_a_subset():
    photos, split, exhibits, _ = tables()
    d1 = gold.build_tool_data(photos, split, exhibits, CATS, sample=2, seed=42)
    d2 = gold.build_tool_data(photos, split, exhibits, CATS, sample=2, seed=42)
    assert d1["photos"] == d2["photos"] and len(d1["photos"]) == 2


def test_at_g_3_import_makes_contract_and_marks_corrections():
    photos, split, _, silver = tables()
    exp = export_rows({("th_0001", "a"): (1, 0), ("th_0001", "b"): (1, 0), ("th_0002", "c"): (0, 0)})
    out = gold.gold_from_export(exp, photos, split, silver, CATS)
    assert len(out) == 3 * 2
    row = lambda p, c: out[(out.photo_id == p) & (out.category == c)].iloc[0]
    assert row("a", "blade").corrected == 0                  # gold 1, silver 1
    assert row("b", "blade").corrected == 1                  # gold 1, silver 0: silver missed it
    assert row("c", "bottle_glass").corrected == 1           # gold 0, silver 1: silver false positive


def test_at_g_4_incomplete_labels_are_refused_unless_partial():
    photos, split, _, silver = tables()
    exp = export_rows({("th_0001", "a"): (1, 0)})
    with pytest.raises(ValueError):
        gold.gold_from_export(exp, photos, split, silver, CATS)
    part = gold.gold_from_export(exp, photos, split, silver, CATS, allow_partial=True)
    assert len(part) == 2


def test_at_g_5_kappa():
    assert gold.cohens_kappa([1, 0, 1, 0], [1, 0, 1, 0]) == pytest.approx(1.0)
    assert gold.cohens_kappa([1, 1, 0, 0], [0, 0, 1, 1]) == pytest.approx(-1.0)
    assert gold.cohens_kappa([1, 0, 1, 0], [1, 0, 0, 1]) == pytest.approx(0.0)
    assert pd.isna(gold.cohens_kappa([0, 0], [0, 0]))


def test_at_g_6_intra_rater_agreement():
    keys = ["scene_id", "photo_id", "category"]
    a = pd.DataFrame([("s", "1", "blade", 1), ("s", "2", "blade", 0), ("s", "3", "blade", 1)], columns=keys + ["present"])
    b = pd.DataFrame([("s", "1", "blade", 1), ("s", "2", "blade", 0), ("s", "3", "blade", 0)], columns=keys + ["present"])
    r = gold.intra_rater_agreement(a, b)
    assert r.iloc[0].photos == 3 and r.iloc[0].agreement == pytest.approx(2 / 3)
