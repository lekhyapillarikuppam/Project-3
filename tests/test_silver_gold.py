"""Acceptance tests AT-G-7 to AT-G-9 (SPEC 4.7, R-G-7): silver against gold, with a scene-level bootstrap."""

import numpy as np
import pandas as pd
import pytest

from ffs import gold, silver_gold

KEYS = ["scene_id", "photo_id"]


def test_at_g_7_hand_computed_example():
    # 10 photos, one category. gold = 4 positives; silver finds 3 of them and adds 1 false positive.
    g = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
    s = [1, 1, 1, 0, 1, 0, 0, 0, 0, 0]
    tp, fp, fn, tn = silver_gold.counts(g, s)
    assert (tp, fp, fn, tn) == (3, 1, 1, 5)
    m = silver_gold.metrics(tp, fp, fn, tn)
    assert m["precision"] == pytest.approx(3 / 4)
    assert m["recall"] == pytest.approx(3 / 4)
    assert m["f1"] == pytest.approx(3 / 4)
    # po = 0.8, pe = 0.4*0.4 + 0.6*0.6 = 0.52, kappa = 0.28 / 0.48
    assert m["kappa"] == pytest.approx(0.28 / 0.48)
    assert m["kappa"] == pytest.approx(gold.cohens_kappa(g, s))        # same number as the project's kappa


def test_metrics_are_nan_when_undefined():
    m = silver_gold.metrics(0, 0, 0, 5)                                # silver and gold both say "absent" everywhere
    assert np.isnan(m["precision"]) and np.isnan(m["recall"]) and np.isnan(m["f1"]) and np.isnan(m["kappa"])


def tables():
    """4 test scenes, 12 photos, 2 categories, plus a dev scene that must be ignored."""
    rows = [("s1", "a"), ("s1", "b"), ("s1", "c"), ("s2", "d"), ("s2", "e"), ("s2", "f"),
            ("s3", "g"), ("s3", "h"), ("s3", "i"), ("s4", "j"), ("s4", "k"), ("s4", "l"), ("s9", "z")]
    photos = pd.DataFrame(rows, columns=KEYS)
    photos["photo_type"] = ["close", "mid", "mid", "overview", "mid", "close", "overview", "unknown", "mid",
                            "close", "overview", "mid", "mid"]
    split = pd.DataFrame({"scene_id": ["s1", "s2", "s3", "s4", "s9"], "split": ["test"] * 4 + ["dev"]})
    g_blade = [1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1]
    s_blade = [1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    g_bottle = [0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0]
    s_bottle = [0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1]
    mk = lambda cat, vals: pd.DataFrame({"scene_id": photos.scene_id, "photo_id": photos.photo_id,
                                         "category": cat, "present": vals})
    gold_df = pd.concat([mk("blade", g_blade), mk("bottle_glass", g_bottle)], ignore_index=True)
    gold_df = gold_df[gold_df.scene_id != "s9"].assign(corrected=0, labelled_at="t")      # gold has test scenes only
    silver_df = pd.concat([mk("blade", s_blade), mk("bottle_glass", s_bottle)], ignore_index=True).assign(marker_nos="")
    markers = pd.DataFrame({
        "scene_id": ["s1", "s1", "s2", "s3", "s4", "s4"], "photo_id": ["a", "b", "d", "g", "j", "k"],
        "status": ["accepted", "rejected", "snapped", "rejected", "accepted", "accepted"]})
    return gold_df, silver_df, split, photos, markers


def test_at_g_8_bootstrap_is_deterministic_and_resamples_whole_scenes():
    a = silver_gold.draw_scenes(4, 50, seed=42)
    assert np.array_equal(a, silver_gold.draw_scenes(4, 50, seed=42))
    assert not np.array_equal(a, silver_gold.draw_scenes(4, 50, seed=43))
    assert a.shape == (50, 4) and a.min() >= 0 and a.max() <= 3

    # brute force: build each resample by pasting WHOLE scenes together, and compare with the count-based result
    gold_df, silver_df, split, photos, markers = tables()
    df = silver_gold.paired_labels(gold_df, silver_df, split, photos, markers)
    d = df[df.category == "blade"]
    scenes = sorted(d.scene_id.unique())
    per_scene = np.array([silver_gold.counts(d[d.scene_id == s].gold, d[d.scene_id == s].silver) for s in scenes])
    idx = silver_gold.draw_scenes(len(scenes), 30, seed=7)
    fast = silver_gold.bootstrap_stats(per_scene, idx)
    for b in range(30):
        pasted = pd.concat([d[d.scene_id == scenes[i]] for i in idx[b]])
        slow = silver_gold.metrics(*silver_gold.counts(pasted.gold, pasted.silver))
        for name in ("precision", "recall", "f1", "kappa"):
            assert fast[name][b] == pytest.approx(slow[name], nan_ok=True)


def test_one_scene_gives_an_interval_that_collapses_to_the_estimate():
    gold_df, silver_df, split, photos, markers = tables()
    one = photos.assign(scene_id="s1")                                  # every photo in the same scene
    gold_one = gold_df.assign(scene_id="s1")
    silver_one = silver_df.assign(scene_id="s1")
    split_one = pd.DataFrame({"scene_id": ["s1"], "split": ["test"]})
    markers_one = markers.assign(scene_id="s1")
    t = silver_gold.silver_vs_gold(gold_one, silver_one, split_one, one, markers_one, n_boot=100, seed=1)
    row = t[(t.slice == "overall") & (t.category == "blade")].iloc[0]
    assert row.precision_lo == pytest.approx(row.precision) == pytest.approx(row.precision_hi)
    assert row.kappa_lo == pytest.approx(row.kappa) == pytest.approx(row.kappa_hi)


def test_at_g_9_slice_rows_add_up_to_the_overall_counts():
    gold_df, silver_df, split, photos, markers = tables()
    t = silver_gold.silver_vs_gold(gold_df, silver_df, split, photos, markers, n_boot=20, seed=1)
    assert set(t.slice) == {"overall", "photo_type", "certainty", "counted_marker"}
    for cat in ("blade", "bottle_glass"):
        overall = t[(t.slice == "overall") & (t.category == cat)][["TP", "FP", "FN", "TN"]].iloc[0]
        for sl in ("photo_type", "certainty", "counted_marker"):
            part = t[(t.slice == sl) & (t.category == cat)][["TP", "FP", "FN", "TN"]].sum()
            assert part.tolist() == overall.tolist(), (sl, cat)
    assert t[t.slice == "overall"].n_photos.tolist() == [12, 12]          # dev photos are not in the report
    assert set(t[t.slice == "certainty"].value) == {"certain", "uncertain"}
    assert set(t[t.slice == "counted_marker"].value) == {"has counted marker", "no counted marker"}


def test_gold_prevalence_and_overall_numbers():
    gold_df, silver_df, split, photos, markers = tables()
    t = silver_gold.silver_vs_gold(gold_df, silver_df, split, photos, markers, n_boot=20, seed=1)
    row = t[(t.slice == "overall") & (t.category == "blade")].iloc[0]
    # test photos a..l: gold blade = 1,0,1,1,0,0,1,0,0,0,1,1 -> 6 of 12
    assert row.gold_prevalence == pytest.approx(0.5)
    assert (row.TP, row.FP, row.FN, row.TN) == (4, 2, 2, 4)
