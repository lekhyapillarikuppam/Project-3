"""AT-E-1..8: test-set evaluation (SPEC section 6)."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ffs import config, evaluation as ev

CATS = list(config.FINAL_CATEGORIES)


def _load(name):
    path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_at_e_1_average_precision_known_values():
    y = np.array([1, 0, 1, 0])
    # ranking 1,0,1,0: precision at the two hits = 1 and 2/3, recall steps 0.5 each -> AP = 0.5*1 + 0.5*2/3
    assert ev.average_precision(y, np.array([0.9, 0.8, 0.7, 0.1])) == pytest.approx(0.5 + 0.5 * 2 / 3)
    assert ev.average_precision(np.array([1, 1, 0, 0]), np.array([0.9, 0.8, 0.2, 0.1])) == pytest.approx(1.0)
    assert np.isnan(ev.average_precision(np.array([0, 0]), np.array([0.5, 0.4])))
    # all scores tied: one step, AP = prevalence
    assert ev.average_precision(np.array([1, 0, 0, 0]), np.zeros(4)) == pytest.approx(0.25)


def test_at_e_2_precision_recall_f1_and_kappa():
    y = np.array([1, 1, 0, 0, 1])
    p = np.array([1, 0, 1, 0, 1])
    pr, rc, f1 = ev.prf(y, p)
    assert (pr, rc) == (pytest.approx(2 / 3), pytest.approx(2 / 3)) and f1 == pytest.approx(2 / 3)
    assert ev.prf(np.zeros(3, int), np.zeros(3, int)) == (0.0, 0.0, 0.0)          # no division by zero
    assert ev.cohens_kappa(y, y) == pytest.approx(1.0)


def test_at_e_3_clip_score_is_softmax_of_mean_prompt_logit():
    photos = pd.DataFrame({"scene_id": ["s"], "photo_id": ["a"]})
    rows = [("s", "a", c, f"p{i}", v) for c, v in zip(CATS, [10, 0, 0, 0, 0]) for i in range(3)]
    clip = pd.DataFrame(rows, columns=["scene_id", "photo_id", "category", "prompt", "score"])
    out = ev.clip_photo_scores(clip, photos, CATS)
    assert out["clip_score"].sum() == pytest.approx(1.0)
    assert out.set_index("category").loc[CATS[0], "clip_score"] > 0.99
    with pytest.raises(ValueError):
        ev.clip_photo_scores(clip[clip.category != CATS[-1]], photos, CATS)        # a missing category is an error


def test_at_e_4_owl_score_is_best_box_on_tile0_and_zero_without_box():
    photos = pd.DataFrame({"scene_id": ["s", "s"], "photo_id": ["a", "b"]})
    owl = pd.DataFrame([("s", "a", CATS[0], "q1", 0.3, 1, 2, 3, 4, 0), ("s", "a", CATS[0], "q2", 0.6, 5, 6, 7, 8, 0),
                        ("s", "a", CATS[0], "q3", 0.9, 0, 0, 1, 1, 2)],           # tile 2 must be ignored
                       columns=["scene_id", "photo_id", "category", "prompt", "score", "x", "y", "w", "h", "tile"])
    out = ev.owl_photo_scores(owl, photos, CATS, tile=0).set_index(["photo_id", "category"])
    assert out.loc[("a", CATS[0]), "owl_score"] == pytest.approx(0.6) and out.loc[("a", CATS[0]), "box_x"] == 5
    assert out.loc[("b", CATS[0]), "owl_score"] == 0.0 and len(out) == 2 * len(CATS)


@pytest.mark.parametrize("layout", [
    lambda t: {"clip": {"thresholds": t[0]}, "owlv2": {"thresholds": t[1]}},
    lambda t: {"thresholds": {"clip": t[0], "owlv2": t[1]}},
    lambda t: {"models": {"CLIP": {"per_category": t[0]}, "OWLv2": {"per_category": t[1]}}},
    lambda t: {"thresholds": {c: {"clip": t[0][c], "owlv2": t[1][c]} for c in CATS}},
])
def test_at_e_5_thresholds_are_found_in_common_protocol_layouts(layout):
    a = {c: 0.1 * (i + 1) for i, c in enumerate(CATS)}
    b = {c: 0.5 + 0.01 * i for i, c in enumerate(CATS)}
    proto = layout((a, b))
    assert ev.find_thresholds(proto, "clip", CATS) == a
    assert ev.find_thresholds(proto, "owl", CATS) == b
    with pytest.raises(ValueError):
        ev.find_thresholds({"x": 1}, "clip", CATS)


def _frame(n_scenes=12, per=6, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_scenes):
        for p in range(per):
            for c in CATS:
                y = int(rng.random() < 0.3)
                good = y * 0.6 + rng.random() * 0.4                      # informative score
                noise = rng.random()                                     # uninformative score
                rows.append((f"s{s}", f"p{p}", c, y, good, int(good > 0.5), noise, int(noise > 0.5)))
    return pd.DataFrame(rows, columns=["scene_id", "photo_id", "category", "y", "good", "good_pred", "bad", "bad_pred"])


def test_at_e_6_bootstrap_is_reproducible_and_ci_contains_point_and_good_beats_bad():
    d = _frame()
    systems = {"good": ("good", "good_pred"), "bad": ("bad", "bad_pred")}
    m1, p1 = ev.evaluate(d, systems, CATS, B=100, seed=42, pairs=[("good", "bad")])
    m2, p2 = ev.evaluate(d, systems, CATS, B=100, seed=42, pairs=[("good", "bad")])
    pd.testing.assert_frame_equal(m1, m2)
    pd.testing.assert_frame_equal(p1, p2)
    c = m1[m1.category != "macro"]
    assert (c.f1_lo <= c.f1_hi).all() and (c.f1 >= 0).all() and (c.f1 <= 1).all()
    macro = m1[m1.category == "macro"].set_index("system")
    assert macro.loc["good", "f1"] > macro.loc["bad", "f1"] and macro.loc["good", "ap"] > macro.loc["bad", "ap"]
    pm = p1[(p1.category == "macro") & (p1.metric == "ap")].iloc[0]
    assert pm["diff"] > 0 and pm["lo"] > 0 and pm["frac_le_0"] == 0


def test_at_e_7_bootstrap_resamples_whole_scenes():
    d = _frame(n_scenes=2, per=3)
    # with 2 scenes the resample can only be {s0,s0}, {s0,s1}, {s1,s1}: the macro F1 bootstrap draws take at most 3 distinct values
    systems = {"good": ("good", "good_pred")}
    m, _ = ev.evaluate(d, systems, CATS, B=50, seed=1)
    assert len(m) == len(CATS) + 1 and set(m.system) == {"good"}


def test_at_e_8_evaluate_test_script_end_to_end(tmp_path, monkeypatch, capsys):
    meta, res = tmp_path / "data_meta", tmp_path / "results"
    meta.mkdir()
    res.mkdir()
    (tmp_path / "protocol").mkdir()
    rng = np.random.default_rng(3)
    photos = [(f"s{s}", f"p{p}") for s in range(8) for p in range(5)]
    pd.DataFrame({"scene_id": [a for a, _ in photos], "photo_id": [b for _, b in photos],
                  "relpath": "x.jpg", "width": 10, "height": 10, "photo_type": "close"}).to_csv(meta / "photos.csv", index=False)
    pd.DataFrame({"scene_id": [f"s{s}" for s in range(8)], "split": "test", "has_markers": 1, "n_photos": 5}).to_csv(
        meta / "scene_split.csv", index=False)
    truth = {(a, b, c): int(rng.random() < 0.3) for a, b in photos for c in CATS}
    pd.DataFrame([(a, b, c, v, 0, "t") for (a, b, c), v in truth.items()],
                 columns=["scene_id", "photo_id", "category", "present", "corrected", "labelled_at"]).to_csv(meta / "gold.csv", index=False)
    pd.DataFrame([(a, b, c, v if rng.random() < 0.8 else 1 - v, "") for (a, b, c), v in truth.items()],
                 columns=["scene_id", "photo_id", "category", "present", "marker_nos"]).to_csv(meta / "silver.csv", index=False)
    pd.DataFrame([(a, b, c, f"p{i}", 10 * truth[(a, b, c)] + rng.random()) for a, b in photos for c in CATS for i in range(3)],
                 columns=["scene_id", "photo_id", "category", "prompt", "score"]).to_csv(res / "clip_scores_test.csv", index=False)
    pd.DataFrame([(a, b, c, "q", 0.1 + 0.6 * truth[(a, b, c)] * rng.random(), 1, 2, 3, 4, 0) for a, b in photos for c in CATS],
                 columns=["scene_id", "photo_id", "category", "prompt", "score", "x", "y", "w", "h", "tile"]).to_csv(res / "owlv2_raw_test.csv", index=False)
    proto = {"thresholds": {"clip": {c: 0.3 for c in CATS}, "owlv2": {c: 0.2 for c in CATS}}}
    (tmp_path / "protocol" / "protocol_v1.json").write_text(json.dumps(proto))
    for k, v in (("META_DIR", meta), ("RESULTS_DIR", res)):
        monkeypatch.setattr(config, k, v)
    mod = _load("evaluate_test")
    monkeypatch.setattr("sys.argv", ["evaluate_test.py", "--B", "30", "--no-tag-check"])
    mod.main()
    out = res / "eval"
    for f in ("test_metrics.csv", "test_paired.csv", "silver_vs_gold.csv", "photo_predictions.csv", "errors.csv", "summary.md"):
        assert (out / f).exists(), f
    pp = pd.read_csv(out / "photo_predictions.csv")
    assert len(pp) == len(photos) * len(CATS) and {"gold", "silver", "clip_pred", "owl_pred", "box_x"} <= set(pp.columns)
    m = pd.read_csv(out / "test_metrics.csv")
    assert set(m.reference) == {"gold", "silver"} and set(m.system) == {"CLIP", "OWLv2"}
    assert m[(m.reference == "gold") & (m.system == "CLIP") & (m.category == "macro")].f1.iloc[0] > 0.8
    # a missing gold row stops the script
    g = pd.read_csv(meta / "gold.csv").iloc[:-1]
    g.to_csv(meta / "gold.csv", index=False)
    with pytest.raises(SystemExit):
        mod.main()


def test_at_e_9_only_disagree_keeps_only_disagreements():
    mod = _load("make_gold_tool")
    r = pd.DataFrame({"scene_id": ["s"] * 3, "photo_id": ["a", "b", "c"], "category": ["x"] * 3,
                      "reason": ["disagree", "low_conf", "disagree"]})
    out = mod.only_disagree(r)
    assert list(out.photo_id) == ["a", "c"] and list(out.index) == [0, 1]
