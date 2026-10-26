"""Acceptance tests AT-G-10 to AT-G-16 (SPEC 4.7, D-026): AI-labelled, human-checked gold."""

import pandas as pd
import pytest

from ffs import gold

CATS = ["blade", "bottle_glass"]
K = ["scene_id", "photo_id"]


def photos_split():
    photos = pd.DataFrame({"scene_id": ["s1", "s1", "s2", "s2", "s3"], "photo_id": list("abcde"),
                           "relpath": [f"{p}.jpg" for p in "abcde"]})
    split = pd.DataFrame({"scene_id": ["s1", "s2", "s3"], "split": ["test", "test", "dev"]})
    return photos, split


def ai_table(rows):
    """rows: {(scene, photo): ((blade, bottle), confidence)}"""
    out = []
    for (s, p), (vals, conf) in rows.items():
        for c, v in zip(CATS, vals):
            out.append((s, p, c, v, conf, "n", "m", "2026-10-27T00:00:00Z"))
    return pd.DataFrame(out, columns=gold.AI_COLUMNS)


def silver_table(rows):
    return pd.DataFrame([(s, p, c, v, "") for (s, p), vals in rows.items() for c, v in zip(CATS, vals)],
                        columns=["scene_id", "photo_id", "category", "present", "marker_nos"])


AI = ai_table({("s1", "a"): ((1, 0), "high"), ("s1", "b"): ((0, 0), "high"),
               ("s2", "c"): ((0, 1), "low"), ("s2", "d"): ((0, 0), "high")})
SILVER = silver_table({("s1", "a"): (1, 0), ("s1", "b"): (1, 0), ("s2", "c"): (0, 1), ("s2", "d"): (0, 0),
                       ("s3", "e"): (0, 0)})


def test_at_g_10_review_pairs_are_disagreements_and_low_confidence():
    r = gold.review_pairs(AI, SILVER)
    got = {(a, b, c): reason for a, b, c, reason in zip(r.scene_id, r.photo_id, r.category, r.reason)}
    assert got == {("s1", "b", "blade"): "disagree",          # AI 0, silver 1
                   ("s2", "c", "blade"): "low_conf", ("s2", "c", "bottle_glass"): "low_conf"}


def test_at_g_11_audit_photos_come_only_from_fully_agreed_photos_and_are_deterministic():
    r = gold.review_pairs(AI, SILVER)
    a1 = gold.audit_photos(AI, r, 5, seed=42)
    assert a1 == [("s1", "a"), ("s2", "d")]                   # b and c are in the review set
    assert gold.audit_photos(AI, r, 1, seed=42) == gold.audit_photos(AI, r, 1, seed=42)


def test_at_g_12_upsert_replaces_a_photo_and_checks_input(tmp_path):
    p = tmp_path / "ai.csv"
    gold.upsert_ai_labels(p, "s1", "a", {"blade": 1, "bottle_glass": 0}, "med", "knife, on sofa", "model-x", CATS)
    gold.upsert_ai_labels(p, "s1", "a", {"blade": 0, "bottle_glass": 0}, "high", "none", "model-x", CATS)
    df = pd.read_csv(p)
    assert len(df) == 2 and df[df.category == "blade"].present.iloc[0] == 0
    with pytest.raises(ValueError):
        gold.upsert_ai_labels(p, "s1", "a", {"blade": 1}, "med", "", "m", CATS)
    with pytest.raises(ValueError):
        gold.upsert_ai_labels(p, "s1", "a", {"blade": 1, "bottle_glass": 0}, "sure", "", "m", CATS)


def test_at_g_13_tool_data_for_pairs_shows_only_the_listed_categories():
    photos, _ = photos_split()
    ex = pd.DataFrame({"scene_id": ["s1"], "marker_no": [1], "description": ["x"]})
    d = gold.tool_data_for_pairs(photos, ex, CATS, {("s2", "c"): ["bottle_glass"], ("s1", "b"): CATS}, "k", "e.csv")
    by = {(p["scene"], p["photo"]): p["show"] for p in d["photos"]}
    assert by == {("s2", "c"): [1], ("s1", "b"): [0, 1]}
    assert "ai" not in str(d["photos"]).lower().replace("photo", "")  # no AI labels leak into the page


def human(rows):
    return pd.DataFrame([(s, p, c, v, "2026-10-27T10:00:00Z") for s, p, c, v in rows],
                        columns=["scene_id", "photo_id", "category", "present", "labelled_at"])


def test_at_g_14_merge_human_wins_and_sources_are_recorded():
    photos, split = photos_split()
    review = gold.review_pairs(AI, SILVER)
    audit = [("s1", "a")]
    rev = human([("s1", "b", "blade", 1), ("s2", "c", "blade", 0), ("s2", "c", "bottle_glass", 0)])
    aud = human([("s1", "a", "blade", 1), ("s1", "a", "bottle_glass", 0)])
    g, src = gold.merge_gold(AI, SILVER, review, audit, rev, aud, photos, split, CATS)
    assert len(g) == 4 * 2                                     # 4 test photos x 2 categories
    row = lambda p, c: g[(g.photo_id == p) & (g.category == c)].iloc[0]
    assert row("b", "blade").present == 1 and row("b", "blade").corrected == 0   # human sided with silver
    assert row("c", "bottle_glass").present == 0 and row("c", "bottle_glass").corrected == 1
    s = src.set_index(["photo_id", "category"]).source
    assert s[("b", "blade")] == "human_review" and s[("a", "blade")] == "human_audit" and s[("d", "blade")] == "ai"


def test_at_g_15_merge_refuses_missing_human_decisions_unless_partial():
    photos, split = photos_split()
    review = gold.review_pairs(AI, SILVER)
    with pytest.raises(ValueError):
        gold.merge_gold(AI, SILVER, review, [("s1", "a")], None, None, photos, split, CATS)
    g, _ = gold.merge_gold(AI, SILVER, review, [("s1", "a")], None, None, photos, split, CATS, allow_partial=True)
    assert len(g) == 8
    with pytest.raises(ValueError):                             # a test photo the AI never labelled
        gold.merge_gold(AI[AI.photo_id != "d"], SILVER, review, [], None, None, photos, split, CATS, allow_partial=True)


def test_at_g_16_check_stats_measure_ai_error_on_audit_and_who_was_right_in_review():
    photos, split = photos_split()
    review = gold.review_pairs(AI, SILVER)
    rev = human([("s1", "b", "blade", 0), ("s2", "c", "blade", 0), ("s2", "c", "bottle_glass", 1)])
    aud = human([("s1", "a", "blade", 1), ("s1", "a", "bottle_glass", 1)])    # AI said bottle 0: AI wrong
    _, src = gold.merge_gold(AI, SILVER, review, [("s1", "a")], rev, aud, photos, split, CATS)
    st = gold.check_stats(src)
    a = st["audit"].set_index("category")
    assert a.loc["ALL", "ai_vs_human_agreement"] == pytest.approx(0.5)
    r = st["review"].set_index("category").loc["blade"]
    assert r.human_sided_with_ai == pytest.approx(1.0) and r.human_sided_with_silver == pytest.approx(0.5)
    assert 0 < st["human_share"] < 1
