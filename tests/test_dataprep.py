"""Acceptance tests AT-P-1 to AT-P-5 (SPEC 4.6): data preparation."""

import pandas as pd

from ffs import config, dataprep


def make_rules():
    return pd.DataFrame({
        "pattern": [r"\b(knife|cutter|scissors)\b", r"\b(bottle|glass|vodka)\b", r"\b(mobile phone|laptop)\b"],
        "category": ["blade", "bottle_glass", "phone_electronics"],
        "version": [1, 1, 1]})


def test_at_p_1_rules_match_descriptions():
    rules = make_rules()
    assert dataprep.match_categories("knife with black handle", rules) == ["blade"]
    assert dataprep.match_categories("Half-empty bottle of vodka", rules) == ["bottle_glass"]   # case-insensitive
    assert dataprep.match_categories("mobile phone", rules) == ["phone_electronics"]
    assert dataprep.match_categories("xylophone", rules) == []


def test_at_p_2_split_is_by_scene_disjoint_deterministic_and_about_40_60():
    scenes = [f"th_{i:04d}" for i in range(2, 21)]
    a = dataprep.split_scenes(scenes)
    assert a == dataprep.split_scenes(list(reversed(scenes)))           # same seed, same split, any input order
    assert set(a) == set(scenes) and set(a.values()) == {"dev", "test"}  # every scene exactly once
    n_dev = sum(v == "dev" for v in a.values())
    assert n_dev == round(config.DEV_FRACTION * len(scenes))
    assert dataprep.split_scenes(scenes, seed=config.SEED + 1) != a     # the seed matters


def test_at_p_3_photo_type_thresholds():
    assert dataprep.photo_type(0.35) == "close"
    assert dataprep.photo_type(0.15) == "mid"
    assert dataprep.photo_type(0.03) == "overview"
    assert dataprep.photo_type(None) == "unknown"
    assert dataprep.photo_type(config.CLOSE_MIN_FRAC) == "close"        # the thresholds themselves count
    assert dataprep.photo_type(config.MID_MIN_FRAC) == "mid"


def test_photo_types_use_the_largest_box_of_each_photo():
    photos = pd.DataFrame({"scene_id": ["s"] * 3, "photo_id": ["a", "b", "c"], "height": [1000, 1000, 1000]})
    markers = pd.DataFrame({"scene_id": ["s", "s", "s"], "photo_id": ["a", "a", "b"], "h": [50, 400, 20]})
    types = dataprep.photo_types(photos, markers)
    assert types.tolist() == ["close", "overview", "unknown"]


def test_at_p_4_exhibits_drop_empty_marker_numbers_and_keep_repeats():
    raw = pd.DataFrame({"scene_id": ["s1"] * 4, "marker_no": [11, None, 15, 15],
                        "description": ["a", "no number", "knife", "bottle"]})
    clean, n_dropped = dataprep.clean_exhibits(raw)
    assert n_dropped == 1
    assert clean["marker_no"].tolist() == [11, 15, 15]
    assert clean["marker_no"].dtype.kind == "i"


def test_at_p_5_finished_scenes_are_skipped(tmp_path):
    (tmp_path / "th_0002.csv").write_text("scene_id\n")                 # done, even with zero markers
    assert dataprep.pending_scenes(["th_0002", "th_0003"], tmp_path) == ["th_0003"]


def test_image_size_swaps_when_exif_says_turned(tmp_path):
    from PIL import Image

    plain, turned = tmp_path / "a.jpg", tmp_path / "b.jpg"
    Image.new("RGB", (40, 20)).save(plain)
    exif = Image.Exif()
    exif[0x0112] = 6                                       # "turn 90 degrees when displaying"
    Image.new("RGB", (40, 20)).save(turned, exif=exif)
    assert dataprep.image_size(plain) == (40, 20)
    assert dataprep.image_size(turned) == (20, 40)

def test_at_p_6_rules_are_multi_label_and_other_rules_are_exclusive():
    rules = pd.concat([pd.DataFrame({"pattern": [r"\bglass table\b"], "category": ["other"], "version": [1]}),
                       make_rules(),
                       pd.DataFrame({"pattern": [r"reddis(h|ch) trace"], "category": ["red_brown_stain"],
                                     "version": [1]})], ignore_index=True)       # same order as category_rules.csv
    assert dataprep.match_categories("Knife with reddish trace", rules) == ["blade", "red_brown_stain"]
    assert dataprep.match_categories("Glass table", rules) == ["other"]       # "glass" alone would be bottle_glass
    assert dataprep.match_categories("Glass table with reddish trace", rules) == ["other"]


def test_rebuilding_exhibit_categories_keeps_confirmed_rows_and_adds_new_suggestions():
    exhibits = pd.DataFrame({"scene_id": ["s1", "s1", "s1"], "marker_no": [1, 2, 3],
                             "description": ["knife", "cans", "vase"]})
    existing = pd.DataFrame({"scene_id": ["s1"], "marker_no": [2], "category": ["other"],
                             "source": ["manual"], "confirmed": [1]})
    out = dataprep.build_exhibit_categories(exhibits, make_rules(), lambda d: "bottle_glass", existing)
    assert out.values.tolist() == [["s1", 1, "blade", "rule", 1],
                                   ["s1", 2, "other", "manual", 1],             # kept, not re-suggested
                                   ["s1", 3, "bottle_glass", "llm", 0]]         # new: unconfirmed
