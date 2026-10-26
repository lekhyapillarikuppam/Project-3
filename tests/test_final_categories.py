"""D-020 and R-S-7: the final categories, and that the committed silver.csv is consistent with them."""

import pandas as pd

from ffs import config
from ffs.schemas import validate
from ffs.silver import category_counts


def load(name):
    return pd.read_csv(config.META_DIR / name)


def test_final_categories_are_candidates_and_never_other():
    assert config.FINAL_CATEGORIES and set(config.FINAL_CATEGORIES) <= set(config.CANDIDATE_CATEGORIES)
    assert config.OTHER not in config.FINAL_CATEGORIES
    assert 4 <= len(config.FINAL_CATEGORIES) <= 8                    # R-P-3 needs at least 4


def test_silver_csv_is_a_full_matrix_of_the_final_categories_only():
    silver, photos = load("silver.csv"), load("photos.csv")
    validate(silver, "silver")
    assert set(silver.category) == set(config.FINAL_CATEGORIES)
    assert len(silver) == len(photos) * len(config.FINAL_CATEGORIES)


def test_every_final_category_passes_the_inclusion_rule_on_test():
    table = category_counts(load("silver.csv"), load("scene_split.csv")).set_index("category")
    assert table.passes_test.all(), table[~table.passes_test]


def test_scene_split_matches_the_seed():
    from ffs.dataprep import split_scenes

    split = load("scene_split.csv")
    assert dict(zip(split.scene_id, split.split)) == split_scenes(split.scene_id)    # never re-drawn (D-017)
