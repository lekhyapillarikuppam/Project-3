import pandas as pd
import pytest

from ffs import config
from ffs.schemas import COLUMNS, SchemaError, validate


def make(name, **overrides):
    """One valid row for a contract, with optional value overrides."""
    row = {c: 0 for c in COLUMNS[name]}
    row.update(overrides)
    return pd.DataFrame([row])


def test_valid_silver_passes():
    df = make("silver", scene_id="th_0001", photo_id="IMG_0001", category="blade",
              present=1, marker_nos="7")
    validate(df, "silver")


def test_missing_column_raises():
    df = make("silver").drop(columns=["marker_nos"])
    with pytest.raises(SchemaError, match="missing"):
        validate(df, "silver")


def test_extra_column_raises():
    df = make("silver")
    df["oops"] = 1
    with pytest.raises(SchemaError, match="unexpected"):
        validate(df, "silver")


def test_bad_split_raises():
    df = make("scene_split", split="train")
    with pytest.raises(SchemaError, match="split"):
        validate(df, "scene_split")


def test_present_must_be_binary():
    df = make("gold", present=2)
    with pytest.raises(SchemaError, match="present"):
        validate(df, "gold")


def test_unknown_contract_raises():
    with pytest.raises(SchemaError, match="Unknown"):
        validate(pd.DataFrame(), "nope")


def test_config_matches_spec():
    assert config.MIN_POSITIVE_PHOTOS == 15
    assert config.MIN_POSITIVE_SCENES == 6
    assert config.DEV_FRACTION == 0.4
    assert set(config.SPLITS) == {"dev", "test"}
    assert config.OTHER not in config.CANDIDATE_CATEGORIES
    assert len(config.CANDIDATE_CATEGORIES) == 8
