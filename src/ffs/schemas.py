"""Data contracts from SPEC.md section 2, as code.

Use `validate(df, "silver")` before writing or after reading any contract file.
"""

from __future__ import annotations

import pandas as pd

from . import config


class SchemaError(ValueError):
    """Raised when a DataFrame does not match its data contract."""


COLUMNS: dict[str, list[str]] = {
    "exhibits": ["scene_id", "marker_no", "description"],
    "category_rules": ["pattern", "category", "version"],
    "exhibit_categories": ["scene_id", "marker_no", "category", "source", "confirmed"],
    "scene_split": ["scene_id", "split", "has_markers", "n_photos"],
    "photos": ["scene_id", "photo_id", "relpath", "width", "height", "photo_type"],
    "markers": [
        "scene_id", "photo_id", "marker_idx", "x", "y", "w", "h", "method",
        "det_score", "raw_ocr", "ocr_conf", "marker_no", "status",
    ],
    "silver": ["scene_id", "photo_id", "category", "present", "marker_nos"],
    "gold": ["scene_id", "photo_id", "category", "present", "corrected", "labelled_at"],
    "clip_scores": ["scene_id", "photo_id", "category", "prompt", "score"],
    "owlv2_raw": ["scene_id", "photo_id", "category", "prompt", "score", "x", "y", "w", "h", "tile"],
}

# column name -> allowed values, per file (None means "any value")
ALLOWED: dict[str, dict[str, tuple]] = {
    "exhibit_categories": {"source": config.CATEGORY_SOURCES, "confirmed": (0, 1)},
    "scene_split": {"split": config.SPLITS, "has_markers": (0, 1)},
    "photos": {"photo_type": config.PHOTO_TYPES},
    "markers": {"method": config.MARKER_METHODS, "status": config.MARKER_STATUSES},
    "silver": {"present": (0, 1)},
    "gold": {"present": (0, 1), "corrected": (0, 1)},
}


def validate(df: pd.DataFrame, name: str) -> None:
    """Check exact column set and allowed values. Raises SchemaError."""
    if name not in COLUMNS:
        raise SchemaError(f"Unknown contract '{name}'. Known: {sorted(COLUMNS)}")

    expected = COLUMNS[name]
    got = list(df.columns)
    missing = [c for c in expected if c not in got]
    extra = [c for c in got if c not in expected]
    if missing or extra:
        raise SchemaError(f"{name}: missing columns {missing}, unexpected columns {extra}")

    for col, allowed in ALLOWED.get(name, {}).items():
        bad = sorted({v for v in df[col].dropna().unique() if v not in allowed})
        if bad:
            raise SchemaError(f"{name}.{col}: values {bad} not in {list(allowed)}")
