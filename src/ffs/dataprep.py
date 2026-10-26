"""Data preparation (SPEC 4.6): exhibit table, category rules, scene split, photo types."""

from __future__ import annotations

import random
import re
from pathlib import Path

import pandas as pd

from . import config


def clean_exhibits(raw: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Drop rows with an empty marker number and return (clean table, how many were dropped) (R-P-1).

    Repeated numbers are kept: th_0014 really lists 15 twice, and silver maps such a marker to both.
    """
    has_number = raw["marker_no"].notna() & (raw["marker_no"].astype(str).str.strip() != "")
    clean = raw[has_number].copy()
    clean["marker_no"] = clean["marker_no"].astype(float).astype(int)
    return clean.reset_index(drop=True), int((~has_number).sum())


def match_categories(description: str, rules: pd.DataFrame) -> list[str]:
    """Every category whose rule matches the description, in rule order (R-P-2, D-018).

    A matching `other` rule is exclusive: the exhibit is `other` only. No match gives [].
    """
    hits = []
    for pattern, category in zip(rules["pattern"], rules["category"]):
        if re.search(pattern, description, flags=re.IGNORECASE) and category not in hits:
            hits.append(category)
    return [config.OTHER] if config.OTHER in hits else hits


def build_exhibit_categories(exhibits: pd.DataFrame, rules: pd.DataFrame, suggest,
                             existing: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per exhibit and category (R-P-2).

    Rule matches give `rule` rows (confirmed 1). An exhibit no rule matches keeps its CONFIRMED llm or manual
    row from `existing`; otherwise `suggest(description)` gives a fresh unconfirmed `llm` row. So rebuilding
    never loses a confirmation, and unconfirmed suggestions follow the latest suggestion list.
    """
    kept = {}
    if existing is not None:
        confirmed = existing[(existing["source"] != "rule") & (existing["confirmed"] == 1)]
        for row in confirmed.itertuples():
            kept.setdefault((row.scene_id, int(row.marker_no)), []).append(row)
    rows = []
    for e in exhibits.itertuples():
        cats = match_categories(e.description, rules)
        if cats:
            rows += [(e.scene_id, e.marker_no, c, "rule", 1) for c in cats]
        elif (e.scene_id, e.marker_no) in kept:
            rows += [(r.scene_id, r.marker_no, r.category, r.source, r.confirmed) for r in kept[(e.scene_id, e.marker_no)]]
        else:
            rows.append((e.scene_id, e.marker_no, suggest(e.description), "llm", 0))
    out = pd.DataFrame(rows, columns=["scene_id", "marker_no", "category", "source", "confirmed"])
    return out.drop_duplicates().reset_index(drop=True)


def split_scenes(scene_ids, seed: int = config.SEED, dev_fraction: float = config.DEV_FRACTION) -> dict[str, str]:
    """scene_id -> "dev" or "test". Sorting first makes the result independent of input order (R-P-4)."""
    scenes = sorted(set(scene_ids))
    random.Random(seed).shuffle(scenes)
    n_dev = round(dev_fraction * len(scenes))
    return {s: ("dev" if i < n_dev else "test") for i, s in enumerate(scenes)}


def photo_type(box_height_frac: float | None) -> str:
    """overview / mid / close from the largest marker box as a fraction of the photo height (R-P-5)."""
    if box_height_frac is None:
        return "unknown"
    if box_height_frac >= config.CLOSE_MIN_FRAC:
        return "close"
    if box_height_frac >= config.MID_MIN_FRAC:
        return "mid"
    return "overview"


def photo_types(photos: pd.DataFrame, markers: pd.DataFrame) -> pd.Series:
    """photo_type for every row of `photos` (needs scene_id, photo_id, height); all found boxes count."""
    tallest = markers.groupby(["scene_id", "photo_id"])["h"].max()
    out = []
    for scene, photo, height in zip(photos["scene_id"], photos["photo_id"], photos["height"]):
        h = tallest.get((scene, photo))
        out.append(photo_type(None if h is None or pd.isna(h) else h / height))
    return pd.Series(out, index=photos.index)


def image_size(path) -> tuple[int, int]:
    """(width, height) as the photo is displayed, read from the file header (R-P-7).

    Phone photos can store a sideways image plus an EXIF turn flag; the marker boxes are found on the
    turned image, so the size must be turned too.
    """
    from PIL import Image

    with Image.open(path) as im:
        w, h = im.size
        if im.getexif().get(0x0112) in (5, 6, 7, 8):
            w, h = h, w
    return w, h


def pending_scenes(scenes, parts_dir: Path) -> list[str]:
    """Scenes with no finished part file yet. An empty part file still means "done" (R-P-6)."""
    parts_dir = Path(parts_dir)
    return [s for s in scenes if not (parts_dir / f"{s}.csv").exists()]
