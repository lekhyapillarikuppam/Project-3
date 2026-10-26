"""Silver labels (SPEC 4.5): the answer key built from markers and exhibit categories."""

from __future__ import annotations

import pandas as pd

from .schemas import validate


def _as_int(value):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def counted_markers(markers: pd.DataFrame, exhibit_categories: pd.DataFrame) -> pd.DataFrame:
    """Accepted or snapped markers whose number is in the scene's list (R-S-1)."""
    listed = {(s, _as_int(n)) for s, n in zip(exhibit_categories["scene_id"], exhibit_categories["marker_no"])}
    m = markers[markers["status"].isin(["accepted", "snapped"])].copy()
    m["marker_no"] = m["marker_no"].map(_as_int)
    keep = pd.Series([(s, n) in listed for s, n in zip(m["scene_id"], m["marker_no"])],
                     index=m.index, dtype=bool)       # an explicit Series: an empty list would select columns
    return m[keep]


def uncertain_photos(markers: pd.DataFrame) -> set[tuple[str, str]]:
    """Photos with at least one rejected box (R-S-5). Used as an analysis slice, not stored in silver."""
    rejected = markers[markers["status"] == "rejected"]
    return set(zip(rejected["scene_id"], rejected["photo_id"]))


def build_silver(photos: pd.DataFrame, markers: pd.DataFrame, exhibit_categories: pd.DataFrame,
                 categories: list[str]) -> pd.DataFrame:
    """One row per photo and final category (R-S-2, R-S-3). `categories` must not contain "other"."""
    usable = exhibit_categories[(exhibit_categories["source"] == "rule") | (exhibit_categories["confirmed"] == 1)]
    # (scene, number) -> categories of every exhibit with that number (R-S-4: repeats give a union)
    cats_of: dict[tuple, set] = {}
    for s, n, c in zip(usable["scene_id"], usable["marker_no"], usable["category"]):
        cats_of.setdefault((s, _as_int(n)), set()).add(c)

    # (scene, photo, category) -> marker numbers that made it present
    found: dict[tuple, set] = {}
    counted = counted_markers(markers, exhibit_categories)
    for s, p, n in zip(counted["scene_id"], counted["photo_id"], counted["marker_no"]):
        for c in cats_of.get((s, n), ()):
            found.setdefault((s, p, c), set()).add(n)

    rows = []
    for s, p in zip(photos["scene_id"], photos["photo_id"]):
        for c in categories:
            nos = sorted(found.get((s, p, c), ()))
            rows.append({"scene_id": s, "photo_id": p, "category": c, "present": int(bool(nos)),
                         "marker_nos": ";".join(str(n) for n in nos)})
    out = pd.DataFrame(rows, columns=["scene_id", "photo_id", "category", "present", "marker_nos"])
    validate(out, "silver")
    return out
