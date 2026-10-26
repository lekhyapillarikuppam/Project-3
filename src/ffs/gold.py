"""Gold labels (SPEC 4.5b): data for the labelling tool, import of its export, agreement statistics."""

from __future__ import annotations

import random

import pandas as pd

from . import config
from .schemas import validate


def ordered_categories(silver: pd.DataFrame) -> list[str]:
    """Final categories in config order if config knows them, else sorted."""
    found = set(silver["category"].unique())
    order = list(getattr(config, "FINAL_CATEGORIES", ()))
    return [c for c in order if c in found] + sorted(found - set(order))


def build_tool_data(photos: pd.DataFrame, scene_split: pd.DataFrame, exhibits: pd.DataFrame,
                    categories: list[str], split: str = "test", sample: int | None = None,
                    seed: int = config.SEED, storage_key: str = "ffs_gold_pass1",
                    export_name: str = "gold_pass1_export.csv", base: str = "../../data/raw/") -> dict:
    """Everything the labelling page needs. The silver labels are NOT included (blind labelling, R-G-2)."""
    scenes = set(scene_split.loc[scene_split["split"] == split, "scene_id"])
    p = photos[photos["scene_id"].isin(scenes)].sort_values(["scene_id", "photo_id"])
    rows = list(zip(p["scene_id"], p["photo_id"], p["relpath"]))
    if sample is not None:                      # second pass: a random subset, still in scene order
        rows = sorted(random.Random(seed).sample(rows, min(sample, len(rows))))
    ex = {}
    for s, n, d in zip(exhibits["scene_id"], exhibits["marker_no"], exhibits["description"]):
        ex.setdefault(s, []).append([None if pd.isna(n) else int(n), str(d)])
    keep = {r[0] for r in rows}
    return {
        "cats": list(categories), "base": base, "storage_key": storage_key, "export_name": export_name,
        "photos": [{"scene": s, "photo": ph, "rel": rel} for s, ph, rel in rows],
        "exhibits": {s: sorted(v, key=lambda x: (x[0] is None, x[0])) for s, v in ex.items() if s in keep},
    }


def gold_from_export(export: pd.DataFrame, photos: pd.DataFrame, scene_split: pd.DataFrame,
                     silver: pd.DataFrame, categories: list[str], split: str = "test",
                     allow_partial: bool = False) -> pd.DataFrame:
    """Turn the tool's export into the `gold` contract; `corrected` = gold differs from silver (R-G-5).

    The export has columns scene_id, photo_id, category, present, labelled_at.
    Unless allow_partial, every photo of the split must have a label for every category (R-G-4).
    """
    need = {"scene_id", "photo_id", "category", "present", "labelled_at"}
    if not need <= set(export.columns):
        raise ValueError(f"export is missing columns {sorted(need - set(export.columns))}")
    export = export[export["category"].isin(categories)].drop_duplicates(["scene_id", "photo_id", "category"], keep="last")
    scenes = set(scene_split.loc[scene_split["split"] == split, "scene_id"])
    want = photos[photos["scene_id"].isin(scenes)][["scene_id", "photo_id"]]
    full = want.merge(pd.DataFrame({"category": categories}), how="cross")
    merged = full.merge(export, on=["scene_id", "photo_id", "category"], how="left")
    missing = merged["present"].isna()
    if missing.any() and not allow_partial:
        n_photos = merged[missing][["scene_id", "photo_id"]].drop_duplicates().shape[0]
        raise ValueError(f"{n_photos} photos of split '{split}' have no complete label; "
                         f"finish labelling or use allow_partial")
    merged = merged[~missing].copy()
    merged["present"] = merged["present"].astype(int)
    sv = silver.rename(columns={"present": "silver_present"})[["scene_id", "photo_id", "category", "silver_present"]]
    merged = merged.merge(sv, on=["scene_id", "photo_id", "category"], how="left")
    merged["corrected"] = (merged["present"] != merged["silver_present"].fillna(0).astype(int)).astype(int)
    out = merged[["scene_id", "photo_id", "category", "present", "corrected", "labelled_at"]].reset_index(drop=True)
    validate(out, "gold")
    return out


def cohens_kappa(a, b) -> float:
    """Cohen's kappa for two 0/1 label lists. NaN if both are constant and equal (undefined)."""
    a, b = list(map(int, a)), list(map(int, b))
    n = len(a)
    if n == 0:
        return float("nan")
    po = sum(x == y for x, y in zip(a, b)) / n
    pa1, pb1 = sum(a) / n, sum(b) / n
    pe = pa1 * pb1 + (1 - pa1) * (1 - pb1)
    if pe == 1.0:
        return float("nan")
    return (po - pe) / (1 - pe)


def intra_rater_agreement(pass1: pd.DataFrame, pass2: pd.DataFrame) -> pd.DataFrame:
    """Per category: photos relabelled, percent agreement, kappa between the first and second pass (R-G-6)."""
    keys = ["scene_id", "photo_id", "category"]
    m = pass1.merge(pass2, on=keys, suffixes=("_1", "_2"))
    rows = []
    for cat, g in m.groupby("category"):
        rows.append({"category": cat, "photos": len(g),
                     "agreement": float((g["present_1"] == g["present_2"]).mean()),
                     "kappa": cohens_kappa(g["present_1"], g["present_2"])})
    return pd.DataFrame(rows, columns=["category", "photos", "agreement", "kappa"])
