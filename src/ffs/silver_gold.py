"""Silver against gold on the test split (SPEC 4.7, R-G-7 and R-G-8).

Silver is treated as the prediction and gold as the truth. Every number comes with a 95% interval from a
scene-level cluster bootstrap: whole scenes are drawn with replacement, because photos of one scene are
correlated and must never be split up.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .gold import cohens_kappa
from .silver import uncertain_photos

KEYS = ["scene_id", "photo_id"]
STATS = ("precision", "recall", "f1", "kappa")


def counts(gold, silver) -> tuple[int, int, int, int]:
    """(TP, FP, FN, TN) with silver as the prediction and gold as the truth."""
    g, s = np.asarray(list(gold), dtype=int), np.asarray(list(silver), dtype=int)
    return (int(((g == 1) & (s == 1)).sum()), int(((g == 0) & (s == 1)).sum()),
            int(((g == 1) & (s == 0)).sum()), int(((g == 0) & (s == 0)).sum()))


def metrics(tp, fp, fn, tn) -> dict:
    """Precision, recall, F1 and Cohen's kappa from counts; NaN where undefined. Works on numbers or arrays."""
    tp, fp, fn, tn = (np.asarray(x, dtype=float) for x in (tp, fp, fn, tn))
    n = tp + fp + fn + tn
    with np.errstate(divide="ignore", invalid="ignore"):
        pa, pb = (tp + fn) / n, (tp + fp) / n                      # share of gold positives, share of silver positives
        pe = pa * pb + (1 - pa) * (1 - pb)
        out = {"precision": tp / (tp + fp), "recall": tp / (tp + fn), "f1": 2 * tp / (2 * tp + fp + fn),
               "kappa": ((tp + tn) / n - pe) / (1 - pe)}
    return {k: (v[()] if np.ndim(v) == 0 else v) for k, v in out.items()}


def draw_scenes(n_scenes: int, n_boot: int, seed: int = config.SEED) -> np.ndarray:
    """Row b = the scenes (as indexes) drawn with replacement for resample b."""
    return np.random.default_rng(seed).integers(0, n_scenes, size=(n_boot, n_scenes))


def bootstrap_stats(per_scene: np.ndarray, idx: np.ndarray) -> dict:
    """Metric values for every resample. `per_scene` is (scenes, 4) counts; a resample adds up whole scenes."""
    sums = per_scene[idx].sum(axis=1)                              # (resamples, 4)
    return metrics(sums[:, 0], sums[:, 1], sums[:, 2], sums[:, 3])


def human_audit_rows(gold: pd.DataFrame, sources: pd.DataFrame) -> pd.DataFrame:
    """Gold rows whose label is the human's on a random audit photo (R-G-12 b): the unbiased subset."""
    keep = sources.loc[sources["source"] == "human_audit", KEYS + ["category"]]
    return gold.merge(keep, on=KEYS + ["category"], how="inner")


def mode_name(human_only: bool) -> str:
    """Which comparison a run made, for the printout (R-G-12)."""
    return "audit photos only (human labels, random sample)" if human_only else "all gold rows (AI-labelled, human-checked)"


def paired_labels(gold: pd.DataFrame, silver: pd.DataFrame, scene_split: pd.DataFrame,
                  photos: pd.DataFrame, markers: pd.DataFrame) -> pd.DataFrame:
    """One row per test photo and category: gold, silver and the three slice labels (R-G-7)."""
    test = set(scene_split.loc[scene_split["split"] == "test", "scene_id"])
    g = gold[gold["scene_id"].isin(test)].rename(columns={"present": "gold"})[KEYS + ["category", "gold"]]
    s = silver.rename(columns={"present": "silver"})[KEYS + ["category", "silver"]]
    df = g.merge(s, on=KEYS + ["category"], how="inner")

    df = df.merge(photos[KEYS + ["photo_type"]], on=KEYS, how="left")
    df["photo_type"] = df["photo_type"].fillna("unknown")
    unc = uncertain_photos(markers)                                # at least one rejected box (R-S-5)
    df["certainty"] = ["uncertain" if k in unc else "certain" for k in zip(df.scene_id, df.photo_id)]
    counted = markers[markers["status"].isin(["accepted", "snapped"])]   # always a number from the scene's list
    has = set(zip(counted.scene_id, counted.photo_id))
    df["counted_marker"] = ["has counted marker" if k in has else "no counted marker"
                            for k in zip(df.scene_id, df.photo_id)]
    return df


def silver_vs_gold(gold: pd.DataFrame, silver: pd.DataFrame, scene_split: pd.DataFrame, photos: pd.DataFrame,
                   markers: pd.DataFrame, n_boot: int = config.N_BOOTSTRAP, seed: int = config.SEED) -> pd.DataFrame:
    """The report: overall and sliced counts, metrics and 95% scene-bootstrap intervals, per category."""
    df = paired_labels(gold, silver, scene_split, photos, markers)
    scenes = sorted(df["scene_id"].unique())
    scene_pos = {s: i for i, s in enumerate(scenes)}
    idx = draw_scenes(len(scenes), n_boot, seed)                   # the same resamples for every row
    lo, hi = (1 - config.CI_LEVEL) / 2 * 100, (1 + config.CI_LEVEL) / 2 * 100

    rows = []
    for slice_name in ("overall", "photo_type", "certainty", "counted_marker"):
        values = [("all", df)] if slice_name == "overall" else list(df.groupby(slice_name))
        for value, part in values:
            for category, d in part.groupby("category"):
                per_scene = np.zeros((len(scenes), 4))
                for scene, g in d.groupby("scene_id"):
                    per_scene[scene_pos[scene]] = counts(g["gold"], g["silver"])
                tp, fp, fn, tn = counts(d["gold"], d["silver"])
                point = metrics(tp, fp, fn, tn)
                point["kappa"] = cohens_kappa(d["gold"], d["silver"])          # the project's kappa, same number
                boot = bootstrap_stats(per_scene, idx)
                row = {"slice": slice_name, "value": value, "category": category, "n_photos": len(d),
                       "gold_prevalence": float(d["gold"].mean()), "TP": tp, "FP": fp, "FN": fn, "TN": tn}
                for stat in STATS:
                    row[stat] = float(point[stat])
                    finite = boot[stat][~np.isnan(boot[stat])]
                    row[f"{stat}_lo"] = float(np.percentile(finite, lo)) if len(finite) else float("nan")
                    row[f"{stat}_hi"] = float(np.percentile(finite, hi)) if len(finite) else float("nan")
                rows.append(row)
    cols = ["slice", "value", "category", "n_photos", "gold_prevalence", "TP", "FP", "FN", "TN"]
    cols += [c for stat in STATS for c in (stat, f"{stat}_lo", f"{stat}_hi")]
    return pd.DataFrame(rows, columns=cols)
