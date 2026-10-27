"""Stage B scoring rules and dev tuning (SPEC section 5.1 and D-029).

Reads only the saved model outputs (clip_scores, owlv2_raw) and, for tuning, silver labels of DEV scenes.
No model is run here. Everything is plain numpy/pandas so each rule can be tested on tiny inputs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from . import config

NMS_IOU = 0.5
MIN_POSITIVES_TO_TUNE = 5
CLIP_GRID = [round(v, 2) for v in np.arange(0.05, 1.0, 0.05)]    # softmax probability, 19 values
OWL_GRID = [round(v, 3) for v in np.arange(0.05, 0.55, 0.025)]    # box score, 20 values


# ---- Photo scores ------------------------------------------------------------------------------
def clip_photo_scores(clip: pd.DataFrame) -> pd.DataFrame:
    """Photo x category score in 0..1.

    Step 1: mean of the category's 3 prompt logits. Step 2: softmax across the categories of the photo.
    There is no "none" prompt: a photo with no object simply has no category with a high share, and the
    per-category threshold decides. The logits are already cosine x 100, so no extra temperature is used.
    """
    logit = clip.groupby(["scene_id", "photo_id", "category"])["score"].mean().unstack("category")
    z = logit.to_numpy()
    e = np.exp(z - z.max(axis=1, keepdims=True))          # subtract the max for numerical safety
    prob = pd.DataFrame(e / e.sum(axis=1, keepdims=True), index=logit.index, columns=logit.columns)
    return prob.stack().rename("score").reset_index()


def owl_photo_scores(owl: pd.DataFrame, photos: pd.DataFrame, tiles: str) -> pd.DataFrame:
    """Photo x category = best box score. tiles is 'full' (tile 0 only) or 'all' (tiles 0 to 4).

    Photos or categories without any box above the raw cut-off get 0.
    """
    use = owl[owl.tile == 0] if tiles == "full" else owl
    best = use.groupby(["scene_id", "photo_id", "category"])["score"].max()
    grid = pd.MultiIndex.from_tuples(
        [(s, p, c) for s, p in zip(photos.scene_id, photos.photo_id) for c in config.FINAL_CATEGORIES],
        names=["scene_id", "photo_id", "category"])
    return best.reindex(grid, fill_value=0.0).rename("score").reset_index()


# ---- Box merge rule ----------------------------------------------------------------------------
def iou(a, b) -> float:
    """Intersection over union of two boxes (x, y, w, h)."""
    ix = max(0.0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def nms(boxes: pd.DataFrame, iou_thr: float = NMS_IOU) -> pd.DataFrame:
    """Merge boxes of one photo and category across prompts and tiles.

    Highest score first; a box is dropped if it overlaps an already kept box with IoU above the threshold.
    The best box always survives, so the photo score (max box score) is the same before and after merging.
    """
    kept = []
    for _, row in boxes.sort_values("score", ascending=False).iterrows():
        if all(iou((row.x, row.y, row.w, row.h), (k.x, k.y, k.w, k.h)) <= iou_thr for k in kept):
            kept.append(row)
    return pd.DataFrame(kept, columns=boxes.columns).reset_index(drop=True)


# ---- Photo-level decision rule -----------------------------------------------------------------
def decide(scores: pd.DataFrame, thresholds: dict) -> pd.DataFrame:
    """present = 1 when the photo score reaches the category threshold (>=)."""
    out = scores.copy()
    out["pred"] = (out["score"] >= out["category"].map(thresholds)).astype(int)
    return out


# ---- Metrics -----------------------------------------------------------------------------------
def prf(y: np.ndarray, pred: np.ndarray) -> tuple[float, float, float]:
    """Precision, recall, F1. A zero denominator gives 0."""
    tp = int(((y == 1) & (pred == 1)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    return p, r, f


def best_threshold(y: np.ndarray, score: np.ndarray, grid) -> float:
    """Grid value with the highest F1; on a tie the lowest threshold."""
    f1s = [prf(y, (score >= t).astype(int))[2] for t in grid]
    return float(grid[int(np.argmax(f1s))])


def choose_thresholds(df: pd.DataFrame, grid) -> tuple[dict, list]:
    """One threshold per category. df has columns category, score, y.

    A category with fewer than MIN_POSITIVES_TO_TUNE positives is not tuned: it gets the median threshold
    of the tuned categories and is returned in the flagged list.
    """
    th, flagged = {}, []
    for cat in config.FINAL_CATEGORIES:
        d = df[df.category == cat]
        if d.y.sum() >= MIN_POSITIVES_TO_TUNE:
            th[cat] = best_threshold(d.y.to_numpy(), d.score.to_numpy(), grid)
        else:
            flagged.append(cat)
    med = float(np.median(list(th.values())))
    for cat in flagged:
        th[cat] = med
    return th, flagged


def cluster_bootstrap_f1(df: pd.DataFrame, n: int = config.N_BOOTSTRAP, seed: int = config.SEED):
    """95% CI of F1 for one category: resample whole scenes with replacement.

    df has scene_id, y, pred. Resamples where F1 is undefined (no positive and no prediction) are skipped.
    """
    rng = np.random.default_rng(seed)
    groups = {s: (g.y.to_numpy(), g.pred.to_numpy()) for s, g in df.groupby("scene_id")}
    scenes = list(groups)
    vals = []
    for _ in range(n):
        pick = rng.choice(len(scenes), size=len(scenes), replace=True)
        y = np.concatenate([groups[scenes[i]][0] for i in pick])
        p = np.concatenate([groups[scenes[i]][1] for i in pick])
        if y.sum() + p.sum() == 0:
            continue
        vals.append(prf(y, p)[2])
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def ap(y: np.ndarray, score: np.ndarray) -> float:
    return float(average_precision_score(y, score)) if y.sum() else float("nan")
