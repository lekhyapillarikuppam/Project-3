"""Test-set evaluation: metrics, scene-level bootstrap, paired comparison (SPEC section 6).

Plain numpy/pandas. Reads saved model outputs only; no model is run here.
Photo scores follow the frozen protocol (SPEC R-B-8):
  CLIP   = softmax across the categories of the mean of each category's prompt logits
  OWLv2  = best box score of the category on the full photo (tile 0)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

KEYS = ["scene_id", "photo_id"]


# ---- photo-level scores --------------------------------------------------------------------------
def clip_photo_scores(clip: pd.DataFrame, photos: pd.DataFrame, cats: list[str]) -> pd.DataFrame:
    """One row per photo x category: softmax over categories of the mean prompt logit."""
    m = clip.groupby(KEYS + ["category"])["score"].mean().unstack("category").reindex(columns=cats)
    m = m.reindex(pd.MultiIndex.from_frame(photos[KEYS]))
    if m.isna().any().any():
        raise ValueError("CLIP scores are missing for some photos or categories")
    z = m.to_numpy(float)
    z = z - z.max(axis=1, keepdims=True)
    p = np.exp(z)
    p = p / p.sum(axis=1, keepdims=True)
    out = pd.DataFrame(p, index=m.index, columns=cats).stack().rename("clip_score").reset_index()
    out.columns = KEYS + ["category", "clip_score"]
    return out


def owl_photo_scores(owl: pd.DataFrame, photos: pd.DataFrame, cats: list[str], tile: int = 0) -> pd.DataFrame:
    """One row per photo x category: best box score on the given tile, plus that box (0 / no box if none)."""
    use = owl[owl["tile"] == tile]
    idx = use.groupby(KEYS + ["category"])["score"].idxmax()
    best = use.loc[idx, KEYS + ["category", "score", "prompt", "x", "y", "w", "h"]]
    best = best.rename(columns={"score": "owl_score", "prompt": "owl_prompt",
                                "x": "box_x", "y": "box_y", "w": "box_w", "h": "box_h"})
    grid = photos[KEYS].merge(pd.DataFrame({"category": cats}), how="cross")
    out = grid.merge(best, on=KEYS + ["category"], how="left")
    out["owl_score"] = out["owl_score"].fillna(0.0)
    return out


def find_thresholds(protocol: dict, model: str, cats: list[str]) -> dict[str, float]:
    """Find the per-category thresholds of a model ('clip' or 'owl') in protocol_v1.json, whatever its layout."""
    def numeric(d):
        return all(isinstance(d.get(c), (int, float)) and not isinstance(d.get(c), bool) for c in cats)

    def walk(node, path):
        if isinstance(node, dict):
            if all(c in node for c in cats):
                if numeric(node) and model in "/".join(path).lower():
                    yield {c: float(node[c]) for c in cats}
                if all(isinstance(node[c], dict) for c in cats):          # {category: {model: value}}
                    got = {}
                    for c in cats:
                        hit = [v for k, v in node[c].items()
                               if model in k.lower() and isinstance(v, (int, float)) and not isinstance(v, bool)]
                        if hit:
                            got[c] = float(hit[0])
                    if len(got) == len(cats):
                        yield got
            for k, v in node.items():
                yield from walk(v, path + [str(k)])

    found = list(walk(protocol, []))
    if not found:
        raise ValueError(f"No thresholds for '{model}' found in the protocol file (need one number per category: {cats})")
    return found[0]


# ---- metrics -------------------------------------------------------------------------------------
def prf(y: np.ndarray, pred: np.ndarray) -> tuple[float, float, float]:
    tp = float(((y == 1) & (pred == 1)).sum())
    fp = float(((y == 0) & (pred == 1)).sum())
    fn = float(((y == 1) & (pred == 0)).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def average_precision(y: np.ndarray, score: np.ndarray) -> float:
    """Area under the precision-recall step curve; ties in score are handled as one step. NaN if no positives."""
    y = np.asarray(y)
    n_pos = int(y.sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-score, kind="stable")
    s, yy = score[order], y[order]
    tp = np.cumsum(yy)
    fp = np.cumsum(1 - yy)
    last = np.r_[s[1:] != s[:-1], True]                 # last row of each distinct score
    tp, fp = tp[last], fp[last]
    prec = tp / (tp + fp)
    rec = tp / n_pos
    return float(np.sum(np.diff(np.r_[0.0, rec]) * prec))


def cohens_kappa(y: np.ndarray, pred: np.ndarray) -> float:
    n = len(y)
    if n == 0:
        return float("nan")
    po = float((y == pred).mean())
    pe = float(y.mean() * pred.mean() + (1 - y.mean()) * (1 - pred.mean()))
    return float("nan") if pe == 1 else (po - pe) / (1 - pe)


# ---- bootstrap over scenes -----------------------------------------------------------------------
def _metrics_matrix(Y, S, P, idx):
    """F1 and AP per category for the rows idx. Y, S, P are (n_photos, n_cats); S may be None (no score)."""
    f1 = np.empty(Y.shape[1])
    ap = np.full(Y.shape[1], np.nan)
    for j in range(Y.shape[1]):
        f1[j] = prf(Y[idx, j], P[idx, j])[2]
        if S is not None:
            ap[j] = average_precision(Y[idx, j], S[idx, j])
    return f1, ap


def evaluate(df: pd.DataFrame, systems: dict[str, tuple[str | None, str]], cats: list[str],
             B: int = 1000, seed: int = 42, pairs: list[tuple[str, str]] | None = None):
    """df: one row per photo x category with columns scene_id, photo_id, category, y and, per system,
    a prediction column and an optional score column. systems: name -> (score_col or None, pred_col).
    Returns (metrics table, paired-difference table). CIs: percentile, scenes resampled with replacement."""
    keys = df[KEYS].drop_duplicates().reset_index(drop=True)
    pos = pd.MultiIndex.from_frame(keys)

    def mat(col):
        return df.pivot(index=KEYS, columns="category", values=col).reindex(pos)[cats].to_numpy(float)

    Y = mat("y").astype(int)
    M = {}
    for name, (sc, pc) in systems.items():
        M[name] = (mat(sc) if sc else None, mat(pc).astype(int))
    scenes = keys["scene_id"].to_numpy()
    uniq = np.array(sorted(set(scenes)))
    rows_of = {s: np.flatnonzero(scenes == s) for s in uniq}
    rng = np.random.default_rng(seed)
    allidx = np.arange(len(keys))

    point = {n: _metrics_matrix(Y, M[n][0], M[n][1], allidx) for n in systems}
    boots = {n: ([], []) for n in systems}
    for _ in range(B):
        draw = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([rows_of[s] for s in draw])
        for n in systems:
            f1, ap = _metrics_matrix(Y, M[n][0], M[n][1], idx)
            boots[n][0].append(f1)
            boots[n][1].append(ap)
    boots = {n: (np.array(a), np.array(b)) for n, (a, b) in boots.items()}

    def ci(arr):
        with np.errstate(all="ignore"):
            return np.nanpercentile(arr, 2.5), np.nanpercentile(arr, 97.5)

    rows = []
    for n, (sc, pc) in systems.items():
        pf, pa = point[n]
        bf, ba = boots[n]
        for j, c in enumerate(cats + ["macro"]):
            if c == "macro":
                f1v, apv = float(np.nanmean(pf)), (float(np.nanmean(pa)) if sc else float("nan"))
                f1b, apb = np.nanmean(bf, axis=1), (np.nanmean(ba, axis=1) if sc else np.full(B, np.nan))
                p = r = float("nan")
                prec = np.mean([prf(Y[:, k], M[n][1][:, k])[0] for k in range(len(cats))])
                rec = np.mean([prf(Y[:, k], M[n][1][:, k])[1] for k in range(len(cats))])
                kap = float(np.mean([cohens_kappa(Y[:, k], M[n][1][:, k]) for k in range(len(cats))]))
                npos = int(Y.sum())
            else:
                f1v, apv = float(pf[j]), float(pa[j])
                f1b, apb = bf[:, j], ba[:, j]
                prec, rec, _ = prf(Y[:, j], M[n][1][:, j])
                kap = cohens_kappa(Y[:, j], M[n][1][:, j])
                npos = int(Y[:, j].sum())
            flo, fhi = ci(f1b)
            alo, ahi = ci(apb) if sc else (float("nan"), float("nan"))
            rows.append(dict(system=n, category=c, n_pos=npos, precision=prec, recall=rec, f1=f1v, f1_lo=flo,
                             f1_hi=fhi, ap=apv, ap_lo=alo, ap_hi=ahi, kappa=kap))
    metrics = pd.DataFrame(rows)

    prow = []
    for a, b in (pairs or []):
        for mname, k in (("f1", 0), ("ap", 1)):
            if mname == "ap" and (systems[a][0] is None or systems[b][0] is None):
                continue
            for j, c in enumerate(cats + ["macro"]):
                if c == "macro":
                    d = np.nanmean(point[a][k]) - np.nanmean(point[b][k])
                    db = np.nanmean(boots[a][k], axis=1) - np.nanmean(boots[b][k], axis=1)
                else:
                    d = point[a][k][j] - point[b][k][j]
                    db = boots[a][k][:, j] - boots[b][k][:, j]
                lo, hi = ci(db)
                prow.append(dict(a=a, b=b, metric=mname, category=c, diff=float(d), lo=lo, hi=hi,
                                 frac_le_0=float(np.nanmean(db <= 0))))
    return metrics, pd.DataFrame(prow)
