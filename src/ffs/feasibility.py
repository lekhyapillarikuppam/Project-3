"""Feasibility-gate metrics (SPEC 8.2).

Inputs are three tables:
  truth   : scene_id, photo_id, n_markers_visible, true_marker_nos   (filled by hand)
  photos  : scene_id, photo_id, n_boxes, read_nos                    (from the probe)
  markers : scene_id, photo_id, raw_ocr                              (from the probe)
"""

from __future__ import annotations

import pandas as pd

MIN_FINDER_HIT = 0.70
MIN_READ_ACCURACY = 0.60


def _norm(token: str) -> str:
    token = token.strip()
    try:
        return str(int(float(token)))
    except ValueError:
        return token


def number_set(cell) -> set[str]:
    """';'- or ','-separated numbers as a set of normalised strings. Blank gives an empty set."""
    if cell is None or (isinstance(cell, float) and pd.isna(cell)):
        return set()
    return {_norm(t) for t in str(cell).replace(",", ";").split(";") if t.strip()}


def _precision(labelled: pd.DataFrame, markers: pd.DataFrame) -> float:
    """Share of accepted/snapped numbers that are really in the photo (report only, SPEC 8.2)."""
    if not {"marker_no", "status"} <= set(markers.columns):
        return float("nan")
    keys = ["scene_id", "photo_id"]
    m = markers[markers["status"].isin(["accepted", "snapped"])].merge(
        labelled[keys + ["true_set"]], on=keys, how="inner")   # labelled photos only
    if m.empty:
        return float("nan")
    hits = [_norm(str(n)) in t for n, t in zip(m["marker_no"], m["true_set"])]
    return float(sum(hits)) / len(hits)


def compute_metrics(truth: pd.DataFrame, photos: pd.DataFrame, markers: pd.DataFrame) -> dict:
    t = truth.dropna(subset=["n_markers_visible"]).copy()
    t["n_markers_visible"] = t["n_markers_visible"].astype(float).astype(int)
    keys = ["scene_id", "photo_id"]
    df = t.merge(photos, on=keys, how="left")
    df["n_boxes"] = df["n_boxes"].fillna(0).astype(int)

    raw_by_photo = markers.groupby(keys)["raw_ocr"].apply(
        lambda s: {_norm(str(v)) for v in s if str(v).strip() and str(v) != "nan"})
    df["true_set"] = df["true_marker_nos"].map(number_set)
    df["read_set"] = df["read_nos"].map(number_set)
    df["raw_set"] = [raw_by_photo.get((s, p), set()) for s, p in zip(df.scene_id, df.photo_id)]

    n_vis = df["n_markers_visible"]
    found = df[["n_boxes", "n_markers_visible"]].min(axis=1)
    n_true = df["true_set"].map(len).sum()

    def ratio(num, den):
        return float(num) / float(den) if den else float("nan")

    metrics = {
        "n_photos": int(len(df)),
        "n_photos_with_markers": int((n_vis > 0).sum()),
        "n_true_markers": int(n_vis.sum()),
        "finder_hit_rate": ratio(found.sum(), n_vis.sum()),
        "finder_fp_per_photo": ratio((df["n_boxes"] - n_vis).clip(lower=0).sum(), len(df)),
        "read_accuracy": ratio(sum(len(a & b) for a, b in zip(df.true_set, df.read_set)), n_true),
        "raw_read_accuracy": ratio(sum(len(a & b) for a, b in zip(df.true_set, df.raw_set)), n_true),
    }
    metrics["precision"] = _precision(df, markers)
    metrics["pass_finder"] = bool(metrics["finder_hit_rate"] >= MIN_FINDER_HIT)
    metrics["pass_read"] = bool(metrics["read_accuracy"] >= MIN_READ_ACCURACY)

    per_scene = []
    for scene, g in df.groupby("scene_id"):
        gv = g["n_markers_visible"]
        per_scene.append({
            "scene_id": scene, "photos": len(g), "true_markers": int(gv.sum()),
            "finder_hit_rate": ratio(g[["n_boxes", "n_markers_visible"]].min(axis=1).sum(), gv.sum()),
            "read_accuracy": ratio(sum(len(a & b) for a, b in zip(g.true_set, g.read_set)),
                                   g["true_set"].map(len).sum()),
        })
    metrics["per_scene"] = pd.DataFrame(per_scene)
    return metrics


def verdict(metrics: dict) -> str:
    """Plain-language decision following the SPEC 8.3 table."""
    if metrics["pass_finder"] and metrics["pass_read"]:
        return "PASS: continue to Phase 2 with colour + shape. Log the final parameters."
    if not metrics["pass_finder"]:
        return ("FAIL (finder): tune MarkerParams using the overlays (max 2 rounds, log each "
                "change). If still failing, use OWLv2 to find markers.")
    return ("FAIL (reading): improve preprocessing (rotation, contrast) for at most 2 rounds, "
            "then try PaddleOCR. If still failing, fall back to scene-level ground truth.")
