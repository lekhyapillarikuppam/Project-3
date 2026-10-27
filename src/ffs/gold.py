"""Gold labels (SPEC 4.5b): data for the labelling tool, import of its export, agreement statistics."""

from __future__ import annotations

import random
import secrets

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


# ======================================================================================
# AI-labelled, human-checked gold (SPEC 4.7, D-026)
# ======================================================================================
import json  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

CONFIDENCES = ("low", "med", "high")
AI_COLUMNS = ["scene_id", "photo_id", "category", "present", "confidence", "note", "labeller", "labelled_at"]
KEYS = ["scene_id", "photo_id"]


def validate_ai(ai: pd.DataFrame, categories: list[str]) -> None:
    if list(ai.columns) != AI_COLUMNS:
        raise ValueError(f"gold_ai columns must be {AI_COLUMNS}, got {list(ai.columns)}")
    if not ai["present"].isin([0, 1]).all():
        raise ValueError("gold_ai.present must be 0 or 1")
    if not ai["confidence"].isin(CONFIDENCES).all():
        raise ValueError(f"gold_ai.confidence must be one of {CONFIDENCES}")
    bad = sorted(set(ai["category"]) - set(categories))
    if bad:
        raise ValueError(f"gold_ai has unknown categories {bad}")


def upsert_ai_labels(path, scene: str, photo: str, labels: dict, confidence: str, note: str,
                     labeller: str, categories: list[str], now: str | None = None) -> None:
    """Write one photo's AI labels (all categories at once), replacing any earlier rows for that photo."""
    if set(labels) != set(categories):
        raise ValueError(f"labels must cover exactly {categories}, got {sorted(labels)}")
    if any(v not in (0, 1) for v in labels.values()):
        raise ValueError("each label must be 0 or 1")
    if confidence not in CONFIDENCES:
        raise ValueError(f"confidence must be one of {CONFIDENCES}")
    note = " ".join(str(note).replace(",", ";").split())[:140]
    now = now or pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
    path = Path(path)
    old = pd.read_csv(path, dtype={"note": str}).fillna({"note": ""}) if path.exists() else pd.DataFrame(columns=AI_COLUMNS)
    old = old[~((old["scene_id"] == scene) & (old["photo_id"] == photo))]
    new = pd.DataFrame([(scene, photo, c, int(labels[c]), confidence, note, labeller, now) for c in categories],
                       columns=AI_COLUMNS)
    pd.concat([old, new], ignore_index=True).sort_values(KEYS + ["category"]).to_csv(path, index=False)


def next_unlabelled(photos: pd.DataFrame, scene_split: pd.DataFrame, ai: pd.DataFrame | None,
                    split: str = "test", n: int = 10) -> list[tuple[str, str]]:
    scenes = set(scene_split.loc[scene_split["split"] == split, "scene_id"])
    p = photos[photos["scene_id"].isin(scenes)].sort_values(KEYS)
    done = set() if ai is None or ai.empty else set(zip(ai["scene_id"], ai["photo_id"]))
    return [(s, ph) for s, ph in zip(p["scene_id"], p["photo_id"]) if (s, ph) not in done][:n]


def review_pairs(ai: pd.DataFrame, silver: pd.DataFrame) -> pd.DataFrame:
    """Photo x category pairs the human must decide: AI and silver disagree, or the AI is not confident (R-G-12)."""
    sv = silver.rename(columns={"present": "silver_present"})[KEYS + ["category", "silver_present"]]
    m = ai.merge(sv, on=KEYS + ["category"], how="left")
    m["silver_present"] = m["silver_present"].fillna(0).astype(int)
    disagree = m["present"] != m["silver_present"]
    low = m["confidence"] == "low"
    sel = m[disagree | low].copy()
    sel["reason"] = np.where(disagree[sel.index], "disagree", "low_conf")
    return sel[KEYS + ["category", "reason"]].sort_values(KEYS + ["category"]).reset_index(drop=True)


def audit_photos(ai: pd.DataFrame, review: pd.DataFrame, n: int, seed: int = config.SEED) -> list[tuple[str, str]]:
    """n random photos with no review pair (everything agreed and confident), for a full blind check (R-G-13)."""
    allp = sorted(set(zip(ai["scene_id"], ai["photo_id"])))
    reviewed = set(zip(review["scene_id"], review["photo_id"]))
    pool = [p for p in allp if p not in reviewed]
    return sorted(random.Random(seed).sample(pool, min(n, len(pool))))


def tool_data_for_pairs(photos: pd.DataFrame, exhibits: pd.DataFrame, categories: list[str],
                        show: dict, storage_key: str, export_name: str,
                        base: str = "../../data/raw/") -> dict:
    """Labelling-page data where each photo shows only some categories. show: (scene, photo) -> [categories]."""
    rel = dict(zip(zip(photos["scene_id"], photos["photo_id"]), photos["relpath"]))
    keys = sorted(show)
    data = {"cats": list(categories), "base": base, "storage_key": storage_key, "export_name": export_name,
            "photos": [{"scene": s, "photo": p, "rel": rel[(s, p)],
                        "show": [categories.index(c) for c in categories if c in show[(s, p)]]} for s, p in keys],
            "exhibits": {}}
    ex = {}
    for s, n, d in zip(exhibits["scene_id"], exhibits["marker_no"], exhibits["description"]):
        ex.setdefault(s, []).append([None if pd.isna(n) else int(n), str(d)])
    scenes = {s for s, _ in keys}
    data["exhibits"] = {s: sorted(v, key=lambda x: (x[0] is None, x[0])) for s, v in ex.items() if s in scenes}
    return data


def merge_gold(ai: pd.DataFrame, silver: pd.DataFrame, review: pd.DataFrame, audit: list,
               review_export: pd.DataFrame | None, audit_export: pd.DataFrame | None,
               photos: pd.DataFrame, scene_split: pd.DataFrame, categories: list[str],
               split: str = "test", allow_partial: bool = False):
    """Final gold from AI labels + human decisions (SPEC R-G-14).

    Human label wins where there is one (review pairs, audit photos x all categories); otherwise the AI label.
    Returns (gold, sources). sources has one row per photo x category with source in
    {human_review, human_audit, ai} plus ai_present, silver_present, human_present.
    """
    validate_ai(ai, categories)
    scenes = set(scene_split.loc[scene_split["split"] == split, "scene_id"])
    want = photos[photos["scene_id"].isin(scenes)][KEYS]
    full = want.merge(pd.DataFrame({"category": categories}), how="cross")
    base = full.merge(ai.rename(columns={"present": "ai_present", "confidence": "ai_confidence",
                                         "labelled_at": "ai_at"})[KEYS + ["category", "ai_present", "ai_confidence", "ai_at"]],
                      on=KEYS + ["category"], how="left")
    if base["ai_present"].isna().any():
        n = base[base["ai_present"].isna()][KEYS].drop_duplicates().shape[0]
        raise ValueError(f"{n} photos of split '{split}' have no AI labels yet")
    sv = silver.rename(columns={"present": "silver_present"})[KEYS + ["category", "silver_present"]]
    base = base.merge(sv, on=KEYS + ["category"], how="left")
    base["silver_present"] = base["silver_present"].fillna(0).astype(int)

    need_review = {(a, b, c) for a, b, c in zip(review["scene_id"], review["photo_id"], review["category"])}
    need_audit = {(a, b, c) for a, b in audit for c in categories}
    human, src = {}, {}
    for exp, need, label in ((review_export, need_review, "human_review"), (audit_export, need_audit, "human_audit")):
        if exp is None:
            continue
        for a, b, c, v, t in zip(exp["scene_id"], exp["photo_id"], exp["category"], exp["present"], exp["labelled_at"]):
            if (a, b, c) in need:
                human[(a, b, c)] = (int(v), t)
                src[(a, b, c)] = label
    missing = (need_review | need_audit) - set(human)
    if missing and not allow_partial:
        raise ValueError(f"{len(missing)} required human decisions are missing "
                         f"(review: {len(need_review - set(human))}, audit: {len(need_audit - set(human))})")

    rows = []
    for r in base.itertuples():
        k = (r.scene_id, r.photo_id, r.category)
        if k in human:
            rows.append((r.scene_id, r.photo_id, r.category, human[k][0], human[k][1], src[k], human[k][0]))
        else:
            rows.append((r.scene_id, r.photo_id, r.category, int(r.ai_present), r.ai_at, "ai", np.nan))
    out = pd.DataFrame(rows, columns=KEYS + ["category", "present", "labelled_at", "source", "human_present"])
    out = out.merge(base[KEYS + ["category", "ai_present", "ai_confidence", "silver_present"]],
                    on=KEYS + ["category"])
    out["corrected"] = (out["present"] != out["silver_present"]).astype(int)
    gold_df = out[KEYS + ["category", "present", "corrected", "labelled_at"]].copy()
    validate(gold_df, "gold")
    sources = out[KEYS + ["category", "source", "ai_present", "ai_confidence", "silver_present", "human_present"]]
    return gold_df, sources


def check_stats(sources: pd.DataFrame) -> dict:
    """How good were the AI labels and silver, judged by the human decisions (SPEC R-G-15).

    audit: random photos where AI and silver agreed and the AI was confident: unbiased check of those labels.
    review: pairs where AI and silver disagreed or AI was unsure: who was right?
    """
    out = {}
    a = sources[sources["source"] == "human_audit"]
    rows = []
    for cat, g in a.groupby("category"):
        rows.append({"category": cat, "rows": len(g),
                     "ai_vs_human_agreement": float((g["ai_present"] == g["human_present"]).mean()),
                     "ai_vs_human_kappa": cohens_kappa(g["ai_present"], g["human_present"])})
    if len(a):
        rows.append({"category": "ALL", "rows": len(a),
                     "ai_vs_human_agreement": float((a["ai_present"] == a["human_present"]).mean()),
                     "ai_vs_human_kappa": cohens_kappa(a["ai_present"], a["human_present"])})
    out["audit"] = pd.DataFrame(rows, columns=["category", "rows", "ai_vs_human_agreement", "ai_vs_human_kappa"])
    r = sources[sources["source"] == "human_review"]
    rows = []
    for cat, g in list(r.groupby("category")) + [("ALL", r)]:
        if len(g):
            rows.append({"category": cat, "rows": len(g), "human_sided_with_ai": float((g["human_present"] == g["ai_present"]).mean()),
                         "human_sided_with_silver": float((g["human_present"] == g["silver_present"]).mean())})
    out["review"] = pd.DataFrame(rows, columns=["category", "rows", "human_sided_with_ai", "human_sided_with_silver"])
    n = len(sources)
    out["human_share"] = float((sources["source"] != "ai").mean()) if n else float("nan")
    return out


# ---- View guard: no AI label without a produced view (SPEC R-G-2) ----------------------------
CODE_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"      # no 0/O/1/I, easy to read off an image


def new_code(n: int = 4) -> str:
    """Random code burned into the images, so a label proves the images were actually seen."""
    return "".join(secrets.choice(CODE_ALPHABET) for _ in range(n))


def stamp_code(img, code: str):
    """Burn the code into the top-left corner: white text on a black box, ~8% of the image height."""
    from PIL import ImageDraw, ImageFont

    img = img.copy()
    size = max(16, round(img.height * 0.08))
    try:
        font = ImageFont.truetype("arial.ttf", size)
    except OSError:
        font = ImageFont.load_default(size)
    draw = ImageDraw.Draw(img)
    x0, y0, x1, y1 = draw.textbbox((0, 0), code, font=font)
    pad = size // 4
    draw.rectangle((0, 0, x1 - x0 + 2 * pad, y1 - y0 + 2 * pad), fill="black")
    draw.text((pad - x0, pad - y0), code, fill="white", font=font)
    return img


def log_view(path, scene: str, photo: str, code: str, now: str | None = None) -> None:
    """Remember the code written into the latest images of this photo."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    now = now or pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
    new = not path.exists()
    with open(path, "a", encoding="utf-8") as f:
        if new:
            f.write("scene_id,photo_id,code,viewed_at\n")
        f.write(f"{scene},{photo},{code},{now}\n")


def code_ok(path, scene: str, photo: str, code: str | None) -> bool:
    """True only if code equals the LATEST logged code for this photo (case-insensitive)."""
    path = Path(path)
    if not code or not path.exists():
        return False
    v = pd.read_csv(path, dtype=str)
    v = v[(v["scene_id"] == scene) & (v["photo_id"] == photo)]
    return bool(len(v)) and v["code"].iloc[-1].upper() == code.strip().upper()
