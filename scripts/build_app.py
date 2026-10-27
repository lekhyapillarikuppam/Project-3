"""Build the static demo site in docs/ (index.html, data.js, img/) from the saved test results.

  python scripts/build_app.py              # 30 sample photos, long side 1000 px
  python scripts/build_app.py --n 40

Needs results/eval/photo_predictions.csv and test_metrics.csv (from evaluate_test.py), data_meta/photos.csv and the raw photos.
Selection is deterministic (seed 42): 4 gold-positive photos per category, then random others up to N. Publish docs/ with GitHub Pages.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ffs import config  # noqa: E402

TEMPLATE = Path(__file__).resolve().parents[1] / "src" / "ffs" / "app_template.html"
KEYS = ["scene_id", "photo_id"]


def select_photos(pred: pd.DataFrame, cats: list[str], n: int, seed: int = config.SEED, per_cat: int = 4):
    rng = np.random.default_rng(seed)
    allp = sorted(set(zip(pred.scene_id, pred.photo_id)))
    chosen = []
    for c in cats:
        pos = sorted(set(zip(*pred[(pred.category == c) & (pred.gold == 1)][KEYS].values.T)))
        pos = [p for p in pos if p not in chosen]
        for i in rng.permutation(len(pos))[:per_cat]:
            chosen.append(pos[i])
    rest = [p for p in allp if p not in chosen]
    for i in rng.permutation(len(rest))[: max(0, n - len(chosen))]:
        chosen.append(rest[i])
    return chosen[:n]


def build(root: Path, n: int, out: Path, size: int = 1000):
    from PIL import Image
    meta, res = root / "data_meta", root / "results"
    pred = pd.read_csv(res / "eval" / "photo_predictions.csv")
    metrics = pd.read_csv(res / "eval" / "test_metrics.csv")
    sg = pd.read_csv(res / "eval" / "silver_vs_gold.csv")
    photos = pd.read_csv(meta / "photos.csv").set_index(KEYS)
    cats = list(config.FINAL_CATEGORIES)
    chosen = select_photos(pred, cats, n)
    (out / "img").mkdir(parents=True, exist_ok=True)
    items = []
    for s, p in chosen:
        row = photos.loc[(s, p)]
        img = Image.open(config.RAW_DIR / row["relpath"]).convert("RGB")
        w0, h0 = img.size
        img.thumbnail((size, size))
        name = f"{s}__{p}.jpg"
        img.save(out / "img" / name, "JPEG", quality=82)
        cats_d = {}
        for r in pred[(pred.scene_id == s) & (pred.photo_id == p)].itertuples():
            has = not pd.isna(r.box_x)
            box = [float(r.box_x) / w0, float(r.box_y) / h0, float(r.box_w) / w0, float(r.box_h) / h0] if has else None
            if box:
                box = [min(max(v, 0.0), 1.0) for v in box]
            cats_d[r.category] = dict(gold=int(r.gold), silver=int(r.silver), clip_score=float(r.clip_score),
                                      clip_pred=int(r.clip_pred), owl_score=float(r.owl_score),
                                      owl_pred=int(r.owl_pred), box=box)
        items.append(dict(scene=s, photo=p, img=f"img/{name}", type=str(row["photo_type"]), cats=cats_d))
    keep = ["system", "category", "n_pos", "precision", "recall", "f1", "f1_lo", "f1_hi", "ap", "ap_lo", "ap_hi"]
    mg = metrics[(metrics.reference == "gold") & metrics.system.isin(["CLIP", "OWLv2"])][keep]
    order = cats + ["macro"]
    mg = mg.assign(o=mg.category.map(order.index)).sort_values(["system", "o"]).drop(columns="o")
    sg = sg[["category", "n_pos", "precision", "recall", "f1", "f1_lo", "f1_hi", "kappa"]]
    sg = sg.assign(o=sg.category.map(order.index)).sort_values("o").drop(columns="o")
    data = dict(categories=cats, photos=items, metrics=json.loads(mg.to_json(orient="records")),
                sg=json.loads(sg.to_json(orient="records")), n_scenes=int(pred.scene_id.nunique()),
                n_photos=int(pred[KEYS].drop_duplicates().shape[0]),
                footer="Data: FFS, University of Lausanne, CC BY-NC-SA 4.0. Non-commercial use.")
    (out / "data.js").write_text("window.DATA = " + json.dumps(data) + ";\n", encoding="utf-8")
    shutil.copy(TEMPLATE, out / "index.html")
    (out / ".nojekyll").write_text("", encoding="utf-8")
    return len(items)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--out", default=str(config.META_DIR.parent / "docs"))
    args = ap.parse_args()
    k = build(config.META_DIR.parent, args.n, Path(args.out))
    print(f"Built the demo site with {k} photos in {args.out}. Open {Path(args.out) / 'index.html'} in Chrome to check it.")


if __name__ == "__main__":
    main()
