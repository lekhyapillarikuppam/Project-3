"""Tune Stage B thresholds on DEV only and write protocol/protocol_v1.json (SPEC D-029).

  python scripts/tune_stage_b.py

Reads results/clip_scores_dev.csv, results/owlv2_raw_dev.csv, results/run_info_dev.json and the silver
labels of dev scenes. Never opens gold files, the test split or exhibit lists.
"""

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ffs import config  # noqa: E402
from ffs import stage_b_eval as ev  # noqa: E402

CATS = config.FINAL_CATEGORIES


def load_run_script():
    spec = importlib.util.spec_from_file_location("stage_b_run", ROOT / "kaggle" / "stage_b_run.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def evaluate(scores: pd.DataFrame, thresholds: dict) -> pd.DataFrame:
    """Per category: positives, precision, recall, F1, AP, bootstrap CI of F1."""
    d = ev.decide(scores, thresholds)
    rows = []
    for cat in CATS:
        g = d[d.category == cat]
        y, pred = g.y.to_numpy(), g.pred.to_numpy()
        p, r, f = ev.prf(y, pred)
        lo, hi = ev.cluster_bootstrap_f1(g)
        rows.append(dict(category=cat, positives=int(y.sum()), threshold=thresholds[cat], precision=p,
                         recall=r, f1=f, f1_lo=lo, f1_hi=hi, ap=ev.ap(y, g.score.to_numpy())))
    return pd.DataFrame(rows)


def main():
    res = config.RESULTS_DIR
    split = pd.read_csv(config.META_DIR / "scene_split.csv")
    dev_scenes = set(split.loc[split.split == "dev", "scene_id"])
    silver = pd.read_csv(config.META_DIR / "silver.csv")
    silver = silver[silver.scene_id.isin(dev_scenes)][["scene_id", "photo_id", "category", "present"]]
    silver = silver.rename(columns={"present": "y"})
    photos = silver[["scene_id", "photo_id"]].drop_duplicates()

    run_info = json.loads((res / "run_info_dev.json").read_text(encoding="utf-8"))
    code = load_run_script()
    assert run_info["clip_prompts"] == code.CLIP_PROMPTS and run_info["owl_queries"] == code.OWL_QUERIES, \
        "prompts in run_info_dev.json differ from kaggle/stage_b_run.py"
    prompts_hash = hashlib.sha256(json.dumps({"clip": code.CLIP_PROMPTS, "owl": code.OWL_QUERIES},
                                             sort_keys=True).encode()).hexdigest()

    clip = pd.read_csv(res / "clip_scores_dev.csv")
    owl = pd.read_csv(res / "owlv2_raw_dev.csv")

    pos_any = silver.groupby(["scene_id", "photo_id"]).y.max()
    print(f"dev photos: {len(photos)}, scenes: {silver.scene_id.nunique()}, "
          f"photos with no silver positive at all: {int((pos_any == 0).sum())}")
    print("silver positive photos:", silver.groupby("category").y.sum().to_dict(), "\n")

    # (a) CLIP
    clip_s = ev.clip_photo_scores(clip).merge(silver, on=["scene_id", "photo_id", "category"])
    clip_th, clip_flag = ev.choose_thresholds(clip_s, ev.CLIP_GRID)
    clip_tab = evaluate(clip_s, clip_th)

    # (b) OWLv2: compare tile options
    options = {}
    for opt in ("full", "all"):
        s = ev.owl_photo_scores(owl, photos, opt).merge(silver, on=["scene_id", "photo_id", "category"])
        th, flag = ev.choose_thresholds(s, ev.OWL_GRID)
        options[opt] = (s, th, flag, evaluate(s, th))
        t = options[opt][3]
        print(f"OWLv2 option {opt!r}: macro F1 {t.f1.mean():.3f}, macro AP {t.ap.mean():.3f}")
    chosen = max(options, key=lambda o: (round(options[o][3].f1.mean(), 4), options[o][3].ap.mean()))
    owl_s, owl_th, owl_flag, owl_tab = options[chosen]
    print(f"chosen tile option: {chosen!r}\n")

    for name, tab in (("CLIP", clip_tab), ("OWLv2 (all)", options["all"][3]), ("OWLv2 (full)", options["full"][3])):
        print(name)
        print(tab.round(3).to_string(index=False), "\n")

    table = pd.concat([clip_tab.assign(model="clip"), owl_tab.assign(model=f"owlv2_{chosen}")])
    table.to_csv(res / "stage_b_dev_table.csv", index=False)

    protocol = {
        "status": "chosen on dev only; not frozen until git tag protocol-v1",
        "seed": config.SEED,
        "categories": list(CATS),
        "clip": {"model": run_info["clip_model"], "photo_score":
                 "softmax across the 5 categories of the mean of each category's 3 prompt logits",
                 "thresholds": clip_th, "grid": ev.CLIP_GRID, "untuned_categories": clip_flag},
        "owlv2": {"model": run_info["owl_model"], "tile_option": "tile 0 only" if chosen == "full" else "tiles 0-4",
                  "tiles_used": [0] if chosen == "full" else [0, 1, 2, 3, 4],
                  "photo_score": "max box score of the category over the used tiles",
                  "thresholds": owl_th, "grid": ev.OWL_GRID, "untuned_categories": owl_flag,
                  "nms_iou": ev.NMS_IOU, "box_merge": "NMS across prompts and tiles per photo and category"},
        "photo_rule": "present when photo score >= category threshold",
        "threshold_selection": "grid search maximising photo-level F1 on dev; ties go to the lowest threshold",
        "min_positives_to_tune": ev.MIN_POSITIVES_TO_TUNE,
        "bootstrap": {"n": config.N_BOOTSTRAP, "unit": "scene", "seed": config.SEED},
        "prompts_sha256": prompts_hash,
        "run_info": {k: run_info[k] for k in ("clip_model", "owl_model", "tile_grid", "tile_overlap",
                                              "raw_cutoff", "top_k", "torch", "transformers")},
    }
    out = ROOT / "protocol"
    out.mkdir(exist_ok=True)
    (out / "protocol_v1.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    print("wrote protocol/protocol_v1.json and results/stage_b_dev_table.csv")


if __name__ == "__main__":
    main()
