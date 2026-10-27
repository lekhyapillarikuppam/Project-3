"""Evaluate CLIP and OWLv2 on the TEST split with the frozen protocol (SPEC section 6).

  python scripts/evaluate_test.py              # needs: git tag protocol-v1, results/clip_scores_test.csv, owlv2_raw_test.csv,
                                               #        data_meta/gold.csv, silver.csv, protocol/protocol_v1.json
  python scripts/evaluate_test.py --B 200      # quick run with fewer bootstrap resamples

Writes to results/eval/:
  test_metrics.csv       precision, recall, F1, AP per system and category, with scene-level 95% CIs, for reference = gold and = silver
  test_paired.csv        OWLv2 minus CLIP (F1 and AP), paired over the same scene resamples
  silver_vs_gold.csv     the automatic answer key (silver) judged against gold: precision, recall, F1, kappa
  photo_predictions.csv  every test photo x category: scores, predictions, best OWLv2 box, silver, gold (for the web app)
  errors.csv             the 10 highest-scoring false positives and false negatives per model and category (error analysis)
  summary.md             the main tables in plain text
  f1_by_category.png     figure
Never changes thresholds: it reads them from the protocol file.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ffs import config, evaluation as ev  # noqa: E402

KEYS = ["scene_id", "photo_id"]


def tag_exists(tag: str) -> bool:
    try:
        r = subprocess.run(["git", "tag", "--list", tag], capture_output=True, text=True, cwd=config.META_DIR.parent)
        return tag in r.stdout.split()
    except OSError:
        return False


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--B", type=int, default=1000)
    ap.add_argument("--protocol", default=str(config.META_DIR.parent / "protocol" / "protocol_v1.json"))
    ap.add_argument("--no-tag-check", action="store_true")
    args = ap.parse_args()
    if not args.no_tag_check and not tag_exists("protocol-v1"):
        sys.exit("Git tag protocol-v1 not found. Freeze the protocol first.")

    meta, res = config.META_DIR, config.RESULTS_DIR
    photos = pd.read_csv(meta / "photos.csv")
    split = pd.read_csv(meta / "scene_split.csv")
    test_scenes = set(split.loc[split["split"] == "test", "scene_id"])
    photos = photos[photos["scene_id"].isin(test_scenes)].sort_values(KEYS).reset_index(drop=True)
    cats = list(config.FINAL_CATEGORIES)
    gold = pd.read_csv(meta / "gold.csv")
    silver = pd.read_csv(meta / "silver.csv")
    clip = pd.read_csv(res / "clip_scores_test.csv")
    owl = pd.read_csv(res / "owlv2_raw_test.csv")
    protocol = json.loads(Path(args.protocol).read_text(encoding="utf-8"))

    th_clip = ev.find_thresholds(protocol, "clip", cats)
    th_owl = ev.find_thresholds(protocol, "owl", cats)
    print("Thresholds (from the protocol file):")
    print(pd.DataFrame({"clip": th_clip, "owlv2": th_owl}).round(3).to_string())

    full = photos[KEYS].merge(pd.DataFrame({"category": cats}), how="cross")
    g = gold.rename(columns={"present": "gold"})[KEYS + ["category", "gold"]]
    full = full.merge(g, on=KEYS + ["category"], how="left")
    if full["gold"].isna().any():
        sys.exit(f"gold.csv misses {int(full['gold'].isna().sum())} test photo x category rows. Run the merge first.")
    sv = silver.rename(columns={"present": "silver"})[KEYS + ["category", "silver"]]
    full = full.merge(sv, on=KEYS + ["category"], how="left")
    full["silver"] = full["silver"].fillna(0).astype(int)
    full = full.merge(ev.clip_photo_scores(clip, photos, cats), on=KEYS + ["category"])
    full = full.merge(ev.owl_photo_scores(owl, photos, cats, tile=0), on=KEYS + ["category"])
    full["clip_pred"] = (full["clip_score"] >= full["category"].map(th_clip)).astype(int)
    full["owl_pred"] = (full["owl_score"] >= full["category"].map(th_owl)).astype(int)
    full["gold"] = full["gold"].astype(int)
    if len(full) != len(photos) * len(cats):
        sys.exit("Row count mismatch after merging; check that every test photo has scores.")

    out = res / "eval"
    out.mkdir(parents=True, exist_ok=True)
    all_m, all_p = [], []
    for ref in ("gold", "silver"):
        d = full.assign(y=full[ref])
        systems = {"CLIP": ("clip_score", "clip_pred"), "OWLv2": ("owl_score", "owl_pred")}
        m, p = ev.evaluate(d, systems, cats, B=args.B, seed=config.SEED, pairs=[("OWLv2", "CLIP")])
        m.insert(0, "reference", ref)
        p.insert(0, "reference", ref)
        all_m.append(m)
        all_p.append(p)
    metrics = pd.concat(all_m, ignore_index=True)
    paired = pd.concat(all_p, ignore_index=True)
    metrics.to_csv(out / "test_metrics.csv", index=False)
    paired.to_csv(out / "test_paired.csv", index=False)

    sg, _ = ev.evaluate(full.assign(y=full["gold"]), {"silver": (None, "silver")}, cats, B=args.B, seed=config.SEED)
    sg.to_csv(out / "silver_vs_gold.csv", index=False)

    keep = KEYS + ["category", "clip_score", "clip_pred", "owl_score", "owl_pred", "owl_prompt",
                   "box_x", "box_y", "box_w", "box_h", "silver", "gold"]
    full[keep].to_csv(out / "photo_predictions.csv", index=False)

    src_path = meta / "gold_sources.csv"
    if src_path.exists():
        srcs = pd.read_csv(src_path)[KEYS + ["category", "source"]]
        full = full.merge(srcs, on=KEYS + ["category"], how="left")
    else:
        full["source"] = ""
    err = []
    for model, sc, pc in (("CLIP", "clip_score", "clip_pred"), ("OWLv2", "owl_score", "owl_pred")):
        for c in cats:
            s = full[full.category == c]
            fp = s[(s[pc] == 1) & (s.gold == 0)].nlargest(10, sc).assign(error="false_positive")
            fn = s[(s[pc] == 0) & (s.gold == 1)].nlargest(10, sc).assign(error="false_negative")
            err.append(pd.concat([fp, fn]).assign(model=model, score=lambda x, sc=sc: x[sc]))
    pd.concat(err)[["model", "category", "error"] + KEYS + ["score", "silver", "gold", "source"]].to_csv(
        out / "errors.csv", index=False)

    def table(ref, systems):
        t = metrics[(metrics.reference == ref) & metrics.system.isin(systems)]
        t = t.assign(F1=t.apply(lambda r: f"{r.f1:.2f} [{r.f1_lo:.2f}-{r.f1_hi:.2f}]", axis=1),
                     AP=t.apply(lambda r: f"{r.ap:.2f} [{r.ap_lo:.2f}-{r.ap_hi:.2f}]", axis=1))
        return t[["system", "category", "n_pos", "precision", "recall", "F1", "AP"]].round(2).to_string(index=False)

    lines = ["TEST RESULTS (frozen protocol v1). Scene-level bootstrap, %d resamples, seed %d. [95%% CI]" % (args.B, config.SEED),
             f"{photos.scene_id.nunique()} scenes, {len(photos)} photos.", "",
             "== Against GOLD (AI-labelled, human-checked) ==", table("gold", ["CLIP", "OWLv2"]), "",
             "== OWLv2 minus CLIP, paired (gold) ==", paired[paired.reference == "gold"].drop(columns="reference").round(3).to_string(index=False), "",
             "== Against SILVER (marker-derived answer key) ==", table("silver", ["CLIP", "OWLv2"]), "",
             "== Silver answer key judged against gold ==", sg[["category", "n_pos", "precision", "recall", "f1", "f1_lo", "f1_hi", "kappa"]].round(2).to_string(index=False)]
    text = "\n".join(lines)
    (out / "summary.md").write_text(text, encoding="utf-8")
    print(text)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        m = metrics[(metrics.reference == "gold") & metrics.system.isin(["CLIP", "OWLv2"])]
        labels = cats + ["macro"]
        x = np.arange(len(labels))
        fig, ax = plt.subplots(figsize=(8, 4))
        for i, sysname in enumerate(["CLIP", "OWLv2"]):
            t = m[m.system == sysname].set_index("category").loc[labels]
            ax.bar(x + (i - 0.5) * 0.38, t.f1, 0.38, label=sysname,
                   yerr=[t.f1 - t.f1_lo, t.f1_hi - t.f1], capsize=3)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20)
        ax.set_ylabel("F1 vs gold (95% CI over scenes)")
        ax.set_ylim(0, 1)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / "f1_by_category.png", dpi=150)
    except Exception as e:  # the figure is optional
        print("figure skipped:", e)
    print(f"\nWritten to {out}")


if __name__ == "__main__":
    main()
