"""Build the labelling pages for gold labels (SPEC 4.7).

  python scripts/make_gold_tool.py --review        # you decide only the categories listed for each photo (blind)
  python scripts/make_gold_tool.py --audit 60      # 60 random photos where everything agreed: you label all categories
  python scripts/make_gold_tool.py                 # (older mode) all test photos by hand
  python scripts/make_gold_tool.py --pass2 55      # (older mode) blind re-label of 55 random test photos

--review and --audit need data_meta/gold_ai.csv (the AI labels, complete for all test photos). The page never
shows silver or the AI's labels. Open the .html in Chrome. Labels are saved in the browser as you go and
downloaded as a CSV every 100 photos and when you press S.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from ffs import config, gold  # noqa: E402

TEMPLATE = Path(__file__).resolve().parents[1] / "src" / "ffs" / "gold_tool_template.html"


def write_page(data: dict, out: Path):
    out.parent.mkdir(parents=True, exist_ok=True)
    html = TEMPLATE.read_text(encoding="utf-8").replace("__DATA__", json.dumps(data).replace("</", "<\\/"))
    out.write_text(html, encoding="utf-8")
    n_dec = sum(len(p.get("show", data["cats"])) for p in data["photos"])
    print(f"Wrote {out}  ({len(data['photos'])} photos, {n_dec} yes/no decisions)")
    print("Open it in Chrome. Photos are read from ../../data/raw/ relative to this file.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--review", action="store_true")
    ap.add_argument("--audit", type=int, default=None, metavar="N")
    ap.add_argument("--pass2", type=int, default=None, metavar="N")
    ap.add_argument("--meta", default=str(config.META_DIR))
    args = ap.parse_args()
    meta = Path(args.meta)

    photos = pd.read_csv(meta / "photos.csv")
    split = pd.read_csv(meta / "scene_split.csv")
    silver = pd.read_csv(meta / "silver.csv")
    exhibits = pd.read_csv(meta / "exhibits.csv")
    cats = gold.ordered_categories(silver)
    gdir = config.RESULTS_DIR / "gold"

    if args.review or args.audit:
        ai = pd.read_csv(meta / "gold_ai.csv").fillna({"note": ""})
        gold.validate_ai(ai, cats)
        test_scenes = set(split.loc[split["split"] == "test", "scene_id"])
        want = photos[photos["scene_id"].isin(test_scenes)]
        n_missing = len(want) - ai[["scene_id", "photo_id"]].drop_duplicates().shape[0]
        if n_missing:
            sys.exit(f"The AI has not labelled {n_missing} test photos yet. Finish gold_ai.csv first.")
        review = gold.review_pairs(ai, silver)
        review.to_csv(meta / "gold_review_pairs.csv", index=False)       # kept for provenance
        if args.review:
            show = {}
            for r in review.itertuples():
                show.setdefault((r.scene_id, r.photo_id), []).append(r.category)
            data = gold.tool_data_for_pairs(photos, exhibits, cats, show, "ffs_gold_review", "gold_review_export.csv")
            n_dis = int((review.reason == "disagree").sum())
            print(f"Review set: {len(review)} photo x category pairs in {len(show)} photos "
                  f"({n_dis} AI-vs-silver disagreements, {len(review) - n_dis} low-confidence only)")
            write_page(data, gdir / "gold_tool_review.html")
        else:
            path = meta / "gold_audit_photos.csv"
            if path.exists():                                              # the audit set is drawn once
                aud = [tuple(x) for x in pd.read_csv(path)[["scene_id", "photo_id"]].values]
                print(f"Using the existing audit set ({len(aud)} photos) from {path}")
            else:
                aud = gold.audit_photos(ai, review, args.audit)
                pd.DataFrame(aud, columns=["scene_id", "photo_id"]).to_csv(path, index=False)
                print(f"Drew a new audit set of {len(aud)} photos (seed {config.SEED}); saved to {path}")
            show = {k: list(cats) for k in aud}
            data = gold.tool_data_for_pairs(photos, exhibits, cats, show, "ffs_gold_audit", "gold_audit_export.csv")
            write_page(data, gdir / "gold_tool_audit.html")
        return

    if args.pass2:
        data = gold.build_tool_data(photos, split, exhibits, cats, sample=args.pass2,
                                    storage_key="ffs_gold_pass2", export_name="gold_pass2_export.csv")
        write_page(data, gdir / "gold_tool_pass2.html")
    else:
        data = gold.build_tool_data(photos, split, exhibits, cats)
        write_page(data, gdir / "gold_tool_pass1.html")


if __name__ == "__main__":
    main()
