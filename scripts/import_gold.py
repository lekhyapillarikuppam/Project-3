"""Build data_meta/gold.csv from the AI labels and your checks (SPEC 4.7), or the older all-by-hand import.

  python scripts/import_gold.py --merge results/gold/gold_review_export.csv results/gold/gold_audit_export.csv
  python scripts/import_gold.py --merge <review.csv> <audit.csv> --allow-partial --dry-run     # progress check
  python scripts/import_gold.py path/to/gold_pass1_export.csv                                  # older mode
  python scripts/import_gold.py --agreement path/to/gold_pass2_export.csv                      # older pass 2 check

--merge writes data_meta/gold.csv (contract) and data_meta/gold_sources.csv (who decided each row) and prints
how good the AI labels and silver were according to your decisions.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from ffs import config, gold  # noqa: E402


def merge_mode(args, meta, photos, split, silver, cats):
    ai = pd.read_csv(meta / "gold_ai.csv").fillna({"note": ""})
    review = pd.read_csv(meta / "gold_review_pairs.csv")
    audit = [tuple(x) for x in pd.read_csv(meta / "gold_audit_photos.csv")[["scene_id", "photo_id"]].values]
    rev_exp = pd.read_csv(args.merge[0]) if Path(args.merge[0]).exists() else None
    aud_exp = pd.read_csv(args.merge[1]) if Path(args.merge[1]).exists() else None
    g, src = gold.merge_gold(ai, silver, review, audit, rev_exp, aud_exp, photos, split, cats,
                             allow_partial=args.allow_partial)
    st = gold.check_stats(src)
    print(f"{len(g)} gold rows. Decided by: " + ", ".join(f"{k} {v}" for k, v in src.source.value_counts().items())
          + f"  (human-decided share {st['human_share']:.1%})")
    print("\nAudit (random photos where AI and silver agreed): how often was the AI wrong?")
    print(st["audit"].round(3).to_string(index=False) if len(st["audit"]) else "  (no audit decisions yet)")
    print("\nReview (AI and silver disagreed, or AI unsure): who did you side with?")
    print(st["review"].round(3).to_string(index=False) if len(st["review"]) else "  (no review decisions yet)")
    if args.dry_run:
        return
    g.to_csv(meta / "gold.csv", index=False)
    src.to_csv(meta / "gold_sources.csv", index=False)
    print(f"\nWrote {meta / 'gold.csv'} and {meta / 'gold_sources.csv'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("export", nargs="?", help="CSV downloaded from the all-by-hand page (older mode)")
    ap.add_argument("--merge", nargs=2, metavar=("REVIEW_CSV", "AUDIT_CSV"))
    ap.add_argument("--allow-partial", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="check and report, write nothing")
    ap.add_argument("--agreement", action="store_true", help="export is pass 2; compare with data_meta/gold.csv")
    ap.add_argument("--meta", default=str(config.META_DIR))
    args = ap.parse_args()
    meta = Path(args.meta)

    photos = pd.read_csv(meta / "photos.csv")
    split = pd.read_csv(meta / "scene_split.csv")
    silver = pd.read_csv(meta / "silver.csv")
    cats = gold.ordered_categories(silver)

    if args.merge:
        return merge_mode(args, meta, photos, split, silver, cats)
    if not args.export:
        ap.error("give an export CSV, or use --merge")
    export = pd.read_csv(args.export)

    if args.agreement:
        g1 = pd.read_csv(meta / "gold.csv")
        g2 = gold.gold_from_export(export, photos, split, silver, cats, allow_partial=True)
        print(gold.intra_rater_agreement(g1, g2).round(3).to_string(index=False))
        return

    out = gold.gold_from_export(export, photos, split, silver, cats, allow_partial=args.allow_partial)
    n_photos = out[["scene_id", "photo_id"]].drop_duplicates().shape[0]
    print(f"{n_photos} photos labelled, {len(out)} rows; corrected (gold differs from silver): {int(out.corrected.sum())} rows")
    print(out.groupby("category").agg(gold_positive=("present", "sum"), corrected=("corrected", "sum")).to_string())
    if args.dry_run:
        return
    out.to_csv(meta / "gold.csv", index=False)
    print(f"Wrote {meta / 'gold.csv'}")


if __name__ == "__main__":
    main()
