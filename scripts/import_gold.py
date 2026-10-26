"""Turn the labelling page's CSV export into data_meta/gold.csv (SPEC 4.5b), or compare two passes.

  python scripts/import_gold.py path/to/gold_pass1_export.csv              # complete labels required
  python scripts/import_gold.py path/to/gold_pass1_export.csv --allow-partial --dry-run   # progress check
  python scripts/import_gold.py --agreement path/to/gold_pass2_export.csv  # pass 2 vs data_meta/gold.csv
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from ffs import config, gold  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("export", help="CSV downloaded from the labelling page")
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
    export = pd.read_csv(args.export)

    if args.agreement:
        g1 = pd.read_csv(meta / "gold.csv")
        g2 = gold.gold_from_export(export, photos, split, silver, cats, allow_partial=True)
        res = gold.intra_rater_agreement(g1, g2)
        print(res.round(3).to_string(index=False))
        print(f"\nphotos relabelled: {g2[['scene_id', 'photo_id']].drop_duplicates().shape[0]}")
        return

    out = gold.gold_from_export(export, photos, split, silver, cats, allow_partial=args.allow_partial)
    n_photos = out[["scene_id", "photo_id"]].drop_duplicates().shape[0]
    print(f"{n_photos} photos labelled, {len(out)} rows; corrected (gold differs from silver): "
          f"{int(out.corrected.sum())} rows")
    print(out.groupby("category").agg(gold_positive=("present", "sum"), corrected=("corrected", "sum")).to_string())
    if args.dry_run:
        return
    out.to_csv(meta / "gold.csv", index=False)
    print(f"Wrote {meta / 'gold.csv'}")


if __name__ == "__main__":
    main()
