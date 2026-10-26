"""Silver against gold on the test split: how trustworthy is the marker-based answer key? (SPEC R-G-7, R-G-8)

  python scripts/silver_vs_gold.py
  python scripts/silver_vs_gold.py --boot 200      # quicker, for a look
  python scripts/silver_vs_gold.py --human-only    # audit photos only (source human_audit in gold_sources.csv)

Needs data_meta/gold.csv (and gold_sources.csv for --human-only) (python scripts/import_gold.py ...), silver.csv, scene_split.csv, photos.csv and
markers.csv. Writes results/silver_vs_gold.csv and prints the table. Intervals are 95% scene-cluster bootstrap.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from ffs import config  # noqa: E402
from ffs.silver_gold import human_audit_rows, mode_name, silver_vs_gold  # noqa: E402

KEPT = ["TP", "FP", "FN", "TN"]


def show(t: pd.DataFrame) -> str:
    """Compact text: 'estimate [low, high]' for each metric."""
    out = t[["slice", "value", "category", "n_photos", "gold_prevalence"] + KEPT].copy()
    for stat in ("precision", "recall", "f1", "kappa"):
        out[stat] = [f"{e:.2f} [{lo:.2f}, {hi:.2f}]" for e, lo, hi in zip(t[stat], t[f"{stat}_lo"], t[f"{stat}_hi"])]
    out["gold_prevalence"] = out["gold_prevalence"].round(2)
    return out.to_string(index=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--meta", default=str(config.META_DIR))
    ap.add_argument("--out", default=None, help="default results/silver_vs_gold.csv, or ..._human_only.csv with --human-only")
    ap.add_argument("--boot", type=int, default=config.N_BOOTSTRAP, help="bootstrap resamples (default 1000)")
    ap.add_argument("--human-only", action="store_true", help="use only the audit photos, where every label is the human's (R-G-12 b)")
    args = ap.parse_args()
    meta = Path(args.meta)
    if args.out is None:
        args.out = str(config.RESULTS_DIR / ("silver_vs_gold_human_only.csv" if args.human_only else "silver_vs_gold.csv"))

    if not (meta / "gold.csv").exists():
        sys.exit("data_meta/gold.csv does not exist yet. Label the test photos and run scripts/import_gold.py first.")
    gold, silver = pd.read_csv(meta / "gold.csv"), pd.read_csv(meta / "silver.csv")
    if args.human_only:
        if not (meta / "gold_sources.csv").exists():
            sys.exit("data_meta/gold_sources.csv does not exist yet; --human-only needs it (scripts/import_gold.py --merge).")
        gold = human_audit_rows(gold, pd.read_csv(meta / "gold_sources.csv"))
        if gold.empty:
            sys.exit("No rows with source human_audit in gold_sources.csv.")
    split, photos = pd.read_csv(meta / "scene_split.csv"), pd.read_csv(meta / "photos.csv")
    markers = pd.read_csv(meta / "markers.csv")

    table = silver_vs_gold(gold, silver, split, photos, markers, n_boot=args.boot, seed=config.SEED)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)

    print(f"Mode: {mode_name(args.human_only)}")
    print(f"Silver (prediction) against gold (truth), test split, {args.boot} scene-bootstrap resamples, seed {config.SEED}\n")
    print("Overall:")
    print(show(table[table.slice == "overall"]))
    for name in ("photo_type", "certainty", "counted_marker"):
        print(f"\nBy {name}:")
        print(show(table[table.slice == name]))
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
