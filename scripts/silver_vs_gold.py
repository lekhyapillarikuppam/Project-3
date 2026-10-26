"""Silver against gold on the test split: how trustworthy is the marker-based answer key? (SPEC R-G-7, R-G-8)

  python scripts/silver_vs_gold.py
  python scripts/silver_vs_gold.py --boot 200      # quicker, for a look

Needs data_meta/gold.csv (python scripts/import_gold.py ...), silver.csv, scene_split.csv, photos.csv and
markers.csv. Writes results/silver_vs_gold.csv and prints the table. Intervals are 95% scene-cluster bootstrap.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from ffs import config  # noqa: E402
from ffs.silver_gold import silver_vs_gold  # noqa: E402

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
    ap.add_argument("--out", default=str(config.RESULTS_DIR / "silver_vs_gold.csv"))
    ap.add_argument("--boot", type=int, default=config.N_BOOTSTRAP, help="bootstrap resamples (default 1000)")
    args = ap.parse_args()
    meta = Path(args.meta)

    if not (meta / "gold.csv").exists():
        sys.exit("data_meta/gold.csv does not exist yet. Label the test photos and run scripts/import_gold.py first.")
    gold, silver = pd.read_csv(meta / "gold.csv"), pd.read_csv(meta / "silver.csv")
    split, photos = pd.read_csv(meta / "scene_split.csv"), pd.read_csv(meta / "photos.csv")
    markers = pd.read_csv(meta / "markers.csv")

    table = silver_vs_gold(gold, silver, split, photos, markers, n_boot=args.boot, seed=config.SEED)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)

    print(f"Silver (prediction) against gold (truth), test split, {args.boot} scene-bootstrap resamples, seed {config.SEED}\n")
    print("Overall:")
    print(show(table[table.slice == "overall"]))
    for name in ("photo_type", "certainty", "counted_marker"):
        print(f"\nBy {name}:")
        print(show(table[table.slice == name]))
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
