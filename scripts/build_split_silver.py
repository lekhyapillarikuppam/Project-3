"""Build scene_split.csv and silver.csv, then print the tables used to choose the final categories.

  python scripts/build_split_silver.py

Needs data_meta/exhibits.csv, exhibit_categories.csv, photos.csv and markers.csv
(build_exhibits.py, build_categories.py, build_markers.py). The seed is config.SEED and is never re-drawn
(decision D-017). Nothing is dropped here: every candidate category gets silver rows, and the table shows
which ones pass the inclusion rule on the test split (SPEC 3.2).
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from ffs import config  # noqa: E402
from ffs.dataprep import split_scenes  # noqa: E402
from ffs.schemas import validate  # noqa: E402
from ffs.silver import build_silver, category_counts, uncertain_photos  # noqa: E402


def build_scene_split(photos: pd.DataFrame, exhibits: pd.DataFrame) -> pd.DataFrame:
    """One row per scene that has photos (R-P-4). A scene with no exhibit list has no markers."""
    split = split_scenes(photos["scene_id"].unique())
    n_photos = photos.groupby("scene_id").size()
    listed = set(exhibits["scene_id"])
    rows = [{"scene_id": s, "split": split[s], "has_markers": int(s in listed), "n_photos": int(n_photos[s])}
            for s in sorted(split)]
    return pd.DataFrame(rows, columns=["scene_id", "split", "has_markers", "n_photos"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--meta", default=str(config.META_DIR))
    ap.add_argument("--candidates-out", default=str(config.RESULTS_DIR / "phase2" / "silver_candidates.csv"))
    args = ap.parse_args()
    meta = Path(args.meta)

    photos, markers = pd.read_csv(meta / "photos.csv"), pd.read_csv(meta / "markers.csv")
    exhibits, cats = pd.read_csv(meta / "exhibits.csv"), pd.read_csv(meta / "exhibit_categories.csv")

    scene_split = build_scene_split(photos, exhibits)
    validate(scene_split, "scene_split")
    scene_split.to_csv(meta / "scene_split.csv", index=False)

    # all candidates first: this table is where the final categories come from (D-020)
    candidates = build_silver(photos, markers, cats, list(config.CANDIDATE_CATEGORIES))
    Path(args.candidates_out).parent.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(args.candidates_out, index=False)
    silver = candidates[candidates.category.isin(config.FINAL_CATEGORIES)].reset_index(drop=True)
    validate(silver, "silver")
    silver.to_csv(meta / "silver.csv", index=False)

    split_of = dict(zip(scene_split.scene_id, scene_split.split))
    n_scenes = scene_split.split.value_counts().to_dict()
    print(f"scene_split.csv: {len(scene_split)} scenes {n_scenes}; {int((scene_split.has_markers == 0).sum())} without markers")
    print(f"silver.csv: {len(silver)} rows = {len(photos)} photos x {len(config.FINAL_CATEGORIES)} final categories "
          f"(all-candidate version: {args.candidates_out})\n")

    print("Positive photos and scenes per candidate category (silver), and the inclusion rule on test "
          f"({config.MIN_POSITIVE_PHOTOS} photos, {config.MIN_POSITIVE_SCENES} scenes):")
    table = category_counts(candidates, scene_split)
    print(table.to_string(index=False))
    passing = tuple(c for c in config.CANDIDATE_CATEGORIES if c in set(table[table.passes_test].category))
    same = passing == tuple(c for c in config.CANDIDATE_CATEGORIES if c in config.FINAL_CATEGORIES)
    print(f"\nPassing the rule: {list(passing)}")
    print("FINAL_CATEGORIES in config.py " + ("matches the table." if same else
          "DOES NOT MATCH the table: update config.FINAL_CATEGORIES and log it (D-020)."))
    print("Positive rows per final category in silver.csv:",
          silver[silver.present == 1].category.value_counts().reindex(list(config.FINAL_CATEGORIES)).to_dict())

    unc = uncertain_photos(markers)
    p = photos.assign(split=photos.scene_id.map(split_of),
                      uncertain=[(s, ph) in unc for s, ph in zip(photos.scene_id, photos.photo_id)])
    by = p.groupby("split").agg(photos=("photo_id", "size"), uncertain=("uncertain", "sum"))
    by["share"] = (by.uncertain / by.photos).round(2)
    print("\nUncertain photos (at least one rejected box), by split:")
    print(by.to_string())
    print("\nPhoto types:", photos.photo_type.value_counts().to_dict())
    print("Photo types by split:")
    print(pd.crosstab(p.split, p.photo_type).to_string())


if __name__ == "__main__":
    main()
