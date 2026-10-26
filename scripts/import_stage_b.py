"""Check and store the CSVs downloaded from Kaggle (SPEC section 5).

  python scripts/import_stage_b.py path/to/output_folder --name dev

Copies clip_scores_<name>.csv and owlv2_raw_<name>.csv into results/, after checking the columns
(data contracts) and that every photo of the split has scores. Also copies run_info_<name>.json.
"""

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from ffs import config  # noqa: E402
from ffs.schemas import validate  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder")
    ap.add_argument("--name", required=True, choices=["dev", "test"])
    args = ap.parse_args()
    src = Path(args.folder)

    photos = pd.read_csv(config.META_DIR / "photos.csv")
    split = pd.read_csv(config.META_DIR / "scene_split.csv")
    scenes = set(split.loc[split["split"] == args.name, "scene_id"])
    want = set(zip(*photos[photos.scene_id.isin(scenes)][["scene_id", "photo_id"]].values.T))

    clip = pd.read_csv(src / f"clip_scores_{args.name}.csv")
    owl = pd.read_csv(src / f"owlv2_raw_{args.name}.csv")
    validate(clip, "clip_scores")
    validate(owl, "owlv2_raw")
    got = set(zip(clip.scene_id, clip.photo_id))
    if got != want:
        sys.exit(f"CLIP scores cover {len(got)} photos, the {args.name} split has {len(want)}. "
                 f"Missing: {len(want - got)}, unexpected: {len(got - want)}.")
    n_owl = owl[["scene_id", "photo_id"]].drop_duplicates().shape[0]
    print(f"clip_scores: {len(clip)} rows, {len(got)} photos | owlv2_raw: {len(owl)} boxes in {n_owl} photos "
          f"(photos with no box above the cut-off are fine)")

    config.RESULTS_DIR.mkdir(exist_ok=True)
    for name in (f"clip_scores_{args.name}.csv", f"owlv2_raw_{args.name}.csv", f"run_info_{args.name}.json"):
        if (src / name).exists():
            shutil.copy(src / name, config.RESULTS_DIR / name)
    print("Stored in results/.")


if __name__ == "__main__":
    main()
