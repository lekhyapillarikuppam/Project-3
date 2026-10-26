"""Record the AI labeller's labels for one photo, or show progress (SPEC 4.7, R-G-10 and R-G-11).

  python scripts/ai_label_write.py --next 8                      # the next 8 unlabelled test photos
  python scripts/ai_label_write.py th_0002 IMG_x --labels ammunition=0,blade=1,bottle_glass=0,paper_document=0,red_brown_stain=1 \
        --conf med --note "knife on the sofa; stain on the floor" --labeller "<your exact model id>"
  python scripts/ai_label_write.py --status

Writes data_meta/gold_ai.csv. One call sets all categories of one photo (re-running a photo replaces it).
Only test-split photos are accepted. Confidence: high = clearly sure, med = fairly sure, low = unsure or the
object would be tiny, dark or hidden. Low confidence sends the label to the human review set.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from ffs import config, gold  # noqa: E402

PATH = config.META_DIR / "gold_ai.csv"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scene", nargs="?")
    ap.add_argument("photo", nargs="?")
    ap.add_argument("--labels", help="category=0/1 pairs, comma separated, all categories")
    ap.add_argument("--conf", choices=gold.CONFIDENCES)
    ap.add_argument("--note", default="")
    ap.add_argument("--labeller", help="the exact model id of the AI doing the labelling")
    ap.add_argument("--next", type=int, metavar="N")
    ap.add_argument("--status", action="store_true")
    args = ap.parse_args()

    photos = pd.read_csv(config.META_DIR / "photos.csv")
    split = pd.read_csv(config.META_DIR / "scene_split.csv")
    silver = pd.read_csv(config.META_DIR / "silver.csv")           # only for the category list
    cats = gold.ordered_categories(silver)
    ai = pd.read_csv(PATH).fillna({"note": ""}) if PATH.exists() else None

    if args.next or args.status:
        todo = gold.next_unlabelled(photos, split, ai, n=10**6)
        total = int(photos.scene_id.isin(split.loc[split.split == "test", "scene_id"]).sum())
        print(f"labelled {total - len(todo)} of {total} test photos; {len(todo)} to go")
        if ai is not None and len(ai):
            print(ai.drop_duplicates(["scene_id", "photo_id"]).confidence.value_counts().to_string())
        for s, p in todo[: args.next or 0]:
            print(s, p)
        return

    if not (args.scene and args.photo and args.labels and args.conf and args.labeller):
        ap.error("need scene, photo, --labels, --conf and --labeller (or use --next / --status)")
    test = set(split.loc[split.split == "test", "scene_id"])
    if args.scene not in test or not ((photos.scene_id == args.scene) & (photos.photo_id == args.photo)).any():
        sys.exit("Not a test photo.")
    labels = {k.strip(): int(v) for k, v in (kv.split("=") for kv in args.labels.split(","))}
    gold.upsert_ai_labels(PATH, args.scene, args.photo, labels, args.conf, args.note, args.labeller, cats)
    print("saved", args.scene, args.photo)


if __name__ == "__main__":
    main()
