"""Make the upload package for the Kaggle run (SPEC section 5).

  python scripts/package_stage_b.py --split dev      # dev photos: allowed any time
  python scripts/package_stage_b.py --split test     # test photos: only after the git tag protocol-v1 exists

Writes results/stage_b/ffs-stage-b-<split>.zip containing manifest.csv, stage_b_run.py and the photos
(shrunk so the longest side is at most 2400 px, EXIF turn applied). Upload the zip as a PRIVATE Kaggle dataset.
The photos are CC BY-NC-SA 4.0: keep the dataset private and non-commercial.
"""

import argparse
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from ffs import config  # noqa: E402

MAX_SIDE = 2400
TAG = "protocol-v1"


def protocol_frozen(tags: list[str]) -> bool:
    return TAG in [t.strip() for t in tags]


def shrink(src: Path, dest: Path, max_side: int = MAX_SIDE) -> tuple[int, int]:
    """Save a turned, shrunk copy. Returns the ORIGINAL (turned) size."""
    from PIL import Image, ImageOps

    img = ImageOps.exif_transpose(Image.open(src)).convert("RGB")
    w, h = img.size
    if max(w, h) > max_side:
        k = max_side / max(w, h)
        img = img.resize((round(w * k), round(h * k)), Image.LANCZOS)
    dest.parent.mkdir(parents=True, exist_ok=True)
    img.save(dest, "JPEG", quality=92)
    return w, h


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", required=True, choices=["dev", "test"])
    ap.add_argument("--meta", default=str(config.META_DIR))
    args = ap.parse_args()

    if args.split == "test":
        tags = subprocess.run(["git", "tag", "-l", TAG], capture_output=True, text=True,
                              cwd=config.REPO_ROOT).stdout.split()
        if not protocol_frozen(tags):
            sys.exit(f"Refusing: the git tag '{TAG}' does not exist. Test photos are not scored before the "
                     f"protocol is frozen (SPEC 6, D-017).")

    meta = Path(args.meta)
    photos = pd.read_csv(meta / "photos.csv")
    split = pd.read_csv(meta / "scene_split.csv")
    scenes = set(split.loc[split["split"] == args.split, "scene_id"])
    p = photos[photos["scene_id"].isin(scenes)].sort_values(["scene_id", "photo_id"])

    work = config.RESULTS_DIR / "stage_b" / f"ffs-stage-b-{args.split}"
    if work.exists():
        shutil.rmtree(work)
    rows = []
    for r in p.itertuples():
        rel = f"photos/{r.scene_id}/{r.photo_id}.jpg"
        w, h = shrink(config.RAW_DIR / r.relpath, work / rel)
        rows.append({"scene_id": r.scene_id, "photo_id": r.photo_id, "file": rel, "orig_w": w, "orig_h": h})
    pd.DataFrame(rows).to_csv(work / "manifest.csv", index=False)
    shutil.copy(config.REPO_ROOT / "kaggle" / "stage_b_run.py", work / "stage_b_run.py")

    zpath = work.parent / f"ffs-stage-b-{args.split}.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_STORED) as z:       # JPEGs do not compress further
        for f in sorted(work.rglob("*")):
            if f.is_file():
                z.write(f, f.relative_to(work).as_posix())
    print(f"{len(rows)} photos -> {zpath} ({zpath.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
