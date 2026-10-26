"""Make viewable images of one test photo for the AI labeller (SPEC 4.7, R-G-10).

  python scripts/ai_label_view.py th_0002 IMG_20240227_085541            # full photo, long side 1568 px
  python scripts/ai_label_view.py th_0002 IMG_20240227_085541 --tiles    # plus 4 zoomed quarter tiles

Writes JPEGs under results/gold/ai_views/ and prints their paths. Only the photo is used: no silver,
markers, exhibit lists or other labels. Only test-split photos are accepted.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402
from PIL import Image, ImageOps  # noqa: E402

from ffs import config  # noqa: E402

LONG_SIDE = 1568


def fit(img: Image.Image, side: int = LONG_SIDE) -> Image.Image:
    k = side / max(img.size)
    return img if k >= 1 else img.resize((round(img.width * k), round(img.height * k)), Image.LANCZOS)


def quarter_boxes(w: int, h: int, overlap: float = 0.15):
    tw, th = w * (0.5 + overlap / 2), h * (0.5 + overlap / 2)
    xs, ys = [0, w - tw], [0, h - th]
    return [(round(x), round(y), round(x + tw), round(y + th)) for y in ys for x in xs]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scene")
    ap.add_argument("photo")
    ap.add_argument("--tiles", action="store_true")
    args = ap.parse_args()

    photos = pd.read_csv(config.META_DIR / "photos.csv")
    split = pd.read_csv(config.META_DIR / "scene_split.csv")
    if split.loc[split.scene_id == args.scene, "split"].tolist() != ["test"]:
        sys.exit(f"{args.scene} is not a test scene.")
    row = photos[(photos.scene_id == args.scene) & (photos.photo_id == args.photo)]
    if row.empty:
        sys.exit("No such photo.")
    img = ImageOps.exif_transpose(Image.open(config.RAW_DIR / row.iloc[0].relpath)).convert("RGB")

    out = config.RESULTS_DIR / "gold" / "ai_views"
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{args.scene}__{args.photo}"
    paths = [out / f"{stem}_full.jpg"]
    fit(img).save(paths[0], "JPEG", quality=90)
    if args.tiles:
        for i, box in enumerate(quarter_boxes(*img.size), 1):
            p = out / f"{stem}_t{i}.jpg"
            fit(img.crop(box)).save(p, "JPEG", quality=90)
            paths.append(p)
    for p in paths:
        print(p)


if __name__ == "__main__":
    main()
