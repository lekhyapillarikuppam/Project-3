"""Contact sheet to check photo types by eye (SPEC R-P-5).

  python scripts/phototype_check.py

Picks 20 random photos (seed config.SEED) and writes results/phase2/phototype_check.png. Each thumbnail
shows the photo's type and the height of its largest marker box as a share of the photo (the box is drawn).
"""

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ffs import config  # noqa: E402

THUMB_W, THUMB_H, COLS = 360, 480, 5


def thumbnail(path: Path, photo: pd.Series, boxes: pd.DataFrame) -> np.ndarray:
    img = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)
    s = min(THUMB_W / img.shape[1], THUMB_H / img.shape[0])
    small = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    label = f"{photo.photo_type}"
    if len(boxes):
        b = boxes.loc[boxes.h.idxmax()]
        cv2.rectangle(small, (int(b.x * s), int(b.y * s)), (int((b.x + b.w) * s), int((b.y + b.h) * s)), (0, 0, 255), 2)
        label += f"  {b.h / photo.height:.2f}"
    canvas = np.full((THUMB_H + 28, THUMB_W, 3), 40, np.uint8)
    canvas[28:28 + small.shape[0], :small.shape[1]] = small
    cv2.putText(canvas, label, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return canvas


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--meta", default=str(config.META_DIR))
    ap.add_argument("--out", default=str(config.RESULTS_DIR / "phase2" / "phototype_check.png"))
    args = ap.parse_args()

    photos = pd.read_csv(Path(args.meta) / "photos.csv")
    markers = pd.read_csv(Path(args.meta) / "markers.csv")
    pick = random.Random(config.SEED).sample(range(len(photos)), args.n)
    tiles = []
    for i in pick:
        p = photos.iloc[i]
        boxes = markers[(markers.scene_id == p.scene_id) & (markers.photo_id == p.photo_id)]
        tiles.append(thumbnail(config.RAW_DIR / p.relpath, p, boxes))
        print(f"{len(tiles):2d}  {p.scene_id} {p.photo_id}  {p.photo_type}")
    while len(tiles) % COLS:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[r:r + COLS]) for r in range(0, len(tiles), COLS)]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(args.out, np.vstack(rows))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
