"""Make a fake scene 'th_9999' so you can test the whole Phase 1 pipeline with no download.

  python scripts/make_demo_scene.py
  python scripts/run_marker_probe.py --scenes th_9999

It writes 5 synthetic photos (yellow numbered markers on a cluttered background)
and an exhibit list into data/raw/th_9999/. The numbers on the markers are in
the file names' log below, so you can check the reads by eye.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv2  # noqa: E402
import pandas as pd  # noqa: E402

from ffs import config  # noqa: E402
from ffs.synthetic import cluttered_background, draw_marker  # noqa: E402

PHOTOS = {  # photo -> list of (x, y, w, h, number)
    "IMG_0001": [(600, 400, 150, 190, "14")],
    "IMG_0002": [(300, 300, 150, 190, "32"), (1000, 700, 150, 190, "11")],
    "IMG_0003": [(700, 500, 160, 200, "35")],   # 35 is not in the list: should be snapped to 33/36 or rejected
    "IMG_0004": [],                              # no marker
    "IMG_0005": [(500, 800, 140, 180, "7")],    # one single digit, as a control
}
EXHIBITS = [["Marker", "Description"], [7, "knife with black handle"], [11, "mobile phone"],
            [14, "5 white A4 sheets"], [32, "half-empty bottle of vodka"], [36, "red-brown stain"]]


def main():
    scene = config.RAW_DIR / "th_9999"
    (scene / "img").mkdir(parents=True, exist_ok=True)
    (scene / "IPC" / "IPC_ENG").mkdir(parents=True, exist_ok=True)
    for i, (name, markers) in enumerate(PHOTOS.items()):
        img = cluttered_background(1560, 2080, seed=i)
        for x, y, w, h, text in markers:
            draw_marker(img, x, y, w, h, text)
        cv2.imwrite(str(scene / "img" / f"{name}.jpg"), img)
        print(f"{name}.jpg: markers {[m[4] for m in markers] or 'none'}")
    pd.DataFrame(EXHIBITS).to_csv(scene / "IPC" / "IPC_ENG" / "th_9999_IPC_ENG.csv",
                                  header=False, index=False)
    print(f"\nWrote demo scene to {scene}")


if __name__ == "__main__":
    main()
