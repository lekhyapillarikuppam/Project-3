"""Build the hand-labelling page for gold labels (SPEC 4.5b).

  python scripts/make_gold_tool.py                 # pass 1: all test photos -> results/gold/gold_tool_pass1.html
  python scripts/make_gold_tool.py --pass2 55      # pass 2: 55 random test photos, later, blind again

Open the .html in Chrome (double-click). It shows photos from data/raw/ next to it, and never shows
the silver labels. Labels are saved in the browser as you go and downloaded as a CSV every 100 photos
and when you press S.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from ffs import config, gold  # noqa: E402

TEMPLATE = Path(__file__).resolve().parents[1] / "src" / "ffs" / "gold_tool_template.html"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pass2", type=int, default=None, metavar="N", help="relabel N random test photos")
    ap.add_argument("--meta", default=str(config.META_DIR))
    args = ap.parse_args()
    meta = Path(args.meta)

    photos = pd.read_csv(meta / "photos.csv")
    split = pd.read_csv(meta / "scene_split.csv")
    silver = pd.read_csv(meta / "silver.csv")           # only used for the category list; never shown
    exhibits = pd.read_csv(meta / "exhibits.csv")
    cats = gold.ordered_categories(silver)

    if args.pass2:
        data = gold.build_tool_data(photos, split, exhibits, cats, sample=args.pass2,
                                    storage_key="ffs_gold_pass2", export_name="gold_pass2_export.csv")
        out = config.RESULTS_DIR / "gold" / "gold_tool_pass2.html"
    else:
        data = gold.build_tool_data(photos, split, exhibits, cats)
        out = config.RESULTS_DIR / "gold" / "gold_tool_pass1.html"

    out.parent.mkdir(parents=True, exist_ok=True)
    html = TEMPLATE.read_text(encoding="utf-8").replace("__DATA__", json.dumps(data).replace("</", "<\\/"))
    out.write_text(html, encoding="utf-8")
    print(f"Wrote {out}  ({len(data['photos'])} photos, categories: {', '.join(cats)})")
    print("Open it in Chrome. Photos are read from ../../data/raw/ relative to this file.")


if __name__ == "__main__":
    main()
