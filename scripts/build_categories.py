"""Apply category_rules.csv to exhibits.csv (SPEC 4.6, R-P-2 and R-P-3).

  python scripts/build_categories.py

Writes:
  data_meta/exhibit_categories.csv   one row per exhibit and category: rule rows, plus the llm/manual row of
                                     every exhibit no rule matches. Existing rows keep their `confirmed` value;
                                     NEW suggestions are unconfirmed (confirmed = 0) until you confirm them.
  results/phase2/unmatched.csv       the exhibits no rule matches, with their category and source
Prints the NEW unmatched descriptions and exhibit-level counts per category (exhibits and scenes), counting
only rule rows and confirmed rows. Unconfirmed llm rows count as `other` downstream (silver ignores them).
"""

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from ffs import config  # noqa: E402
from ffs.dataprep import build_exhibit_categories, match_categories  # noqa: E402
from ffs.schemas import validate  # noqa: E402

# Suggestions for descriptions no rule matches. First match wins; anything else is suggested as `other`.
# These are guesses to be confirmed by a person, not decisions.
SUGGESTIONS = [
    (r"\b(books?|notebook|notepad|newspaper|envelope|bible)\b", "paper_document"),   # printed or written paper
    (r"\b(vase|beer can)\b", "bottle_glass"),                                          # drinking vessels, glass or metal
    (r"\b(pc screen|camera)\b", "phone_electronics"),
]


def suggest(description: str) -> str:
    for pattern, category in SUGGESTIONS:
        if re.search(pattern, description, flags=re.IGNORECASE):
            return category
    return config.OTHER


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--meta", default=str(config.META_DIR))
    ap.add_argument("--out", default=str(config.RESULTS_DIR / "phase2"))
    args = ap.parse_args()
    meta, out = Path(args.meta), Path(args.out)

    exhibits = pd.read_csv(meta / "exhibits.csv")
    rules = pd.read_csv(meta / "category_rules.csv")
    path = meta / "exhibit_categories.csv"
    existing = pd.read_csv(path) if path.exists() else None

    cats = build_exhibit_categories(exhibits, rules, suggest, existing)
    cats = cats.sort_values(["scene_id", "marker_no"], kind="stable").reset_index(drop=True)
    validate(cats, "exhibit_categories")
    cats.to_csv(path, index=False)

    no_rule = [not match_categories(d, rules) for d in exhibits.description]     # per exhibit, so repeated numbers stay apart
    keys = ["scene_id", "marker_no"]
    unmatched = exhibits[no_rule].merge(cats[cats.source != "rule"], on=keys, how="left")
    out.mkdir(parents=True, exist_ok=True)
    unmatched[["scene_id", "marker_no", "description", "category", "source", "confirmed"]].to_csv(
        out / "unmatched.csv", index=False)

    known = set() if existing is None else set(zip(existing.scene_id, existing.marker_no))
    new = unmatched[[(s, n) not in known for s, n in zip(unmatched.scene_id, unmatched.marker_no)]]
    print(f"{len(exhibits)} exhibits, {exhibits.scene_id.nunique()} scenes; {len(unmatched)} match no rule, "
          f"{len(new)} of them new:")
    if len(new):
        print(new[["scene_id", "marker_no", "description", "category"]].to_string(index=False))

    usable = cats[(cats.source == "rule") | (cats.confirmed == 1)]
    rows = []
    for cat in list(config.CANDIDATE_CATEGORIES) + [config.OTHER]:
        g = usable[usable.category == cat]               # one row per marker number, so a repeated number counts once
        rows.append({"category": cat, "exhibits": len(g), "scenes": g.scene_id.nunique()})
    print("\nExhibit-level counts (rule rows and confirmed rows):")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
