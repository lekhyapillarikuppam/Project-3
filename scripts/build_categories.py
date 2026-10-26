"""Apply category_rules.csv to exhibits.csv (SPEC 4.6, R-P-2 and R-P-3).

  python scripts/build_categories.py

Writes:
  results/phase2/unmatched.csv       descriptions no rule matches, with a suggested category
                                     (source = llm, confirmed = 0): YOU confirm or correct these
  data_meta/exhibit_categories.csv   rule rows (confirmed 1) + the unconfirmed suggestions
Prints exhibit-level counts (exhibits and scenes per category) before any model is run.
Unconfirmed llm rows count as `other` everywhere downstream (silver ignores them).
"""

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from ffs import config  # noqa: E402
from ffs.dataprep import match_category  # noqa: E402
from ffs.schemas import validate  # noqa: E402

# Suggestions for descriptions no rule matches. First match wins; anything else is suggested as `other`.
# These are guesses to be confirmed by a person, not decisions.
SUGGESTIONS = [
    (r"\bbook\b", "paper_document"),          # printed paper
    (r"\bvase\b", "bottle_glass"),            # glass or ceramic object
    (r"\b(jar|cans?)\b", "bottle_glass"),
    (r"\bsyringe\b", "other"),
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
    exhibits["rule_category"] = [match_category(d, rules) for d in exhibits["description"]]

    matched = exhibits[exhibits.rule_category.notna()]
    unmatched = exhibits[exhibits.rule_category.isna()].copy()
    unmatched["category"] = unmatched["description"].map(suggest)
    unmatched["source"], unmatched["confirmed"] = "llm", 0

    cats = pd.concat([
        pd.DataFrame({"scene_id": matched.scene_id, "marker_no": matched.marker_no,
                      "category": matched.rule_category, "source": "rule", "confirmed": 1}),
        unmatched[["scene_id", "marker_no", "category", "source", "confirmed"]]], ignore_index=True)
    cats = cats.sort_values(["scene_id", "marker_no"], kind="stable").reset_index(drop=True)
    validate(cats, "exhibit_categories")
    cats.to_csv(meta / "exhibit_categories.csv", index=False)
    out.mkdir(parents=True, exist_ok=True)
    unmatched[["scene_id", "marker_no", "description", "category", "source", "confirmed"]].to_csv(
        out / "unmatched.csv", index=False)

    print(f"{len(exhibits)} exhibits: {len(matched)} matched by a rule, {len(unmatched)} unmatched")
    rows = []
    for cat in list(config.CANDIDATE_CATEGORIES) + [config.OTHER]:
        by_rule = matched[matched.rule_category == cat]
        with_sugg = cats[cats.category == cat]
        rows.append({"category": cat, "rule: exhibits": len(by_rule), "rule: scenes": by_rule.scene_id.nunique(),
                     "rule+suggested: exhibits": len(with_sugg), "rule+suggested: scenes": with_sugg.scene_id.nunique()})
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
