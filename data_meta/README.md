# data_meta

Small, committed CSV files that define the dataset and labels (see `SPEC.md` section 2).
Files appear here as the phases produce them:

- Phase 2: `exhibits.csv`, `category_rules.csv`, `exhibit_categories.csv`, `scene_split.csv`, `photos.csv`
- Phase 3: `markers.csv`, `silver.csv`
- Phase 4: `gold.csv`

Raw photos and downloaded spreadsheets live in `data/raw/` and are not committed.
Every file must pass `ffs.schemas.validate` before it is written.

Labels derived from FFS carry the dataset license: CC BY-NC-SA 4.0.
