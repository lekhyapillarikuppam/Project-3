# Phase 3b: AI-labelled, human-checked gold (replaces part A of PHASE3_STEPS.md)

## 1. Claude Code labels the 545 test photos (no work from you)
Use the two prompts in the chat. It looks at the photos (never at silver, markers or exhibit lists) and writes
`data_meta/gold_ai.csv`, then builds your two checking pages.

## 2. You check (blind: the pages show neither the AI's label nor silver)
1. `python scripts/make_gold_tool.py --review` then open `results\gold\gold_tool_review.html` in Chrome.
   Each photo lists only the categories to decide. Press the number key for each one that IS present, then Enter.
   Click or Z zooms. L opens the exhibit list if you are unsure.
2. Press S at the end of a session. It downloads `gold_review_export.csv`; move it into `results\gold\`.
3. `python scripts/make_gold_tool.py --audit 60` then open `results\gold\gold_tool_audit.html`. Label all categories
   of these 60 photos the same way. Save `gold_audit_export.csv` into `results\gold\`.
4. Progress check any time:
   `python scripts/import_gold.py --merge results\gold\gold_review_export.csv results\gold\gold_audit_export.csv --allow-partial --dry-run`
5. When both are complete, run it without the two flags: it writes `data_meta\gold.csv` and `data_meta\gold_sources.csv` and
   prints how often the AI was wrong and who you sided with. Paste that output in the chat.
Do not clear Chrome's site data until the exports are saved.
