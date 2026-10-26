"""Stage B on a GPU (Kaggle): CLIP whole-photo scores and OWLv2 raw boxes (SPEC section 5).

  python stage_b_run.py --root <folder with manifest.csv and photos> --name dev --out <output folder>

Reads  <root>/manifest.csv   columns: scene_id, photo_id, file, orig_w, orig_h
Writes <out>/clip_scores_<name>.csv   scene_id, photo_id, category, prompt, score
       <out>/owlv2_raw_<name>.csv     scene_id, photo_id, category, prompt, score, x, y, w, h, tile
       <out>/run_info_<name>.json     models, settings, versions

This file never reads silver, gold or exhibit lists. Boxes are in ORIGINAL photo pixels
(the photos may have been shrunk for upload; the manifest holds the original size).
All boxes above a low cut-off are kept so thresholds can be tuned later without re-running.
The prompts below are tuned on dev only and frozen with the protocol tag (SPEC 6).
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

# ---- Settings (frozen with the protocol tag) -----------------------------------------------
CLIP_MODEL = "openai/clip-vit-large-patch14"
OWL_MODEL = "google/owlv2-base-patch16-ensemble"

# CLIP sees the whole photo. Never use the word "blood": a red-brown stain is only a red-brown stain.
CLIP_PROMPTS = {
    "ammunition": ["a photo of a bullet cartridge case", "a photo of spent shell casings on the floor",
                   "a photo of ammunition"],
    "blade": ["a photo of a knife", "a photo of a kitchen knife on the floor", "a photo of scissors or a blade"],
    "bottle_glass": ["a photo of a bottle", "a photo of a drinking glass", "a photo of broken glass"],
    "paper_document": ["a photo of sheets of paper", "a photo of a notebook or a book",
                       "a photo of a letter or a document"],
    "red_brown_stain": ["a photo of a red-brown stain", "a photo of a reddish stain on the floor",
                        "a photo of a reddish trace on a surface"],
}
# OWLv2 gets short noun phrases.
OWL_QUERIES = {
    "ammunition": ["a cartridge case", "a bullet", "a shotgun shell"],
    "blade": ["a knife", "scissors", "an axe"],
    "bottle_glass": ["a bottle", "a drinking glass", "broken glass"],
    "paper_document": ["a sheet of paper", "a notebook", "a book"],
    "red_brown_stain": ["a red-brown stain", "a reddish stain"],
}
TILE_GRID = 2          # 2 x 2 tiles plus the full photo (tile 0)
TILE_OVERLAP = 0.25    # fraction of a tile that overlaps its neighbour
RAW_CUTOFF = 0.05      # keep boxes with score at least this
TOP_K = 20             # at most this many boxes per photo, query and tile


# ---- Pure helpers (tested in tests/test_stage_b.py) -----------------------------------------
def make_tiles(w: int, h: int, grid: int = TILE_GRID, overlap: float = TILE_OVERLAP):
    """[(tile_id, x0, y0, x1, y1)]. Tile 0 is the full photo; 1..grid*grid are overlapping tiles, row by row."""
    tiles = [(0, 0, 0, w, h)]
    tw, th = w / (grid - (grid - 1) * overlap), h / (grid - (grid - 1) * overlap)
    sx = (w - tw) / (grid - 1) if grid > 1 else 0
    sy = (h - th) / (grid - 1) if grid > 1 else 0
    tid = 1
    for r in range(grid):
        for c in range(grid):
            x0, y0 = int(round(c * sx)), int(round(r * sy))
            tiles.append((tid, x0, y0, min(w, int(round(x0 + tw))), min(h, int(round(y0 + th)))))
            tid += 1
    return tiles


def decode_boxes(scores: np.ndarray, boxes: np.ndarray, side: float, off_x: float, off_y: float,
                 max_x: float, max_y: float, scale: float = 1.0, cutoff: float = RAW_CUTOFF,
                 top_k: int = TOP_K):
    """OWLv2 output for one tile -> {query_index: [(score, x, y, w, h)]} in original photo pixels.

    scores: [patches, queries] in 0..1. boxes: [patches, 4] as (cx, cy, w, h) normalised to the PADDED
    SQUARE tile of size `side`. off_x/off_y: tile position in the (shrunk) photo. max_x/max_y: photo size
    in the same (shrunk) pixels. scale: shrunk -> original pixels.
    """
    out = {}
    for q in range(scores.shape[1]):
        s = scores[:, q]
        order = np.argsort(-s)[:top_k]
        rows = []
        for i in order:
            if s[i] < cutoff:
                break
            cx, cy, bw, bh = boxes[i] * side
            x0, y0 = max(0.0, cx - bw / 2 + off_x), max(0.0, cy - bh / 2 + off_y)
            x1, y1 = min(max_x, cx + bw / 2 + off_x), min(max_y, cy + bh / 2 + off_y)
            if x1 <= x0 or y1 <= y0:
                continue                      # the box lies in the padding
            rows.append((float(s[i]), x0 * scale, y0 * scale, (x1 - x0) * scale, (y1 - y0) * scale))
        out[q] = rows
    return out


def flat_queries(queries: dict):
    """[(category, prompt)] in a fixed order."""
    return [(c, p) for c, ps in queries.items() for p in ps]


# ---- Heavy part (needs torch + transformers + a GPU) ----------------------------------------
def load_image(path):
    from PIL import Image, ImageOps

    return ImageOps.exif_transpose(Image.open(path)).convert("RGB")


def pad_square(img, fill=(124, 116, 104)):
    from PIL import Image

    w, h = img.size
    s = max(w, h)
    canvas = Image.new("RGB", (s, s), fill)
    canvas.paste(img, ((s - w) // 2, (s - h) // 2))
    return canvas


def main():
    import torch
    import transformers
    from transformers import CLIPModel, CLIPProcessor, Owlv2ForObjectDetection, Owlv2Processor

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--name", required=True, help="dev or test")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None, help="only the first N photos (smoke test)")
    args = ap.parse_args()
    root, out = Path(args.root), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", dev, "| transformers", transformers.__version__)

    rows = list(csv.DictReader(open(root / "manifest.csv", encoding="utf-8")))[: args.limit]
    clip_q, owl_q = flat_queries(CLIP_PROMPTS), flat_queries(OWL_QUERIES)

    clip = CLIPModel.from_pretrained(CLIP_MODEL).to(dev).eval()
    clip_proc = CLIPProcessor.from_pretrained(CLIP_MODEL)
    owl = Owlv2ForObjectDetection.from_pretrained(OWL_MODEL).to(dev).eval()
    owl_proc = Owlv2Processor.from_pretrained(OWL_MODEL)

    f_clip = open(out / f"clip_scores_{args.name}.csv", "w", newline="", encoding="utf-8")
    f_owl = open(out / f"owlv2_raw_{args.name}.csv", "w", newline="", encoding="utf-8")
    w_clip, w_owl = csv.writer(f_clip), csv.writer(f_owl)
    w_clip.writerow(["scene_id", "photo_id", "category", "prompt", "score"])
    w_owl.writerow(["scene_id", "photo_id", "category", "prompt", "score", "x", "y", "w", "h", "tile"])

    t0 = time.time()
    for n, r in enumerate(rows, 1):
        img = load_image(root / r["file"])
        W, H = img.size
        scale = int(r["orig_w"]) / W                           # shrunk -> original pixels

        with torch.no_grad():                                  # CLIP: whole photo, padded to a square
            inp = clip_proc(text=[p for _, p in clip_q], images=pad_square(img), return_tensors="pt",
                            padding=True).to(dev)
            logits = clip(**inp).logits_per_image[0].float().cpu().numpy()
        for (cat, prompt), sc in zip(clip_q, logits):
            w_clip.writerow([r["scene_id"], r["photo_id"], cat, prompt, round(float(sc), 4)])

        for tid, x0, y0, x1, y1 in make_tiles(W, H):           # OWLv2: full photo + 2x2 tiles
            tile = img.crop((x0, y0, x1, y1))
            with torch.no_grad():
                inp = owl_proc(text=[[p for _, p in owl_q]], images=tile, return_tensors="pt").to(dev)
                o = owl(**inp)
            scores = o.logits[0].sigmoid().float().cpu().numpy()
            boxes = o.pred_boxes[0].float().cpu().numpy()
            dec = decode_boxes(scores, boxes, side=max(tile.size), off_x=x0, off_y=y0,
                               max_x=W, max_y=H, scale=scale)
            for qi, found in dec.items():
                cat, prompt = owl_q[qi]
                for s, x, y, bw, bh in found:
                    w_owl.writerow([r["scene_id"], r["photo_id"], cat, prompt, round(s, 4),
                                    round(x, 1), round(y, 1), round(bw, 1), round(bh, 1), tid])
        if n % 10 == 0 or n == len(rows):
            f_clip.flush(); f_owl.flush()
            print(f"{n}/{len(rows)} photos, {time.time() - t0:.0f}s", flush=True)

    f_clip.close(); f_owl.close()
    info = {"name": args.name, "n_photos": len(rows), "clip_model": CLIP_MODEL, "owl_model": OWL_MODEL,
            "clip_prompts": CLIP_PROMPTS, "owl_queries": OWL_QUERIES, "tile_grid": TILE_GRID,
            "tile_overlap": TILE_OVERLAP, "raw_cutoff": RAW_CUTOFF, "top_k": TOP_K,
            "torch": torch.__version__, "transformers": transformers.__version__, "device": dev,
            "seconds": round(time.time() - t0)}
    (out / f"run_info_{args.name}.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print("Done. Download clip_scores, owlv2_raw and run_info from the Output tab.")


if __name__ == "__main__":
    main()
