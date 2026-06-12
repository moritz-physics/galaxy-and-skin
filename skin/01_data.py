"""Download a balanced HAM10000 subset by streaming from Hugging Face.

Streams marmal88/skin_cancer (a HAM10000 mirror) so the full ~2.6GB dataset
is never stored locally. Keeps up to TRAIN_CAP / VAL_CAP images per class,
resized so the shorter side is 256px, saved as ImageFolder-style directories:

    skin/data/train/<class>/<image_id>.jpg
    skin/data/val/<class>/<image_id>.jpg

HAM10000 is heavily imbalanced (67% melanocytic nevi), so capping per class
also rebalances the training set. Rare classes (dermatofibroma: 115,
vascular: 142) contribute everything they have.
"""

import argparse
import csv
from pathlib import Path

from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

DATA_DIR = Path(__file__).resolve().parent / "data"
SHORT_SIDE = 256
JPEG_QUALITY = 87


def resize_short_side(img: Image.Image, target: int) -> Image.Image:
    w, h = img.size
    scale = target / min(w, h)
    if scale >= 1.0:
        return img
    return img.resize((round(w * scale), round(h * scale)), Image.LANCZOS)


def harvest(split: str, out_name: str, cap: int) -> list[dict]:
    ds = load_dataset("marmal88/skin_cancer", split=split, streaming=True)
    counts: dict[str, int] = {}
    rows = []
    pbar = tqdm(ds, desc=f"{split} -> {out_name}", unit="img")
    for ex in pbar:
        dx = ex["dx"].lower()
        if counts.get(dx, 0) >= cap:
            continue
        counts[dx] = counts.get(dx, 0) + 1
        out_dir = DATA_DIR / out_name / dx
        out_dir.mkdir(parents=True, exist_ok=True)
        img = resize_short_side(ex["image"].convert("RGB"), SHORT_SIDE)
        img.save(out_dir / f"{ex['image_id']}.jpg", quality=JPEG_QUALITY)
        rows.append(
            {
                "image_id": ex["image_id"],
                "lesion_id": ex["lesion_id"],
                "split": out_name,
                "dx": dx,
                "age": ex["age"],
                "sex": ex["sex"],
                "localization": ex["localization"],
            }
        )
        pbar.set_postfix(kept=len(rows))
    print(f"{out_name}: kept {len(rows)} images, per class: {counts}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-cap", type=int, default=500, help="max train images per class")
    parser.add_argument("--val-cap", type=int, default=100, help="max val images per class")
    args = parser.parse_args()

    rows = harvest("train", "train", args.train_cap)
    rows += harvest("validation", "val", args.val_cap)

    meta_path = DATA_DIR / "metadata.csv"
    with open(meta_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {meta_path}")

    total_mb = sum(p.stat().st_size for p in DATA_DIR.rglob("*.jpg")) / 1e6
    print(f"Total size on disk: {total_mb:.0f} MB")


if __name__ == "__main__":
    main()
