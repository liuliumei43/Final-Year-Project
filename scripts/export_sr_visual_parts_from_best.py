import argparse
import csv
from pathlib import Path

from PIL import Image, ImageDraw


def load_rgb(path):
    return Image.open(path).convert("RGB")


def load_best_row(csv_path, rank):
    with Path(csv_path).open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"No rows found in {csv_path}")
    idx = rank - 1
    if idx < 0 or idx >= len(rows):
        raise IndexError(f"rank={rank} is out of range; csv has {len(rows)} rows")
    return rows[idx]


def find_lq(gt_path, lq_dir):
    gt_path = Path(gt_path)
    lq_dir = Path(lq_dir)
    stem = gt_path.stem
    candidates = [
        lq_dir / f"{stem}x2.png",
        lq_dir / f"{stem}_LRBI_x2.png",
        lq_dir / f"{stem}.png",
    ]
    for path in candidates:
        if path.exists():
            return path
    matches = sorted(lq_dir.glob(f"{stem}*.png"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Cannot find LR image for {gt_path.name} in {lq_dir}")


def crop_and_resize(img, crop, out_size):
    x, y, w, h = crop
    patch = img.crop((x, y, x + w, y + h))
    return patch.resize((out_size, out_size), Image.Resampling.BICUBIC)


def make_overview(gt, crop, out_width):
    scale = out_width / gt.width
    out_height = max(1, int(gt.height * scale))
    overview = gt.resize((out_width, out_height), Image.Resampling.BICUBIC)
    draw = ImageDraw.Draw(overview)
    x, y, w, h = crop
    box = (
        int(x * scale),
        int(y * scale),
        int((x + w) * scale),
        int((y + h) * scale),
    )
    draw.rectangle(box, outline=(230, 0, 0), width=3)
    return overview


def main():
    parser = argparse.ArgumentParser(
        description="Export five separate images for LaTeX SR visual comparison: overview, bicubic, tiny, ssgr, gt."
    )
    parser.add_argument("--csv", required=True, help="best_patches.csv from find_best_sr_visual_examples.py.")
    parser.add_argument("--rank", type=int, default=1, help="Rank in CSV, starting from 1.")
    parser.add_argument("--lq-dir", required=True, help="LR_bicubic/X2 directory.")
    parser.add_argument("--out-dir", required=True, help="Output folder for separated visual parts.")
    parser.add_argument("--crop-size", type=int, default=220, help="Output size of each crop image.")
    parser.add_argument("--overview-width", type=int, default=520, help="Output width of overview image.")
    args = parser.parse_args()

    row = load_best_row(args.csv, args.rank)
    tiny_path = Path(row["tiny_path"])
    ssgr_path = Path(row["ssgr_path"])
    gt_path = Path(row["gt_path"])
    lq_path = find_lq(gt_path, args.lq_dir)
    crop = (int(row["x"]), int(row["y"]), int(row["size"]), int(row["size"]))

    tiny = load_rgb(tiny_path)
    ssgr = load_rgb(ssgr_path)
    gt = load_rgb(gt_path)
    lq = load_rgb(lq_path)

    if tiny.size != gt.size:
        tiny = tiny.resize(gt.size, Image.Resampling.BICUBIC)
    if ssgr.size != gt.size:
        ssgr = ssgr.resize(gt.size, Image.Resampling.BICUBIC)
    bicubic = lq.resize(gt.size, Image.Resampling.BICUBIC)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    make_overview(gt, crop, args.overview_width).save(out_dir / "overview.png")
    crop_and_resize(bicubic, crop, args.crop_size).save(out_dir / "bicubic.png")
    crop_and_resize(tiny, crop, args.crop_size).save(out_dir / "mair_tiny.png")
    crop_and_resize(ssgr, crop, args.crop_size).save(out_dir / "ssgr.png")
    crop_and_resize(gt, crop, args.crop_size).save(out_dir / "gt.png")

    meta = out_dir / "meta.txt"
    meta.write_text(
        "\n".join([
            f"name: {row['name']}",
            f"rank: {args.rank}",
            f"crop: {crop[0]},{crop[1]},{crop[2]},{crop[3]}",
            f"tiny_path: {tiny_path}",
            f"ssgr_path: {ssgr_path}",
            f"gt_path: {gt_path}",
            f"lq_path: {lq_path}",
        ]),
        encoding="utf-8",
    )

    print(f"Exported visual parts to: {out_dir}")
    print(f"Crop x,y,w,h = {crop[0]},{crop[1]},{crop[2]},{crop[3]}")
    print("Files: overview.png, bicubic.png, mair_tiny.png, ssgr.png, gt.png, meta.txt")


if __name__ == "__main__":
    main()
