import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from make_sr_visual_comparison_batch import collect_pairs


def load_rgb(path):
    return Image.open(path).convert("RGB")


def rgb_to_y(img):
    arr = np.asarray(img.convert("RGB"), dtype=np.float32)
    return 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]


def laplacian_abs(gray):
    arr = gray.astype(np.float32)
    out = np.zeros_like(arr, dtype=np.float32)
    out[1:-1, 1:-1] = np.abs(
        arr[1:-1, 1:-1] * 4.0
        - arr[:-2, 1:-1]
        - arr[2:, 1:-1]
        - arr[1:-1, :-2]
        - arr[1:-1, 2:]
    )
    return out


def psnr_from_mse(mse):
    if mse <= 1e-12:
        return 99.0
    return 10.0 * np.log10((255.0 * 255.0) / mse)


def score_patch(tiny_y, ssgr_y, gt_y, edge, x, y, size, edge_weight):
    tiny_patch = tiny_y[y:y + size, x:x + size]
    ssgr_patch = ssgr_y[y:y + size, x:x + size]
    gt_patch = gt_y[y:y + size, x:x + size]
    edge_patch = edge[y:y + size, x:x + size]

    err_tiny = (tiny_patch - gt_patch) ** 2
    err_ssgr = (ssgr_patch - gt_patch) ** 2
    gain_map = err_tiny - err_ssgr

    edge_mean = float(edge_patch.mean())
    edge_norm = edge_patch / (edge_patch.mean() + 1e-6)
    weighted_gain = gain_map * (1.0 + edge_weight * edge_norm)
    score = float(weighted_gain.mean())
    mse_tiny = float(err_tiny.mean())
    mse_ssgr = float(err_ssgr.mean())
    psnr_tiny = psnr_from_mse(mse_tiny)
    psnr_ssgr = psnr_from_mse(mse_ssgr)
    return {
        "score": score,
        "edge": edge_mean,
        "mse_tiny": mse_tiny,
        "mse_ssgr": mse_ssgr,
        "psnr_tiny": psnr_tiny,
        "psnr_ssgr": psnr_ssgr,
        "psnr_gain": psnr_ssgr - psnr_tiny,
        "x": x,
        "y": y,
        "size": size,
    }


def find_best_patches(tiny, ssgr, gt, patch_size, stride, edge_weight, min_edge, per_image_top):
    if tiny.size != gt.size:
        tiny = tiny.resize(gt.size, Image.Resampling.BICUBIC)
    if ssgr.size != gt.size:
        ssgr = ssgr.resize(gt.size, Image.Resampling.BICUBIC)

    tiny_y = rgb_to_y(tiny)
    ssgr_y = rgb_to_y(ssgr)
    gt_y = rgb_to_y(gt)
    edge = laplacian_abs(gt_y)

    h, w = gt_y.shape
    size = min(patch_size, h, w)
    scored = []
    for y in range(0, h - size + 1, stride):
        for x in range(0, w - size + 1, stride):
            item = score_patch(tiny_y, ssgr_y, gt_y, edge, x, y, size, edge_weight)
            if item["edge"] >= min_edge and item["psnr_gain"] > 0:
                scored.append(item)
    scored.sort(key=lambda v: (v["score"], v["psnr_gain"]), reverse=True)
    return scored[:per_image_top]


def crop_resize(img, x, y, size, zoom):
    crop = img.crop((x, y, x + size, y + size))
    return crop.resize((size * zoom, size * zoom), Image.Resampling.BICUBIC)


def gain_heatmap(tiny, ssgr, gt, x, y, size, zoom):
    tiny_y = rgb_to_y(tiny)[y:y + size, x:x + size]
    ssgr_y = rgb_to_y(ssgr)[y:y + size, x:x + size]
    gt_y = rgb_to_y(gt)[y:y + size, x:x + size]
    gain = (tiny_y - gt_y) ** 2 - (ssgr_y - gt_y) ** 2

    pos = np.clip(gain, 0, None)
    neg = np.clip(-gain, 0, None)
    scale = np.percentile(np.abs(gain), 98) + 1e-6
    pos = np.clip(pos / scale, 0, 1)
    neg = np.clip(neg / scale, 0, 1)

    heat = np.ones((size, size, 3), dtype=np.float32) * 255.0
    # Green means SSGR has lower error. Red means Tiny has lower error.
    heat[..., 0] = 255.0 * (1.0 - 0.75 * pos)
    heat[..., 1] = 255.0 * (1.0 - 0.75 * neg)
    heat[..., 2] = 255.0 * (1.0 - 0.75 * pos - 0.75 * neg)
    heat = np.clip(heat, 0, 255).astype(np.uint8)
    return Image.fromarray(heat).resize((size * zoom, size * zoom), Image.Resampling.NEAREST)


def draw_label(draw, xy, text, font):
    x, y = xy
    bbox = draw.textbbox((x, y), text, font=font)
    draw.rectangle((bbox[0] - 7, bbox[1] - 4, bbox[2] + 7, bbox[3] + 4), fill=(255, 255, 255), outline=(60, 60, 60))
    draw.text((x, y), text, fill=(20, 20, 20), font=font)


def save_comparison(tiny_path, ssgr_path, gt_path, item, out_path, zoom):
    tiny = load_rgb(tiny_path)
    ssgr = load_rgb(ssgr_path)
    gt = load_rgb(gt_path)
    if tiny.size != gt.size:
        tiny = tiny.resize(gt.size, Image.Resampling.BICUBIC)
    if ssgr.size != gt.size:
        ssgr = ssgr.resize(gt.size, Image.Resampling.BICUBIC)

    x, y, size = item["x"], item["y"], item["size"]
    panels = [
        crop_resize(tiny, x, y, size, zoom),
        crop_resize(ssgr, x, y, size, zoom),
        crop_resize(gt, x, y, size, zoom),
        gain_heatmap(tiny, ssgr, gt, x, y, size, zoom),
    ]
    labels = [
        f"MaIR-Tiny ({item['psnr_tiny']:.2f} dB)",
        f"SSGR ({item['psnr_ssgr']:.2f} dB)",
        "GT",
        "Gain Map",
    ]

    margin, gap, label_h = 18, 14, 34
    width = sum(p.width for p in panels) + gap * (len(panels) - 1) + margin * 2
    height = panels[0].height + margin * 2 + label_h
    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    px = margin
    for panel, label in zip(panels, labels):
        draw_label(draw, (px + 8, margin + 4), label, font)
        py = margin + label_h
        canvas.paste(panel, (px, py))
        draw.rectangle((px, py, px + panel.width - 1, py + panel.height - 1), outline=(20, 20, 20), width=2)
        px += panel.width + gap

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def main():
    parser = argparse.ArgumentParser(
        description="Find the best local SR visual examples where SSGR improves over Tiny."
    )
    parser.add_argument("--tiny-dir", required=True)
    parser.add_argument("--ssgr-dir", required=True)
    parser.add_argument("--gt-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--tiny-suffix", default="tiny")
    parser.add_argument("--ssgr-suffix", default="ssgr")
    parser.add_argument("--patch-size", type=int, default=96)
    parser.add_argument("--stride", type=int, default=24)
    parser.add_argument("--edge-weight", type=float, default=0.5)
    parser.add_argument("--min-edge", type=float, default=1.0)
    parser.add_argument("--per-image-top", type=int, default=2)
    parser.add_argument("--topk", type=int, default=2)
    parser.add_argument("--zoom", type=int, default=3)
    args = parser.parse_args()

    tiny_dir = Path(args.tiny_dir)
    ssgr_dir = Path(args.ssgr_dir)
    gt_dir = Path(args.gt_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs = collect_pairs(tiny_dir, ssgr_dir, gt_dir, args.tiny_suffix, args.ssgr_suffix)
    if not pairs:
        print("No matched image triplets found. Check file names and directories.")
        return

    all_items = []
    for tiny_path, ssgr_path, gt_path, name in pairs:
        tiny = load_rgb(tiny_path)
        ssgr = load_rgb(ssgr_path)
        gt = load_rgb(gt_path)
        patches = find_best_patches(
            tiny, ssgr, gt,
            args.patch_size,
            args.stride,
            args.edge_weight,
            args.min_edge,
            args.per_image_top,
        )
        for patch in patches:
            patch.update({
                "name": name,
                "tiny_path": str(tiny_path),
                "ssgr_path": str(ssgr_path),
                "gt_path": str(gt_path),
            })
            all_items.append(patch)

    all_items.sort(key=lambda v: (v["score"], v["psnr_gain"]), reverse=True)
    best = all_items[: args.topk]
    if not best:
        print("No positive-improvement patches found. Try lowering --min-edge or --patch-size.")
        return

    csv_path = out_dir / "best_patches.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "rank", "name", "x", "y", "size", "score", "edge",
                "psnr_tiny", "psnr_ssgr", "psnr_gain",
                "tiny_path", "ssgr_path", "gt_path",
            ],
        )
        writer.writeheader()
        for rank, item in enumerate(best, 1):
            row = {k: item[k] for k in writer.fieldnames if k != "rank"}
            row["rank"] = rank
            writer.writerow(row)

    for rank, item in enumerate(best, 1):
        out_path = out_dir / f"top{rank:02d}_{item['name']}_x{item['x']}_y{item['y']}.png"
        save_comparison(
            Path(item["tiny_path"]),
            Path(item["ssgr_path"]),
            Path(item["gt_path"]),
            item,
            out_path,
            args.zoom,
        )
        print(
            f"Top {rank}: {item['name']} crop=({item['x']},{item['y']},{item['size']},{item['size']}) "
            f"patch PSNR {item['psnr_tiny']:.3f}->{item['psnr_ssgr']:.3f} "
            f"(+{item['psnr_gain']:.3f} dB), saved {out_path}"
        )
    print(f"CSV saved: {csv_path}")


if __name__ == "__main__":
    main()
