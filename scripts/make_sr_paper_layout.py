import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def load_rgb(path):
    return Image.open(path).convert("RGB")


def parse_crop(text):
    parts = [int(v.strip()) for v in text.split(",")]
    if len(parts) != 4:
        raise ValueError("--crop must be x,y,w,h")
    return tuple(parts)


def rgb_to_y(img):
    arr = np.asarray(img.convert("RGB"), dtype=np.float32)
    return 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]


def laplacian_abs(gray):
    out = np.zeros_like(gray, dtype=np.float32)
    out[1:-1, 1:-1] = np.abs(
        gray[1:-1, 1:-1] * 4.0
        - gray[:-2, 1:-1]
        - gray[2:, 1:-1]
        - gray[1:-1, :-2]
        - gray[1:-1, 2:]
    )
    return out


def auto_crop(tiny, ssgr, gt, patch_size, stride, edge_weight):
    tiny_y = rgb_to_y(tiny)
    ssgr_y = rgb_to_y(ssgr)
    gt_y = rgb_to_y(gt)
    edge = laplacian_abs(gt_y)

    h, w = gt_y.shape
    size = min(patch_size, h, w)
    best = (-1e18, 0, 0)
    for y in range(0, h - size + 1, stride):
        for x in range(0, w - size + 1, stride):
            err_tiny = (tiny_y[y:y + size, x:x + size] - gt_y[y:y + size, x:x + size]) ** 2
            err_ssgr = (ssgr_y[y:y + size, x:x + size] - gt_y[y:y + size, x:x + size]) ** 2
            gain = err_tiny - err_ssgr
            local_edge = edge[y:y + size, x:x + size]
            score = float(gain.mean() + edge_weight * local_edge.mean())
            if score > best[0]:
                best = (score, x, y)
    _, x, y = best
    return x, y, size, size


def crop_and_zoom(img, crop, size):
    x, y, w, h = crop
    patch = img.crop((x, y, x + w, y + h))
    return patch.resize((size, size), Image.Resampling.BICUBIC)


def fit_image(img, target_w, target_h):
    scale = min(target_w / img.width, target_h / img.height)
    new_w = max(1, int(img.width * scale))
    new_h = max(1, int(img.height * scale))
    return img.resize((new_w, new_h), Image.Resampling.BICUBIC), scale


def draw_label_center(draw, box, text, font):
    x0, y0, x1, y1 = box
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    draw.text((x0 + (x1 - x0 - tw) / 2, y0 + (y1 - y0 - th) / 2), text, fill=(0, 0, 0), font=font)


def main():
    parser = argparse.ArgumentParser(description="Create paper-style SR visual layout.")
    parser.add_argument("--tiny", required=True, help="MaIR-Tiny SR output.")
    parser.add_argument("--ssgr", required=True, help="MaIR-Tiny+SSGR SR output.")
    parser.add_argument("--gt", required=True, help="GT HR image.")
    parser.add_argument("--lq", required=True, help="LR bicubic input image. It will be upsampled to HR size.")
    parser.add_argument("--out", required=True, help="Output png path.")
    parser.add_argument("--crop", default=None, help="Crop in HR coordinates: x,y,w,h. If omitted, choose automatically.")
    parser.add_argument("--patch-size", type=int, default=64)
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--edge-weight", type=float, default=0.05)
    parser.add_argument("--crop-display-size", type=int, default=150)
    parser.add_argument("--overview-width", type=int, default=420)
    parser.add_argument("--overview-height", type=int, default=330)
    args = parser.parse_args()

    tiny = load_rgb(args.tiny)
    ssgr = load_rgb(args.ssgr)
    gt = load_rgb(args.gt)
    lq = load_rgb(args.lq)

    if tiny.size != gt.size:
        tiny = tiny.resize(gt.size, Image.Resampling.BICUBIC)
    if ssgr.size != gt.size:
        ssgr = ssgr.resize(gt.size, Image.Resampling.BICUBIC)
    bicubic = lq.resize(gt.size, Image.Resampling.BICUBIC)

    crop = parse_crop(args.crop) if args.crop else auto_crop(tiny, ssgr, gt, args.patch_size, args.stride, args.edge_weight)
    print(f"Using crop x,y,w,h = {crop[0]},{crop[1]},{crop[2]},{crop[3]}")

    margin = 24
    gap = 18
    label_h = 28
    patch_size = args.crop_display_size
    right_cols = 2
    right_rows = 2
    right_w = right_cols * patch_size + (right_cols - 1) * gap
    right_h = right_rows * (patch_size + label_h) + (right_rows - 1) * 10
    canvas_w = margin * 2 + args.overview_width + gap + right_w
    canvas_h = margin * 2 + max(args.overview_height + label_h, right_h)

    canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    # Top rule, close to common SR paper visual style.
    draw.line((margin, 8, canvas_w - margin, 8), fill=(0, 0, 0), width=2)

    overview, scale = fit_image(gt, args.overview_width, args.overview_height)
    ov_x = margin + (args.overview_width - overview.width) // 2
    ov_y = margin
    canvas.paste(overview, (ov_x, ov_y))
    draw.rectangle((ov_x, ov_y, ov_x + overview.width - 1, ov_y + overview.height - 1), outline=(40, 40, 40), width=1)

    x, y, w, h = crop
    rx0 = ov_x + int(x * scale)
    ry0 = ov_y + int(y * scale)
    rx1 = ov_x + int((x + w) * scale)
    ry1 = ov_y + int((y + h) * scale)
    draw.rectangle((rx0, ry0, rx1, ry1), outline=(230, 0, 0), width=3)
    draw_label_center(draw, (ov_x, ov_y + overview.height, ov_x + overview.width, ov_y + overview.height + label_h), "HR Image", font)

    labels = ["Bicubic", "MaIR-Tiny", "MaIR-Tiny+SSGR", "GT"]
    imgs = [bicubic, tiny, ssgr, gt]
    start_x = margin + args.overview_width + gap
    start_y = margin
    for idx, (label, img) in enumerate(zip(labels, imgs)):
        row = idx // right_cols
        col = idx % right_cols
        px = start_x + col * (patch_size + gap)
        py = start_y + row * (patch_size + label_h + 10)
        patch = crop_and_zoom(img, crop, patch_size)
        canvas.paste(patch, (px, py))
        draw.rectangle((px, py, px + patch_size - 1, py + patch_size - 1), outline=(30, 30, 30), width=1)
        draw_label_center(draw, (px, py + patch_size, px + patch_size, py + patch_size + label_h), label, font)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
