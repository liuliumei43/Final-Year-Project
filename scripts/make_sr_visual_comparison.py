import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def load_rgb(path):
    return Image.open(path).convert("RGB")


def parse_crop(text):
    if text is None:
        return None
    parts = [int(v.strip()) for v in text.split(",")]
    if len(parts) != 4:
        raise ValueError("--crop must be x,y,w,h")
    return tuple(parts)


def laplacian_score(gray):
    arr = gray.astype(np.float32)
    center = arr[1:-1, 1:-1] * 4.0
    lap = center - arr[:-2, 1:-1] - arr[2:, 1:-1] - arr[1:-1, :-2] - arr[1:-1, 2:]
    return np.abs(lap)


def auto_crop_from_gt(gt, crop_size, stride):
    gray = np.asarray(gt.convert("L"), dtype=np.float32)
    score = laplacian_score(gray)
    h, w = score.shape
    crop = min(crop_size, w, h)
    best = (-1.0, 0, 0)
    for y in range(0, h - crop + 1, stride):
        for x in range(0, w - crop + 1, stride):
            patch = score[y:y + crop, x:x + crop]
            val = float(patch.mean())
            if val > best[0]:
                best = (val, x + 1, y + 1)
    _, x, y = best
    return x, y, crop, crop


def crop_image(img, crop):
    x, y, w, h = crop
    x = max(0, min(x, img.width - w))
    y = max(0, min(y, img.height - h))
    return img.crop((x, y, x + w, y + h))


def draw_label(draw, xy, text, font):
    x, y = xy
    pad_x, pad_y = 8, 5
    bbox = draw.textbbox((x, y), text, font=font)
    draw.rectangle(
        (bbox[0] - pad_x, bbox[1] - pad_y, bbox[2] + pad_x, bbox[3] + pad_y),
        fill=(255, 255, 255),
        outline=(45, 45, 45),
        width=1,
    )
    draw.text((x, y), text, fill=(20, 20, 20), font=font)


def make_panel(images, labels, crop, zoom, out_path, margin=18, gap=14):
    crops = [crop_image(img, crop).resize((crop[2] * zoom, crop[3] * zoom), Image.Resampling.BICUBIC) for img in images]
    panel_w = sum(c.width for c in crops) + gap * (len(crops) - 1) + margin * 2
    panel_h = max(c.height for c in crops) + margin * 2 + 34
    canvas = Image.new("RGB", (panel_w, panel_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    x = margin
    for crop_img, label in zip(crops, labels):
        y = margin + 28
        canvas.paste(crop_img, (x, y))
        draw.rectangle((x, y, x + crop_img.width - 1, y + crop_img.height - 1), outline=(20, 20, 20), width=2)
        draw_label(draw, (x + 8, margin + 4), label, font)
        x += crop_img.width + gap

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def main():
    parser = argparse.ArgumentParser(
        description="Create a thesis-ready SR visual comparison crop: Tiny vs SSGR vs GT."
    )
    parser.add_argument("--tiny", required=True, help="MaIR-Tiny SR output image.")
    parser.add_argument("--ssgr", required=True, help="MaIR-Tiny+SSGR SR output image.")
    parser.add_argument("--gt", required=True, help="Ground-truth HR image.")
    parser.add_argument("--bicubic", default=None, help="Optional bicubic output image.")
    parser.add_argument("--out", default="figures/sr_visual_compare.png", help="Output figure path.")
    parser.add_argument("--crop", default=None, help="Crop box in HR coordinates: x,y,w,h.")
    parser.add_argument("--auto-crop-size", type=int, default=96, help="Patch size for automatic high-frequency crop.")
    parser.add_argument("--auto-stride", type=int, default=24, help="Stride for automatic crop search.")
    parser.add_argument("--zoom", type=int, default=3, help="Crop zoom factor.")
    args = parser.parse_args()

    gt = load_rgb(args.gt)
    tiny = load_rgb(args.tiny)
    ssgr = load_rgb(args.ssgr)
    images = []
    labels = []

    if args.bicubic:
        bicubic = load_rgb(args.bicubic)
        if bicubic.size != gt.size:
            bicubic = bicubic.resize(gt.size, Image.Resampling.BICUBIC)
        images.append(bicubic)
        labels.append("Bicubic")

    if tiny.size != gt.size:
        tiny = tiny.resize(gt.size, Image.Resampling.BICUBIC)
    if ssgr.size != gt.size:
        ssgr = ssgr.resize(gt.size, Image.Resampling.BICUBIC)

    images.extend([tiny, ssgr, gt])
    labels.extend(["MaIR-Tiny", "MaIR-Tiny+SSGR", "GT"])

    crop = parse_crop(args.crop)
    if crop is None:
        crop = auto_crop_from_gt(gt, args.auto_crop_size, args.auto_stride)
        print(f"Auto crop selected: x,y,w,h = {crop[0]},{crop[1]},{crop[2]},{crop[3]}")

    make_panel(images, labels, crop, args.zoom, Path(args.out))
    print(f"Saved comparison figure to: {Path(args.out).resolve()}")


if __name__ == "__main__":
    main()
