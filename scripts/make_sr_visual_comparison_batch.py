import argparse
from pathlib import Path

from make_sr_visual_comparison import load_rgb, auto_crop_from_gt, parse_crop, make_panel


def strip_suffix(name, suffix):
    token = f"_{suffix}"
    if name.endswith(token):
        return name[: -len(token)]
    return name


def gt_name_from_saved_stem(stem):
    """Map BasicSR saved SR filename stem back to the HR filename stem."""
    # Manga109 LR files are saved like Arisa_LRBI_x2_tiny.png.
    for token in ["_LRBI_x2", "_LRBI_x3", "_LRBI_x4"]:
        if stem.endswith(token):
            return stem[: -len(token)]
    # Urban100/Set5/Set14/B100 LR files are often saved like img004x2_tiny.png.
    for token in ["x2", "x3", "x4"]:
        if stem.endswith(token):
            return stem[: -len(token)]
    # Common SR filename templates are img004_tiny.png or baby_tiny.png.
    return stem


def collect_pairs(tiny_dir, ssgr_dir, gt_dir, tiny_suffix, ssgr_suffix):
    tiny_files = sorted(tiny_dir.glob(f"*.png"))
    pairs = []
    for tiny_path in tiny_files:
        tiny_stem = strip_suffix(tiny_path.stem, tiny_suffix)
        ssgr_path = ssgr_dir / f"{tiny_stem}_{ssgr_suffix}.png"
        gt_stem = gt_name_from_saved_stem(tiny_stem)
        gt_path = gt_dir / f"{gt_stem}.png"
        if ssgr_path.exists() and gt_path.exists():
            pairs.append((tiny_path, ssgr_path, gt_path, gt_stem))
    return pairs


def main():
    parser = argparse.ArgumentParser(
        description="Batch-create SR visual comparison crops for Tiny vs SSGR vs GT."
    )
    parser.add_argument("--tiny-dir", required=True, help="Directory with baseline Tiny saved images.")
    parser.add_argument("--ssgr-dir", required=True, help="Directory with SSGR saved images.")
    parser.add_argument("--gt-dir", required=True, help="Directory with GT HR images.")
    parser.add_argument("--out-dir", required=True, help="Output directory for comparison figures.")
    parser.add_argument("--tiny-suffix", default="tiny", help="Suffix used by baseline saved images.")
    parser.add_argument("--ssgr-suffix", default="ssgr", help="Suffix used by SSGR saved images.")
    parser.add_argument("--crop", default=None, help="Optional shared crop box x,y,w,h for all images.")
    parser.add_argument("--auto-crop-size", type=int, default=96)
    parser.add_argument("--auto-stride", type=int, default=24)
    parser.add_argument("--zoom", type=int, default=3)
    parser.add_argument("--limit", type=int, default=0, help="Process only the first N matched images. 0 means all.")
    args = parser.parse_args()

    tiny_dir = Path(args.tiny_dir)
    ssgr_dir = Path(args.ssgr_dir)
    gt_dir = Path(args.gt_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs = collect_pairs(tiny_dir, ssgr_dir, gt_dir, args.tiny_suffix, args.ssgr_suffix)
    if args.limit > 0:
        pairs = pairs[: args.limit]

    if not pairs:
        print("No matched image triplets found.")
        print(f"tiny_dir={tiny_dir}")
        print(f"ssgr_dir={ssgr_dir}")
        print(f"gt_dir={gt_dir}")
        print("\nTiny samples:")
        for p in sorted(tiny_dir.glob("*.png"))[:10]:
            print(f"  {p.name}")
        print("\nSSGR samples:")
        for p in sorted(ssgr_dir.glob("*.png"))[:10]:
            print(f"  {p.name}")
        print("\nGT samples:")
        for p in sorted(gt_dir.glob("*.png"))[:10]:
            print(f"  {p.name}")
        return

    fixed_crop = parse_crop(args.crop)
    print(f"Matched {len(pairs)} image triplets.")

    for tiny_path, ssgr_path, gt_path, name in pairs:
        tiny = load_rgb(tiny_path)
        ssgr = load_rgb(ssgr_path)
        gt = load_rgb(gt_path)
        if tiny.size != gt.size:
            tiny = tiny.resize(gt.size)
        if ssgr.size != gt.size:
            ssgr = ssgr.resize(gt.size)

        crop = fixed_crop or auto_crop_from_gt(gt, args.auto_crop_size, args.auto_stride)
        out_path = out_dir / f"{name}_compare.png"
        make_panel([tiny, ssgr, gt], ["MaIR-Tiny", "MaIR-Tiny+SSGR", "GT"], crop, args.zoom, out_path)
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
