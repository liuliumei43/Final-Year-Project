"""
批量评估 SSGR Deblur checkpoints，找最佳 iter。
支持多卡并行（每张卡处理一半图片，速度减半）。

用法（在 realDenoising 目录下）：
    # 单卡评估指定 iter
    CUDA_VISIBLE_DEVICES=9 python eval_deblur_checkpoints.py --iters 1000 5000

    # 双卡并行评估（速度 2x）
    CUDA_VISIBLE_DEVICES=8,9 python eval_deblur_checkpoints.py --iters 1000 5000

    # 评估所有 checkpoint
    CUDA_VISIBLE_DEVICES=8,9 python eval_deblur_checkpoints.py
"""

import argparse
import glob
import os
import random
import re
import sys
from copy import deepcopy

import torch
import torch.multiprocessing as mp


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--exp', default='experiments/train_MaIRUNet_SSGR_Deblur')
    p.add_argument('--opt', default='options/test/test_MaIRUNet_SSGR_Deblur.yml')
    p.add_argument('--iters', type=int, nargs='*',
                   help='iters to eval; omit to scan all checkpoints')
    p.add_argument('--ckpt', type=str, default=None,
                   help='directly evaluate a single checkpoint path (e.g. baseline)')
    p.add_argument('--max_images', type=int, default=0,
                   help='max images to eval per checkpoint (0=all, e.g. 100 for quick check)')
    return p.parse_args()


def collect_checkpoints(model_dir, iters=None):
    if iters:
        ckpts = []
        for it in sorted(iters):
            p = os.path.join(model_dir, f'net_g_{it}.pth')
            if os.path.exists(p):
                ckpts.append((it, p))
            else:
                print(f'[WARN] not found: {p}')
        return ckpts
    files = sorted(glob.glob(os.path.join(model_dir, 'net_g_*.pth')))
    ckpts = []
    for f in files:
        m = re.search(r'net_g_(\d+)\.pth', os.path.basename(f))
        if m:
            ckpts.append((int(m.group(1)), f))
    return sorted(ckpts)


def eval_worker(rank, num_gpus, opt, ckpt_path, result_queue, max_images=0):
    """Worker: evaluate subset of images on one GPU."""
    import importlib
    from basicsr.models import create_model
    from basicsr.data import create_dataset
    from basicsr.utils import tensor2img
    from tqdm import tqdm

    metric_module = importlib.import_module('basicsr.metrics')

    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}')

    opt = deepcopy(opt)
    opt['path']['pretrain_network_g'] = ckpt_path
    opt['path']['strict_load_g'] = False
    opt['is_train'] = False
    opt['num_gpu'] = 1
    opt['dist'] = False
    opt['rank'] = 0

    model = create_model(opt)
    model.net_g = model.net_g.to(device)
    model.net_g.eval()
    model.device = device

    dataset_opt = list(opt['datasets'].values())[0]
    test_set = create_dataset(dataset_opt)

    # Split dataset across GPUs by index
    total = len(test_set)
    all_indices = list(range(total))
    if max_images > 0 and max_images < total:
        # Random sample with fixed seed for reproducibility across runs
        rng = random.Random(42)
        rng.shuffle(all_indices)
        all_indices = sorted(all_indices[:max_images])  # sort back for consistent ordering
        total = max_images
    chunk = (total + num_gpus - 1) // num_gpus
    indices = all_indices[rank * chunk : min((rank + 1) * chunk, total)]

    from torch.utils.data import Subset, DataLoader
    subset = Subset(test_set, indices)
    loader = DataLoader(subset, batch_size=1, shuffle=False,
                        num_workers=8, pin_memory=True)

    window_size = opt['val'].get('window_size', 0)
    tile_size = opt['val'].get('tile', 0)
    tile_overlap = opt['val'].get('tile_overlap', 32)
    use_image = opt['val'].get('use_image', True)
    rgb2bgr = opt['val'].get('rgb2bgr', True)

    psnr_sum = ssim_sum = cnt = 0
    desc = f'GPU{rank} {os.path.basename(ckpt_path)}'
    with torch.no_grad():
        for val_data in tqdm(loader, desc=desc, unit='img',
                             ncols=80, position=rank):
            model.lq = val_data['lq'].to(device)
            if 'gt' in val_data:
                model.gt = val_data['gt'].to(device)

            if tile_size:
                model.tile_test(tile_size, tile_overlap, window_size)
            elif window_size:
                model.pad_test(window_size)
            else:
                model.nonpad_test()

            visuals = model.get_current_visuals()
            opt_metric = deepcopy(opt['val']['metrics'])

            if use_image:
                sr_img = tensor2img([visuals['result']], rgb2bgr=rgb2bgr)
                gt_img = tensor2img([visuals['gt']], rgb2bgr=rgb2bgr)
                for name, opt_ in opt_metric.items():
                    mtype = opt_.pop('type')
                    v = getattr(metric_module, mtype)(sr_img, gt_img, **opt_)
                    if name == 'psnr':
                        psnr_sum += v
                    elif name == 'ssim':
                        ssim_sum += v
            else:
                for name, opt_ in opt_metric.items():
                    mtype = opt_.pop('type')
                    v = getattr(metric_module, mtype)(
                        visuals['result'], visuals['gt'], **opt_)
                    if name == 'psnr':
                        psnr_sum += v
                    elif name == 'ssim':
                        ssim_sum += v

            del model.lq, model.output
            if hasattr(model, 'gt'):
                del model.gt
            cnt += 1

    result_queue.put((psnr_sum, ssim_sum, cnt))


def eval_one(opt, ckpt_path, max_images=0):
    """Evaluate one checkpoint, using all visible GPUs in parallel."""
    num_gpus = torch.cuda.device_count()
    if num_gpus <= 1:
        # Single GPU fallback
        q = mp.Queue()
        eval_worker(0, 1, opt, ckpt_path, q, max_images)
        psnr_sum, ssim_sum, cnt = q.get()
        return psnr_sum / cnt, ssim_sum / cnt

    # Multi-GPU: spawn one process per GPU
    ctx = mp.get_context('spawn')
    result_queue = ctx.Queue()
    procs = []
    for rank in range(num_gpus):
        p = ctx.Process(
            target=eval_worker,
            args=(rank, num_gpus, opt, ckpt_path, result_queue, max_images),
        )
        p.start()
        procs.append(p)
    for p in procs:
        p.join()

    total_psnr = total_ssim = total_cnt = 0
    for _ in range(num_gpus):
        ps, ss, cnt = result_queue.get()
        total_psnr += ps
        total_ssim += ss
        total_cnt += cnt

    return total_psnr / total_cnt, total_ssim / total_cnt


def main():
    args = parse_args()

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from basicsr.utils.options import parse
    opt = parse(args.opt, is_train=False)
    opt['dist'] = False
    opt['rank'] = 0
    opt['world_size'] = 1
    opt['num_gpu'] = 1

    num_gpus = torch.cuda.device_count()

    if args.ckpt:
        # Directly evaluate a single checkpoint (e.g. baseline)
        if not os.path.exists(args.ckpt):
            print(f'[ERROR] checkpoint not found: {args.ckpt}')
            sys.exit(1)
        ckpts = [('baseline', args.ckpt)]
    else:
        model_dir = os.path.join(args.exp, 'models')
        ckpts = collect_checkpoints(model_dir, args.iters)
    if not ckpts:
        print('[ERROR] no checkpoints found')
        sys.exit(1)

    print(f'Using {num_gpus} GPU(s) — evaluating {len(ckpts)} checkpoint(s)\n')
    print(f"{'iter':>8}  {'PSNR':>8}  {'SSIM':>8}")
    print('-' * 32)

    results = []
    for it, ckpt in ckpts:
        try:
            psnr, ssim = eval_one(opt, ckpt, args.max_images)
            results.append((it, psnr, ssim))
            print(f'{it:>8}  {psnr:>8.4f}  {ssim:>8.4f}')
        except Exception as e:
            import traceback
            print(f'{it:>8}  [ERROR] {e}')
            traceback.print_exc()

    if results and len(results) > 1:
        best = max(results, key=lambda x: x[1])
        print(f'\n★ Best: iter={best[0]}, PSNR={best[1]:.4f}, SSIM={best[2]:.4f}')


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()

