import importlib
import logging
import sys
from copy import deepcopy
from functools import partial
from os import path as osp

import cv2
import torch

if __package__:
    from .data import create_dataloader, create_dataset
    from .models import create_model
    from .train import parse_options
    from .utils import get_env_info, get_root_logger, get_time_str, imwrite, make_exp_dirs, tensor2img
    from .utils.options import dict2str

    metric_module = importlib.import_module('.metrics', package=__package__)
else:
    CURRENT_DIR = osp.dirname(osp.abspath(__file__))
    REAL_DENOISING_ROOT = osp.abspath(osp.join(CURRENT_DIR, '..'))
    if REAL_DENOISING_ROOT not in sys.path:
        sys.path.insert(0, REAL_DENOISING_ROOT)

    from basicsr.data import create_dataloader, create_dataset
    from basicsr.models import create_model
    from basicsr.train import parse_options
    from basicsr.utils import get_env_info, get_root_logger, get_time_str, imwrite, make_exp_dirs, tensor2img
    from basicsr.utils.options import dict2str

    metric_module = importlib.import_module('basicsr.metrics')


def _resize_like(img, target_hw):
    target_h, target_w = target_hw
    if img.shape[0] == target_h and img.shape[1] == target_w:
        return img
    return cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_CUBIC)


def _save_visuals(opt, visuals, dataset_name, img_name, rgb2bgr):
    base_dir = osp.join(opt['path']['visualization'], dataset_name)
    tag = opt['val'].get('suffix') or opt['name']

    sr_img = tensor2img([visuals['result']], rgb2bgr=rgb2bgr)
    lq_img = tensor2img([visuals['lq']], rgb2bgr=rgb2bgr) if 'lq' in visuals else None
    gt_img = tensor2img([visuals['gt']], rgb2bgr=rgb2bgr) if 'gt' in visuals else None

    save_lq = bool(opt['val'].get('save_lq', False) or opt['val'].get('save_input', False))
    save_gt = bool(opt['val'].get('save_gt', False))
    save_compare = bool(opt['val'].get('save_compare', False))

    imwrite(sr_img, osp.join(base_dir, f'{img_name}_{tag}_result.png'))
    if save_lq and lq_img is not None:
        imwrite(lq_img, osp.join(base_dir, f'{img_name}_{tag}_lq.png'))
    if save_gt and gt_img is not None:
        imwrite(gt_img, osp.join(base_dir, f'{img_name}_{tag}_gt.png'))
    if save_compare:
        compare_parts = []
        target_hw = sr_img.shape[:2]
        if lq_img is not None:
            compare_parts.append(_resize_like(lq_img, target_hw))
        compare_parts.append(sr_img)
        if gt_img is not None:
            compare_parts.append(_resize_like(gt_img, target_hw))
        compare = cv2.hconcat(compare_parts)
        imwrite(compare, osp.join(base_dir, f'{img_name}_{tag}_compare.png'))

    return sr_img, gt_img


def _test_loader_ppt(model, dataloader, opt, logger, rgb2bgr=True, use_image=True):
    dataset_name = dataloader.dataset.opt['name']
    with_metrics = opt['val'].get('metrics') is not None
    max_num_images = opt['val'].get('max_num_images', None)
    if max_num_images is not None:
        max_num_images = max(int(max_num_images), 1)
        logger.info(f'Testing {dataset_name}: saving first {max_num_images} images for PPT.')
    else:
        logger.info(f'Testing {dataset_name}: saving all images for PPT.')

    metric_results = None
    if with_metrics:
        metric_results = {metric: 0 for metric in opt['val']['metrics'].keys()}

    window_size = opt['val'].get('window_size', 0)
    if window_size:
        test_fn = partial(model.pad_test, window_size)
    else:
        test_fn = model.nonpad_test

    total = len(dataloader) if max_num_images is None else min(len(dataloader), max_num_images)
    pbar = None
    if opt['val'].get('pbar', False):
        from tqdm import tqdm
        pbar = tqdm(total=total, unit='image')

    cnt = 0
    for idx, val_data in enumerate(dataloader):
        if max_num_images is not None and idx >= max_num_images:
            break

        img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
        model.feed_data(val_data)
        test_fn()

        visuals = model.get_current_visuals()
        sr_img, gt_img = _save_visuals(opt, visuals, dataset_name, img_name, rgb2bgr)

        if with_metrics:
            opt_metric = deepcopy(opt['val']['metrics'])
            if use_image:
                for name, metric_opt in opt_metric.items():
                    metric_type = metric_opt.pop('type')
                    metric_results[name] += getattr(metric_module, metric_type)(sr_img, gt_img, **metric_opt)
            else:
                for name, metric_opt in opt_metric.items():
                    metric_type = metric_opt.pop('type')
                    metric_results[name] += getattr(metric_module, metric_type)(
                        visuals['result'], visuals['gt'], **metric_opt
                    )

        if hasattr(model, 'gt'):
            del model.gt
        if hasattr(model, 'lq'):
            del model.lq
        if hasattr(model, 'output'):
            del model.output
        torch.cuda.empty_cache()

        cnt += 1
        if pbar is not None:
            pbar.update(1)
            pbar.set_description(f'Test {img_name}')

    if pbar is not None:
        pbar.close()

    if with_metrics and cnt > 0:
        for metric in metric_results.keys():
            metric_results[metric] /= cnt
        log_str = f'PPT Validation {dataset_name}\n'
        for metric, value in metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}\n'
        logger.info(log_str)
        return metric_results
    return None


def main():
    opt = parse_options(is_train=False)

    torch.backends.cudnn.benchmark = True

    make_exp_dirs(opt)
    log_file = osp.join(opt['path']['log'], f"test_ppt_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
    logger.info(get_env_info())
    logger.info(dict2str(opt))

    test_loaders = []
    for _, dataset_opt in sorted(opt['datasets'].items()):
        test_set = create_dataset(dataset_opt)
        test_loader = create_dataloader(
            test_set,
            dataset_opt,
            num_gpu=opt['num_gpu'],
            dist=opt['dist'],
            sampler=None,
            seed=opt['manual_seed'])
        logger.info(f"Number of test images in {dataset_opt['name']}: {len(test_set)}")
        test_loaders.append(test_loader)

    model = create_model(opt)

    rgb2bgr = opt['val'].get('rgb2bgr', True)
    use_image = opt['val'].get('use_image', True)
    avg_metrics = []
    for test_loader in test_loaders:
        result = _test_loader_ppt(model, test_loader, opt, logger, rgb2bgr=rgb2bgr, use_image=use_image)
        if result:
            avg_metrics.append(result)

    if avg_metrics:
        metric_names = avg_metrics[0].keys()
        log_str = 'PPT Validation Average\n'
        for metric_name in metric_names:
            metric_val = sum(m[metric_name] for m in avg_metrics) / len(avg_metrics)
            log_str += f'\t # {metric_name}: {metric_val:.4f}\n'
        logger.info(log_str)


if __name__ == '__main__':
    main()
