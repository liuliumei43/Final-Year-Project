import logging
import sys
from os import path as osp

import cv2
import torch
from tqdm import tqdm

# for some possible IMPORT ERROR
sys.path.append('./')

from basicsr.data import build_dataloader, build_dataset
from basicsr.metrics import calculate_metric
from basicsr.models import build_model
from basicsr.utils import get_root_logger, get_time_str, imwrite, make_exp_dirs, tensor2img
from basicsr.utils.options import dict2str, parse_options


def _resize_like(img, target_hw):
    target_h, target_w = target_hw
    if img.shape[0] == target_h and img.shape[1] == target_w:
        return img
    return cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_CUBIC)


def _save_visuals(opt, visuals, dataset_name, img_name, tag):
    base_dir = osp.join(opt['path']['visualization'], dataset_name)
    sr_img = tensor2img([visuals['result']])
    lq_img = tensor2img([visuals['lq']]) if 'lq' in visuals else None
    gt_img = tensor2img([visuals['gt']]) if 'gt' in visuals else None

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


def _test_loader_ppt(model, test_loader, opt, logger):
    dataset_name = test_loader.dataset.opt['name']
    metrics_opt = opt['val'].get('metrics')
    use_pbar = bool(opt['val'].get('pbar', False))
    max_num_images = opt['val'].get('max_num_images', None)
    if max_num_images is not None:
        max_num_images = max(int(max_num_images), 1)
        logger.info(f'Testing {dataset_name}: saving first {max_num_images} images for PPT.')
    else:
        logger.info(f'Testing {dataset_name}: saving all images for PPT.')

    metric_results = None
    if metrics_opt:
        metric_results = {metric: 0 for metric in metrics_opt.keys()}

    total = len(test_loader) if max_num_images is None else min(len(test_loader), max_num_images)
    pbar = tqdm(total=total, unit='image') if use_pbar else None
    num_processed = 0
    suffix = opt['val'].get('suffix') or opt['name']

    for idx, val_data in enumerate(test_loader):
        if max_num_images is not None and idx >= max_num_images:
            break

        img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
        model.feed_data(val_data)
        if opt['val'].get('selfensemble', False):
            model.test_selfensemble()
        else:
            model.test()

        visuals = model.get_current_visuals()
        sr_img, gt_img = _save_visuals(opt, visuals, dataset_name, img_name, suffix)

        if metric_results is not None and gt_img is not None:
            metric_data = {'img': sr_img, 'img2': gt_img}
            for name, metric_opt in metrics_opt.items():
                metric_results[name] += calculate_metric(metric_data, metric_opt)

        if hasattr(model, 'gt'):
            del model.gt
        if hasattr(model, 'lq'):
            del model.lq
        if hasattr(model, 'output'):
            del model.output
        torch.cuda.empty_cache()

        num_processed += 1
        if pbar is not None:
            pbar.update(1)
            pbar.set_description(f'Test {img_name}')

    if pbar is not None:
        pbar.close()

    if metric_results is not None and num_processed > 0:
        for key in metric_results:
            metric_results[key] /= num_processed
        log_str = f'PPT Validation {dataset_name}\n'
        for metric, value in metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}\n'
        logger.info(log_str)
        return metric_results
    return None


def test_pipeline(root_path):
    opt, _ = parse_options(root_path, is_train=False)

    torch.backends.cudnn.benchmark = True

    make_exp_dirs(opt)
    log_file = osp.join(opt['path']['log'], f"test_ppt_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
    logger.info(dict2str(opt))

    test_loaders = []
    for _, dataset_opt in sorted(opt['datasets'].items()):
        test_set = build_dataset(dataset_opt)
        test_loader = build_dataloader(
            test_set, dataset_opt, num_gpu=opt['num_gpu'], dist=opt['dist'], sampler=None, seed=opt['manual_seed'])
        logger.info(f"Number of test images in {dataset_opt['name']}: {len(test_set)}")
        test_loaders.append(test_loader)

    model = build_model(opt)

    avg_metrics = []
    for test_loader in test_loaders:
        result = _test_loader_ppt(model, test_loader, opt, logger)
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
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    test_pipeline(root_path)
