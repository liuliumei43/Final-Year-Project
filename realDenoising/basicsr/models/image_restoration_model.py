import importlib
import math
import torch
from collections import OrderedDict
from copy import deepcopy
from os import path as osp
from tqdm import tqdm

from basicsr.models.archs import define_network
from basicsr.models.base_model import BaseModel
from basicsr.utils import get_root_logger, imwrite, tensor2img

loss_module = importlib.import_module('basicsr.models.losses')
metric_module = importlib.import_module('basicsr.metrics')

import os
import random
import numpy as np
import cv2
import torch.nn.functional as F
from functools import partial

class Mixing_Augment:
    def __init__(self, mixup_beta, use_identity, device):
        self.dist = torch.distributions.beta.Beta(torch.tensor([mixup_beta]), torch.tensor([mixup_beta]))
        self.device = device

        self.use_identity = use_identity

        self.augments = [self.mixup]

    def mixup(self, target, input_):
        lam = self.dist.rsample((1,1)).item()
    
        r_index = torch.randperm(target.size(0)).to(self.device)
    
        target = lam * target + (1-lam) * target[r_index, :]
        input_ = lam * input_ + (1-lam) * input_[r_index, :]
    
        return target, input_

    def __call__(self, target, input_):
        if self.use_identity:
            augment = random.randint(0, len(self.augments))
            if augment < len(self.augments):
                target, input_ = self.augments[augment](target, input_)
        else:
            augment = random.randint(0, len(self.augments)-1)
            target, input_ = self.augments[augment](target, input_)
        return target, input_

class ImageCleanModel(BaseModel):
    """Base Deblur model for single image deblur."""

    def __init__(self, opt):
        super(ImageCleanModel, self).__init__(opt)

        # define network

        train_opt = self.opt.get('train') or {}
        mixing_opt = train_opt.get('mixing_augs', {})
        self.mixing_flag = self.is_train and mixing_opt.get('mixup', False)
        if self.mixing_flag:
            mixup_beta = mixing_opt.get('mixup_beta', 1.2)
            use_identity = mixing_opt.get('use_identity', False)
            self.mixing_augmentation = Mixing_Augment(mixup_beta, use_identity, self.device)

        self.net_g = define_network(deepcopy(opt['network_g']))
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            self.load_network(self.net_g, load_path,
                              self.opt['path'].get('strict_load_g', True), param_key=self.opt['path'].get('param_key', 'params'))

        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(
                f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = define_network(self.opt['network_g']).to(
                self.device)
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path,
                                  self.opt['path'].get('strict_load_g',
                                                       True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        # define losses
        if train_opt.get('pixel_opt'):
            pixel_type = train_opt['pixel_opt'].pop('type')
            cri_pix_cls = getattr(loss_module, pixel_type)
            self.cri_pix = cri_pix_cls(**train_opt['pixel_opt']).to(
                self.device)
        else:
            raise ValueError('pixel loss are None.')

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_type = train_opt['optim_g'].pop('type')
        base_lr = train_opt['optim_g']['lr']
        backbone_lr_scale = train_opt['optim_g'].pop('backbone_lr_scale', None)
        self.backbone_lr_scale = backbone_lr_scale  # save for update_learning_rate

        if backbone_lr_scale is not None:
            # Layered LR: adapter params get base_lr, backbone params get base_lr * scale
            adapter_params = []
            backbone_params = []
            logger = get_root_logger()
            for k, v in self.net_g.named_parameters():
                if not v.requires_grad:
                    logger.warning(f'Params {k} will not be optimized.')
                    continue
                if 'adapters.' in k or 'adapter' in k:
                    adapter_params.append(v)
                else:
                    backbone_params.append(v)
            backbone_lr = base_lr * backbone_lr_scale
            param_groups = [
                {'params': adapter_params, 'lr': base_lr},
                {'params': backbone_params, 'lr': backbone_lr},
            ]
            logger.info(
                f'Layered LR: adapter_lr={base_lr}, '
                f'backbone_lr={backbone_lr} (scale={backbone_lr_scale}), '
                f'adapter_params={len(adapter_params)}, '
                f'backbone_params={len(backbone_params)}'
            )
        else:
            param_groups = []
            for k, v in self.net_g.named_parameters():
                if v.requires_grad:
                    param_groups.append(v)
                else:
                    logger = get_root_logger()
                    logger.warning(f'Params {k} will not be optimized.')

        if optim_type == 'Adam':
            self.optimizer_g = torch.optim.Adam(param_groups, **train_opt['optim_g'])
        elif optim_type == 'AdamW':
            self.optimizer_g = torch.optim.AdamW(param_groups, **train_opt['optim_g'])
        else:
            raise NotImplementedError(
                f'optimizer {optim_type} is not supperted yet.')
        self.optimizers.append(self.optimizer_g)

    def update_learning_rate(self, current_iter, warmup_iter=-1):
        """Override to re-enforce backbone_lr_scale after scheduler step.

        CosineAnnealingRestartCyclicLR uses absolute eta_min values that
        override per-group LR ratios. After the scheduler sets LRs, we
        force backbone group lr = adapter group lr * backbone_lr_scale.
        """
        super().update_learning_rate(current_iter, warmup_iter)
        if self.backbone_lr_scale is not None:
            for optimizer in self.optimizers:
                if len(optimizer.param_groups) >= 2:
                    adapter_lr = optimizer.param_groups[0]['lr']
                    optimizer.param_groups[1]['lr'] = adapter_lr * self.backbone_lr_scale

    def feed_train_data(self, data):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

        if self.mixing_flag:
            self.gt, self.lq = self.mixing_augmentation(self.gt, self.lq)

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        preds = self.net_g(self.lq)
        if not isinstance(preds, list):
            preds = [preds]

        self.output = preds[-1]

        loss_dict = OrderedDict()
        # pixel loss
        l_pix = 0.
        for pred in preds:
            l_pix += self.cri_pix(pred, self.gt)

        loss_dict['l_pix'] = l_pix

        l_pix.backward()
        if self.opt['train']['use_grad_clip']:
            torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), 0.01)
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def pad_test(self, window_size):        
        scale = self.opt.get('scale', 1)
        mod_pad_h, mod_pad_w = 0, 0
        _, _, h, w = self.lq.size()
        if h % window_size != 0:
            mod_pad_h = window_size - h % window_size
        if w % window_size != 0:
            mod_pad_w = window_size - w % window_size
        img = F.pad(self.lq, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        self.nonpad_test(img)
        _, _, h, w = self.output.size()
        self.output = self.output[:, :, 0:h - mod_pad_h * scale, 0:w - mod_pad_w * scale]

    def nonpad_test(self, img=None):
        if img is None:
            img = self.lq      
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                pred = self.net_g_ema(img)
            if isinstance(pred, list):
                pred = pred[-1]
            self.output = pred
        else:
            self.net_g.eval()
            with torch.no_grad():
                pred = self.net_g(img)
            if isinstance(pred, list):
                pred = pred[-1]
            self.output = pred
            self.net_g.train()

    def tile_test(self, tile_size=256, tile_overlap=32, window_size=8):
        """Tile-based inference to avoid OOM on large images (e.g. GoPro 1280x720)."""
        scale = self.opt.get('scale', 1)
        b, c, h, w = self.lq.shape
        stride = tile_size - tile_overlap

        h_tiles = max(1, math.ceil((h - tile_overlap) / stride))
        w_tiles = max(1, math.ceil((w - tile_overlap) / stride))

        output = self.lq.new_zeros(b, c, h * scale, w * scale)
        weight = self.lq.new_zeros(b, 1, h * scale, w * scale)

        for i in range(h_tiles):
            for j in range(w_tiles):
                top = i * stride
                left = j * stride
                bottom = min(top + tile_size, h)
                right = min(left + tile_size, w)
                # pull back so each tile is exactly tile_size when possible
                top = max(0, bottom - tile_size)
                left = max(0, right - tile_size)

                tile = self.lq[:, :, top:bottom, left:right]
                tile_h, tile_w = tile.shape[2], tile.shape[3]

                # pad to window_size multiples
                mod_h = (window_size - tile_h % window_size) % window_size
                mod_w = (window_size - tile_w % window_size) % window_size
                if mod_h or mod_w:
                    tile = F.pad(tile, (0, mod_w, 0, mod_h), 'reflect')

                with torch.no_grad():
                    if hasattr(self, 'net_g_ema'):
                        self.net_g_ema.eval()
                        tile_out = self.net_g_ema(tile)
                    else:
                        self.net_g.eval()
                        tile_out = self.net_g(tile)
                    if isinstance(tile_out, list):
                        tile_out = tile_out[-1]

                tile_out = tile_out[:, :, :tile_h * scale, :tile_w * scale]
                output[:, :, top*scale:bottom*scale, left*scale:right*scale] += tile_out
                weight[:, :, top*scale:bottom*scale, left*scale:right*scale] += 1

        output /= weight
        self.output = output

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image):
        if os.environ['LOCAL_RANK'] == '0':
            result = self.nondist_validation(dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image)
        else:
            result = 0.
        # All ranks must wait here so rank 1 doesn't race ahead into
        # the next training iteration while rank 0 is still validating.
        torch.distributed.barrier()
        return result

    def nondist_validation(self, dataloader, current_iter, tb_logger,
                           save_img, rgb2bgr, use_image):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        if with_metrics:
            self.metric_results = {
                metric: 0
                for metric in self.opt['val']['metrics'].keys()
            }
        window_size = self.opt['val'].get('window_size', 0)
        tile_size = self.opt['val'].get('tile', 0)
        tile_overlap = self.opt['val'].get('tile_overlap', 32)

        if tile_size:
            test = partial(self.tile_test, tile_size, tile_overlap, window_size)
        elif window_size:
            test = partial(self.pad_test, window_size)
        else:
            test = self.nonpad_test

        cnt = 0
        val_iter = enumerate(dataloader)
        if self.opt['val'].get('progress_bar', False):
            val_iter = tqdm(
                val_iter,
                total=len(dataloader),
                unit='image',
                desc=f'Testing {dataset_name}',
            )

        for idx, val_data in val_iter:
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]

            self.feed_data(val_data)
            test()

            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals['result']], rgb2bgr=rgb2bgr)
            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']], rgb2bgr=rgb2bgr)
                del self.gt

            # tentative for out of GPU memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()

            if save_img:
                
                if self.opt['is_train']:
                    
                    save_img_path = osp.join(self.opt['path']['visualization'],
                                             img_name,
                                             f'{img_name}_{current_iter}.png')
                    
                    save_gt_img_path = osp.join(self.opt['path']['visualization'],
                                             img_name,
                                             f'{img_name}_{current_iter}_gt.png')
                else:
                    
                    save_img_path = osp.join(
                        self.opt['path']['visualization'], dataset_name,
                        f'{img_name}.png')
                    save_gt_img_path = osp.join(
                        self.opt['path']['visualization'], dataset_name,
                        f'{img_name}_gt.png')
                    
                imwrite(sr_img, save_img_path)
                imwrite(gt_img, save_gt_img_path)

            if with_metrics:
                # calculate metrics
                opt_metric = deepcopy(self.opt['val']['metrics'])
                if use_image:
                    for name, opt_ in opt_metric.items():
                        metric_type = opt_.pop('type')
                        self.metric_results[name] += getattr(
                            metric_module, metric_type)(sr_img, gt_img, **opt_)
                else:
                    for name, opt_ in opt_metric.items():
                        metric_type = opt_.pop('type')
                        self.metric_results[name] += getattr(
                            metric_module, metric_type)(visuals['result'], visuals['gt'], **opt_)

            cnt += 1

        current_metric = 0.
        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= cnt
                current_metric = self.metric_results[metric]

            self._log_validation_metric_values(current_iter, dataset_name,
                                               tb_logger)
        return current_metric


    def _log_validation_metric_values(self, current_iter, dataset_name,
                                      tb_logger):
        log_str = f'Validation {dataset_name},\t'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{metric}', value, current_iter)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        if self.ema_decay > 0:
            self.save_network([self.net_g, self.net_g_ema],
                              'net_g',
                              current_iter,
                              param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)
