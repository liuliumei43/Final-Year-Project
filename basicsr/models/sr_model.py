import torch
from collections import OrderedDict
from os import path as osp
from tqdm import tqdm

from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.metrics import calculate_metric
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.utils.registry import MODEL_REGISTRY
from .base_model import BaseModel


@MODEL_REGISTRY.register()
class SRModel(BaseModel):
    """Base SR model for single image super-resolution."""

    def __init__(self, opt):
        super(SRModel, self).__init__(opt)

        # define network
        self.net_g = build_network(opt['network_g'])
        # load pretrained models before DDP/DataParallel wrapping
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_g', 'params')
            # 必须对「裸模型」检查，避免 DP/DDP 包装后 hasattr 异常；MaIR_FFT 需用 load_pretrained_mair 把权重加载到 backbone
            if hasattr(self.net_g, 'load_pretrained_mair'):
                self.net_g.load_pretrained_mair(load_path, strict=False, param_key=param_key)
            else:
                self.load_network(self.net_g, load_path, self.opt['path'].get('strict_load_g', True), param_key)

        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)

        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
            # load pretrained model（与 net_g 一致：对裸模型调用 load_pretrained_mair 或走 load_network 的 backbone 兼容）
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                net_ema = self.get_bare_model(self.net_g_ema)
                param_key = self.opt['path'].get('param_key_g', 'params')
                if hasattr(net_ema, 'load_pretrained_mair'):
                    net_ema.load_pretrained_mair(load_path, strict=False, param_key=param_key)
                else:
                    self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), param_key)
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        self.net_t = None
        teacher_path = self.opt['path'].get('pretrain_network_t', None)
        teacher_opt = self.opt.get('network_t', None)
        if teacher_path is not None and teacher_opt is not None:
            logger = get_root_logger()
            self.net_t = build_network(teacher_opt).to(self.device)
            param_key_t = self.opt['path'].get('param_key_t', 'params')
            strict_load_t = self.opt['path'].get('strict_load_t', True)
            if hasattr(self.net_t, 'load_pretrained_mair'):
                self.net_t.load_pretrained_mair(teacher_path, strict=strict_load_t, param_key=param_key_t)
            else:
                self.load_network(self.net_t, teacher_path, strict_load_t, param_key_t)
            self.net_t.eval()
            for p in self.net_t.parameters():
                p.requires_grad = False
            logger.info(f'Loaded teacher network from {teacher_path} (param_key={param_key_t}).')

        # define losses
        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
        else:
            self.cri_pix = None

        if train_opt.get('perceptual_opt'):
            self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(self.device)
        else:
            self.cri_perceptual = None

        if train_opt.get('freq_opt'):
            self.cri_freq = build_loss(train_opt['freq_opt']).to(self.device)
        else:
            self.cri_freq = None

        if train_opt.get('luma_opt'):
            self.cri_luma = build_loss(train_opt['luma_opt']).to(self.device)
        else:
            self.cri_luma = None

        if train_opt.get('teacher_opt'):
            self.cri_teacher = build_loss(train_opt['teacher_opt']).to(self.device)
        else:
            self.cri_teacher = None

        if self.cri_pix is None and self.cri_perceptual is None:
            raise ValueError('Both pixel and perceptual losses are None.')

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        backbone_lr_scale = train_opt['optim_g'].pop('backbone_lr_scale', None)

        net = self.get_bare_model(self.net_g)
        optim_params = []
        backbone_params = []
        other_params = []
        frozen_param_names = []
        for k, v in net.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
                if backbone_lr_scale is not None and k.startswith('backbone.'):
                    backbone_params.append(v)
                else:
                    other_params.append(v)
            else:
                frozen_param_names.append(k)

        if frozen_param_names:
            logger = get_root_logger()
            preview = ', '.join(frozen_param_names[:12])
            more = '' if len(frozen_param_names) <= 12 else f' (+{len(frozen_param_names) - 12} more)'
            logger.info(f'Frozen params: {len(frozen_param_names)}. Preview: {preview}{more}')

        optim_type = train_opt['optim_g'].pop('type')
        lr = train_opt['optim_g'].get('lr')
        if backbone_lr_scale is not None and len(backbone_params) > 0 and len(other_params) > 0 and lr is not None:
            backbone_lr = lr * backbone_lr_scale
            param_groups = [
                {'params': backbone_params, 'lr': backbone_lr},
                {'params': other_params, 'lr': lr},
            ]
            logger = get_root_logger()
            logger.info(f'Optimizer param_groups: backbone lr={backbone_lr}, rest lr={lr} (backbone_lr_scale={backbone_lr_scale})')
            logger.info(
                f'Optimizer trainable tensors: backbone={len(backbone_params)}, other={len(other_params)}, '
                f'total={len(optim_params)}')
            self.optimizer_g = self.get_optimizer(optim_type, param_groups, **train_opt['optim_g'])
        else:
            logger = get_root_logger()
            logger.info(f'Optimizer trainable tensors: total={len(optim_params)}')
            self.optimizer_g = self.get_optimizer(optim_type, optim_params, **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.lq)
        teacher_output = None
        if self.net_t is not None:
            with torch.no_grad():
                teacher_output = self.net_t(self.lq)

        l_total = 0
        loss_dict = OrderedDict()
        # pixel loss
        if self.cri_pix:
            l_pix = self.cri_pix(self.output, self.gt)
            l_total += l_pix
            loss_dict['l_pix'] = l_pix
        # perceptual loss
        if self.cri_perceptual:
            l_percep, l_style = self.cri_perceptual(self.output, self.gt)
            if l_percep is not None:
                l_total += l_percep
                loss_dict['l_percep'] = l_percep
            if l_style is not None:
                l_total += l_style
                loss_dict['l_style'] = l_style
        # 频域损失（配合 FFT 分支）：在频域约束 pred 与 gt 一致，减轻 FFT 分支学到噪声
        if self.cri_freq:
            l_freq = self.cri_freq(self.output, self.gt)
            l_total += l_freq
            loss_dict['l_freq'] = l_freq
        if self.cri_luma:
            l_luma = self.cri_luma(self.output, self.gt)
            l_total += l_luma
            loss_dict['l_luma'] = l_luma
        if self.cri_teacher and teacher_output is not None:
            l_teacher = self.cri_teacher(self.output, teacher_output, gt=self.gt)
            l_total += l_teacher
            loss_dict['l_teacher'] = l_teacher

        if not torch.isfinite(l_total):
            logger = get_root_logger()
            logger.warning(
                f'Non-finite loss detected at iter {current_iter}. '
                f'Skip this update. l_pix={loss_dict.get("l_pix")}, '
                f'l_freq={loss_dict.get("l_freq")}, l_luma={loss_dict.get("l_luma")}'
            )
            self.optimizer_g.zero_grad(set_to_none=True)
            sanitized_loss = OrderedDict()
            for k, v in loss_dict.items():
                sanitized_loss[k] = torch.nan_to_num(v.detach(), nan=0.0, posinf=1e6, neginf=-1e6)
            self.log_dict = self.reduce_loss_dict(sanitized_loss)
            return

        # bypass_fft_fusion=True 时前向只走冻结的 backbone，输出无梯度，不能 backward
        if l_total.requires_grad and l_total.grad_fn is not None:
            l_total.backward()
            net = self.get_bare_model(self.net_g)
            sanitize_nonfinite_grads = bool(self.opt['train'].get('sanitize_nonfinite_grads', False))
            nonfinite_grad_names = []
            nonfinite_grad_stats = []
            for name, p in net.named_parameters():
                if p.grad is not None and not torch.isfinite(p.grad).all():
                    nonfinite_grad_names.append(name)
                    grad = p.grad.detach()
                    finite_grad = grad[torch.isfinite(grad)]
                    if finite_grad.numel() > 0:
                        grad_abs_max = finite_grad.abs().max().item()
                        grad_abs_mean = finite_grad.abs().mean().item()
                    else:
                        grad_abs_max = float('nan')
                        grad_abs_mean = float('nan')
                    nonfinite_grad_stats.append((name, grad_abs_max, grad_abs_mean))
                    if sanitize_nonfinite_grads:
                        p.grad = torch.nan_to_num(p.grad, nan=0.0, posinf=0.0, neginf=0.0)
            if len(nonfinite_grad_names) > 0 and not sanitize_nonfinite_grads:
                logger = get_root_logger()
                first_name, grad_abs_max, grad_abs_mean = nonfinite_grad_stats[0]
                logger.warning(
                    f'Non-finite gradients detected at iter {current_iter}. '
                    f'First bad param: {first_name}, finite_grad_abs_max={grad_abs_max:.4e}, '
                    f'finite_grad_abs_mean={grad_abs_mean:.4e}. Skip optimizer step.'
                )
                self.optimizer_g.zero_grad(set_to_none=True)
                sanitized_loss = OrderedDict()
                for k, v in loss_dict.items():
                    sanitized_loss[k] = torch.nan_to_num(v.detach(), nan=0.0, posinf=1e6, neginf=-1e6)
                self.log_dict = self.reduce_loss_dict(sanitized_loss)
                return
            elif len(nonfinite_grad_names) > 0 and sanitize_nonfinite_grads:
                logger = get_root_logger()
                first_name, grad_abs_max, grad_abs_mean = nonfinite_grad_stats[0]
                logger.warning(
                    f'Non-finite gradients detected at iter {current_iter}. '
                    f'Sanitized {len(nonfinite_grad_names)} param gradients. '
                    f'First bad param: {first_name}, finite_grad_abs_max={grad_abs_max:.4e}, '
                    f'finite_grad_abs_mean={grad_abs_mean:.4e}.'
                )

            grad_clip = self.opt['train'].get('grad_clip', None)
            if grad_clip is not None and float(grad_clip) > 0:
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=float(grad_clip))
            self.optimizer_g.step()
        # else: 仅前向与 loss 记录，不更新参数（如用于验证加载后的 backbone PSNR）

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def test(self):
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                self.output = self.net_g_ema(self.lq)
        else:
            self.net_g.eval()
            with torch.no_grad():
                self.output = self.net_g(self.lq)
            self.net_g.train()

    def test_selfensemble(self):
        # TODO: to be tested
        # 8 augmentations
        # modified from https://github.com/thstkdgus35/EDSR-PyTorch

        def _transform(v, op):
            # if self.precision != 'single': v = v.float()
            v2np = v.data.cpu().numpy()
            if op == 'v':
                tfnp = v2np[:, :, :, ::-1].copy()
            elif op == 'h':
                tfnp = v2np[:, :, ::-1, :].copy()
            elif op == 't':
                tfnp = v2np.transpose((0, 1, 3, 2)).copy()

            ret = torch.Tensor(tfnp).to(self.device)
            # if self.precision == 'half': ret = ret.half()

            return ret

        # prepare augmented data
        lq_list = [self.lq]
        for tf in 'v', 'h', 't':
            lq_list.extend([_transform(t, tf) for t in lq_list])

        # inference
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                out_list = [self.net_g_ema(aug) for aug in lq_list]
        else:
            self.net_g.eval()
            with torch.no_grad():
                out_list = [self.net_g(aug) for aug in lq_list]
            self.net_g.train()

        # merge results
        for i in range(len(out_list)):
            if i > 3:
                out_list[i] = _transform(out_list[i], 't')
            if i % 4 > 1:
                out_list[i] = _transform(out_list[i], 'h')
            if (i % 4) % 2 == 1:
                out_list[i] = _transform(out_list[i], 'v')
        output = torch.cat(out_list, dim=0)

        self.output = output.mean(dim=0, keepdim=True)


    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        if self.opt['rank'] == 0:
            return self.nondist_validation(dataloader, current_iter, tb_logger, save_img)
        return None

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        use_pbar = self.opt['val'].get('pbar', False)

        if with_metrics:
            if not hasattr(self, 'metric_results'):  # only execute in the first run
                self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
            # initialize the best metric results for each dataset_name (supporting multiple validation datasets)
            self._initialize_best_metric_results(dataset_name)
        # zero self.metric_results
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.metric_results}

        metric_data = dict()
        if use_pbar:
            pbar = tqdm(total=len(dataloader), unit='image')

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            self.feed_data(val_data)
            if self.opt['val'].get('selfensemble', False):
                self.test_selfensemble()
            else:
                self.test()

            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals['result']])
            metric_data['img'] = sr_img
            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']])
                metric_data['img2'] = gt_img
                del self.gt

            # tentative for out of GPU memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()

            if save_img:
                if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'], img_name,
                                             f'{img_name}_{current_iter}.png')
                else:
                    if self.opt['val']['suffix']:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["val"]["suffix"]}.png')
                    else:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["name"]}.png')
                imwrite(sr_img, save_img_path)

            if with_metrics:
                # calculate metrics
                for name, opt_ in self.opt['val']['metrics'].items():
                    self.metric_results[name] += calculate_metric(metric_data, opt_)
            if use_pbar:
                pbar.update(1)
                pbar.set_description(f'Test {img_name}')
        if use_pbar:
            pbar.close()

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)
                # update the best metric result
                self._update_best_metric_result(dataset_name, metric, self.metric_results[metric], current_iter)

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)
            return dict(self.metric_results)
        return None

    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f'Validation {dataset_name}\n'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
            if hasattr(self, 'best_metric_results'):
                log_str += (f'\tBest: {self.best_metric_results[dataset_name][metric]["val"]:.4f} @ '
                            f'{self.best_metric_results[dataset_name][metric]["iter"]} iter')
            log_str += '\n'

        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{dataset_name}/{metric}', value, current_iter)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        if hasattr(self, 'net_g_ema'):
            self.save_network([self.net_g, self.net_g_ema], 'net_g', current_iter, param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)
