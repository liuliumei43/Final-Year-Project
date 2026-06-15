import argparse
import os
import sys

import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import yaml
from skimage import img_as_ubyte
from tqdm import tqdm

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

import utils
from basicsr.models.archs.mairunet_arch import MaIRUNet


def _resolve_path(path, base_dir=PROJECT_ROOT):
    if os.path.isabs(path):
        return path
    direct = os.path.abspath(path)
    if os.path.exists(direct):
        return direct
    return os.path.abspath(os.path.join(base_dir, path))


parser = argparse.ArgumentParser(description='Real Image Denoising on SIDD validation mat file')
parser.add_argument(
    '--input_dir',
    default=os.path.join(PROJECT_ROOT, 'datasets', 'SIDD', 'val'),
    type=str,
    help='Directory containing ValidationNoisyBlocksSrgb.mat',
)
parser.add_argument(
    '--result_dir',
    default=os.path.join(CURRENT_DIR, 'results', 'Real_Denoising', 'SIDD'),
    type=str,
    help='Directory for denoised results',
)
parser.add_argument(
    '--weights',
    default=os.path.join(PROJECT_ROOT, 'ckpt', 'MaIR_RealDN.pth'),
    type=str,
    help='Path to model weights',
)
parser.add_argument('--save_images', action='store_true', help='Save denoised PNG patches')

args = parser.parse_args()
args.input_dir = _resolve_path(args.input_dir)
args.result_dir = _resolve_path(args.result_dir, base_dir=CURRENT_DIR)
args.weights = _resolve_path(args.weights)

opt_str = r"""
  type: MaIRUNet
  inp_channels: 3
  out_channels: 3
  dim: 48
  num_blocks: [4, 6, 6, 8]
  num_refinement_blocks: 4

  ssm_ratio: 2.0
  flp_ratio: 4.0
  mlp_ratio: 1.5
  bias: False
  dual_pixel_task: False

  img_size: 128
  scan_len: 4
  batch_size: 8
  dynamic_ids: False
"""

x = yaml.safe_load(opt_str)
x.pop('type')

result_dir_mat = os.path.join(args.result_dir, 'mat')
os.makedirs(result_dir_mat, exist_ok=True)

if args.save_images:
    result_dir_png = os.path.join(args.result_dir, 'png')
    os.makedirs(result_dir_png, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_restoration = MaIRUNet(**x)
checkpoint = torch.load(args.weights, map_location=device)
model_restoration.load_state_dict(checkpoint['params'])
print('===>Testing using weights: ', args.weights)
model_restoration.to(device)
if torch.cuda.is_available():
    model_restoration = nn.DataParallel(model_restoration)
model_restoration.eval()

filepath = os.path.join(args.input_dir, 'ValidationNoisyBlocksSrgb.mat')
if not os.path.exists(filepath):
    raise FileNotFoundError(
        f'SIDD noisy mat file not found: {filepath}\n'
        f'Resolved input_dir: {args.input_dir}'
    )

img = sio.loadmat(filepath)
Inoisy = np.float32(np.array(img['ValidationNoisyBlocksSrgb'])) / 255.0
restored = np.zeros_like(Inoisy)

with torch.no_grad():
    for i in tqdm(range(40)):
        for k in range(32):
            noisy_patch = torch.from_numpy(Inoisy[i, k, :, :, :]).unsqueeze(0).permute(0, 3, 1, 2).to(device)
            restored_patch = model_restoration(noisy_patch)
            restored_patch = torch.clamp(restored_patch, 0, 1).cpu().detach().permute(0, 2, 3, 1).squeeze(0)
            restored[i, k, :, :, :] = restored_patch

            if args.save_images:
                save_file = os.path.join(result_dir_png, f'{i + 1:04d}_{k + 1:02d}.png')
                utils.save_img(save_file, img_as_ubyte(restored_patch))

sio.savemat(os.path.join(result_dir_mat, 'Idenoised.mat'), {'Idenoised': restored})
