import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
import os

# --- 路径配置 ---
base_dir = '/root/autodl-tmp/2025-CVPR-MaIR/realDenoising/'
input_dir = '../datasets/RealDN/test/SIDD/'
result_mat_dir = os.path.join(base_dir, 'results/SIDD/mat/')

# 1. 加载数据
noisy_path = os.path.join(base_dir, input_dir, 'ValidationNoisyBlocksSrgb.mat')
gt_path = os.path.join(base_dir, input_dir, 'ValidationGtBlocksSrgb.mat')
denoised_path = os.path.join(result_mat_dir, 'Idenoised.mat')

print("正在加载数据，")
# 统一归一化到 0-1
noisy_data = sio.loadmat(noisy_path)['ValidationNoisyBlocksSrgb'] / 255.0
gt_data = sio.loadmat(gt_path)['ValidationGtBlocksSrgb'] / 255.0
denoised_data = sio.loadmat(denoised_path)['Idenoised']

# 2. 随机选择 20 个样本
np.random.seed(42) 
indices = np.random.choice(40 * 32, 20, replace=False)

# 3. 创建 20 行 4 列的对比图 (Noisy, Denoised, GT, Error Map)
# 调整 figsize，保证图片不会显得太挤
fig, axes = plt.subplots(20, 4, figsize=(16, 60))

for i, idx in enumerate(indices):
    scene_idx, patch_idx = divmod(idx, 32)
    
    noisy_img = noisy_data[scene_idx, patch_idx]
    denoised_img = denoised_data[scene_idx, patch_idx]
    gt_img = gt_data[scene_idx, patch_idx]
    
    # 计算误差图：归一化处理以便显示更清晰 (将差异放大 5 倍)
    diff_img = np.clip(np.abs(denoised_img - gt_img) * 5, 0, 1)
    
    # 绘制
    axes[i, 0].imshow(noisy_img)
    axes[i, 1].imshow(denoised_img)
    axes[i, 2].imshow(gt_img)
    axes[i, 3].imshow(diff_img, cmap='hot') # 使用 'hot' 颜色映射，误差大的地方显示为亮色
    
    # 标注标题
    if i == 0:
        axes[i, 0].set_title('Noisy (Input)')
        axes[i, 1].set_title('Denoised (MaIR)')
        axes[i, 2].set_title('Ground Truth')
        axes[i, 3].set_title('Error Map (Pred vs GT)')
    
    for j in range(4): axes[i, j].axis('off')

plt.tight_layout()
output_path = os.path.join(base_dir, 'results/comparison_20_samples.png')
plt.savefig(output_path, dpi=150) # 提高 dpi 保证细节清晰
print(f"===> 20个样本对比图已保存为: {output_path}")