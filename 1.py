# # # # import subprocess

# # # # def install_requirements(file_path):
# # # #     with open(file_path, 'r') as file:
# # # #         for line in file:
# # # #             # 去掉空白字符和注释
# # # #             line = line.strip()
# # # #             if not line or line.startswith('#'):
# # # #                 continue

# # # #             # 尝试安装包
# # # #             try:
# # # #                 print(f"正在安装: {line}")
# # # #                 subprocess.check_call(['pip', 'install', line])
# # # #                 print(f"安装成功: {line}")
# # # #             except subprocess.CalledProcessError as e:
# # # #                 print(f"安装失败: {line}, 错误信息: {e}")
# # # #                 continue

# # # # if __name__ == "__main__":
# # # #     requirements_file = "requirement.txt"  # 替换为你的 requirements.txt 文件路径
# # # #     install_requirements(requirements_file)

# # # import os
# # # import json

# # # # 定义文件夹路径
# # # hazy_folder = '/home/uchihawdt/DehazeFormer/data/RESIDE-IN/train_1399/hazy'
# # # gt_folder = '/home/uchihawdt/DehazeFormer/data/RESIDE-IN/train_1399/GT'
# # # output_json_path = '/home/uchihawdt/control-net-main-v0.5/data/train_1399.json'

# # # # 获取hazy文件夹中所有图片的文件名
# # # hazy_files = [f for f in os.listdir(hazy_folder) if f.endswith(('jpg', 'png', 'jpeg'))]

# # # # 打开文件以便写入
# # # with open(output_json_path, 'w') as json_file:
# # #     # 遍历hazy文件夹中的每个文件，并找到对应的GT文件
# # #     for hazy_file in hazy_files:
# # #         # 构造hazy和GT的完整路径
# # #         hazy_path = os.path.join(hazy_folder, hazy_file)
# # #         gt_path = os.path.join(gt_folder, hazy_file)  # 假设GT文件夹有相同的文件名

# # #         # 检查GT文件是否存在
# # #         if os.path.exists(gt_path):
# # #             # 写入每条数据，并确保数据之间有换行
# # #             json.dump({'source': hazy_path, 'target': gt_path}, json_file)
# # #             json_file.write('\n')  # 每条数据写入一行

# # # print(f"数据已成功写入 {output_json_path}")

# # from huggingface_hub import snapshot_download

# # # 下载到指定目录
# # snapshot_download(
# #     repo_id="openai/clip-vit-large-patch14",
# #     local_dir="openai/clip-vit-large-patch14",
# #     local_dir_use_symlinks=False
# # )
# import os
# import numpy as np
# from skimage.metrics import structural_similarity as ssim
# from skimage.metrics import peak_signal_noise_ratio as psnr
# from skimage.io import imread
# from tqdm import tqdm

# # 设置路径
# gt_path = "/home/uchihawdt/DehazeFormer/data/RESIDE-IN/test/GT"
# gen_path = "/home/uchihawdt/control-net-main-v0.5/output/few0_new"

# # 获取文件列表（假设文件名一一对应）
# gt_files = sorted([f for f in os.listdir(gt_path) if f.endswith(('.png', '.jpg', '.jpeg'))])
# gen_files = sorted([f for f in os.listdir(gen_path) if f.endswith(('.png', '.jpg', '.jpeg'))])

# assert len(gt_files) == len(gen_files), "文件数量不匹配"

# # 初始化结果存储
# ssim_values = []
# psnr_values = []

# # 遍历所有图像对
# for gt_file, gen_file in tqdm(zip(gt_files, gen_files), total=len(gt_files)):
#     # 读取图像（转换为float32并归一化到0-1范围）
#     gt_img = imread(os.path.join(gt_path, gt_file)).astype(np.float32) / 255.0
#     gen_img = imread(os.path.join(gen_path, gen_file)).astype(np.float32) / 255.0
    
#     # 确保图像尺寸相同
#     if gt_img.shape != gen_img.shape:
#         gen_img = np.resize(gen_img, gt_img.shape)
    
#     # 计算指标（多通道图像需指定multichannel=True）
#     ssim_val = ssim(gt_img, gen_img, multichannel=True, data_range=1.0)
#     psnr_val = psnr(gt_img, gen_img, data_range=1.0)
    
#     ssim_values.append(ssim_val)
#     psnr_values.append(psnr_val)

# # 输出平均结果
# print(f"平均 SSIM: {np.mean(ssim_values):.4f} ± {np.std(ssim_values):.4f}")
# print(f"平均 PSNR: {np.mean(psnr_values):.2f} dB ± {np.std(psnr_values):.2f}")

import os
import argparse
import torch
import torch.nn.functional as F
from pytorch_msssim import ssim
from torch.utils.data import Dataset, DataLoader
from collections import OrderedDict
from tqdm import tqdm
import numpy as np
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('--gt_dir', required=True, type=str, help='Path to ground truth images')
parser.add_argument('--gen_dir', required=True, type=str, help='Path to generated images')
parser.add_argument('--num_workers', default=4, type=int, help='Number of data loading workers')
args = parser.parse_args()

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
def resize_tensor(tensor, target_size):
    """Resize tensor to match target size using interpolation"""
    return F.interpolate(tensor, size=target_size, mode='bilinear', align_corners=False)

class ImagePairDataset(Dataset):
    def __init__(self, gt_dir, gen_dir):
        self.gt_dir = gt_dir
        self.gen_dir = gen_dir
        self.gt_files = sorted([f for f in os.listdir(gt_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        self.gen_files = sorted([f for f in os.listdir(gen_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        
        assert len(self.gt_files) == len(self.gen_files), \
               f"File count mismatch: GT={len(self.gt_files)}, Generated={len(self.gen_files)}"

    def __len__(self):
        return len(self.gt_files)

    def __getitem__(self, idx):
        gt_path = os.path.join(self.gt_dir, self.gt_files[idx])
        gen_path = os.path.join(self.gen_dir, self.gen_files[idx])
        
        # Load images and convert to tensor
        gt_img = torch.from_numpy(np.array(Image.open(gt_path))).float() / 255.0
        gen_img = torch.from_numpy(np.array(Image.open(gen_path))).float() / 255.0
        
        # Ensure channel dimension (for grayscale images)
        if gt_img.ndim == 2:
            gt_img = gt_img.unsqueeze(-1)
        if gen_img.ndim == 2:
            gen_img = gen_img.unsqueeze(-1)
            
        # Convert HWC to CHW
        gt_img = gt_img.permute(2, 0, 1)
        gen_img = gen_img.permute(2, 0, 1)
        
        return {
            'gt': gt_img,
            'gen': gen_img,
            'filename': self.gt_files[idx]
        }

def calculate_metrics():
    # Initialize metrics
    PSNR = AverageMeter()
    SSIM = AverageMeter()
    
    # Create dataset and loader
    dataset = ImagePairDataset(args.gt_dir, args.gen_dir)
    loader = DataLoader(dataset, batch_size=1, num_workers=args.num_workers)
    
    # Calculate metrics for each image pair
    for batch in tqdm(loader, desc='Evaluating'):
        gt = batch['gt'].cuda()
        gen = batch['gen'].cuda()
        
        if gt.shape[-2:] != gen.shape[-2:]:
            # print(f"尺寸不匹配: GT={gt.shape[2:]} vs Gen={gen.shape[2:]}, 自动调整...")
            target_size = gt.shape[2:]  # 以GT尺寸为准
            gen = resize_tensor(gen, target_size)
        
        with torch.no_grad():
            # PSNR calculation
            mse = F.mse_loss(gt, gen)
            psnr = 10 * torch.log10(1 / mse).item()
            
            # SSIM calculation with adaptive window size
            _, _, H, W = gt.shape
            down_ratio = max(1, round(min(H, W) / 256))
            ssim_val = ssim(
                F.adaptive_avg_pool2d(gen, (int(H/down_ratio), int(W/down_ratio))),
                F.adaptive_avg_pool2d(gt, (int(H/down_ratio), int(W/down_ratio))),
                data_range=1,
                size_average=False
            ).item()
            
        PSNR.update(psnr)
        SSIM.update(ssim_val)
    
    # Print final results
    print(f'\nFinal Results:')
    print(f'PSNR: {PSNR.avg:.2f} dB')
    print(f'SSIM: {SSIM.avg:.4f}')

if __name__ == '__main__':
    torch.cuda.empty_cache()
    calculate_metrics()


# import os
# import json

# # 定义文件夹路径
# hazy_folder = '/home/uchihawdt/DehazeFormer/data/D-HAZY/test/hazy'
# gt_folder = '/home/uchihawdt/DehazeFormer/data/D-HAZY/test/GT'
# output_json_path = '/home/uchihawdt/control-net-main-v0.5/data/testdata_d.json'

# # 获取hazy文件夹中所有图片的文件名
# hazy_files = [f for f in os.listdir(hazy_folder) if f.endswith(('jpg', 'png', 'jpeg'))]

# # 打开文件以便写入
# with open(output_json_path, 'w') as json_file:
#     # 遍历hazy文件夹中的每个文件，并找到对应的GT文件
#     for hazy_file in hazy_files:
#         # 构造hazy和GT的完整路径
#         hazy_path = os.path.join(hazy_folder, hazy_file)
#         gt_path = os.path.join(gt_folder, hazy_file)  # 假设GT文件夹有相同的文件名

#         # 检查GT文件是否存在
#         if os.path.exists(gt_path):
#             # 写入每条数据，并确保数据之间有换行
#             json.dump({'source': hazy_path, 'target': gt_path}, json_file)
#             json_file.write('\n')  # 每条数据写入一行

# print(f"数据已成功写入 {output_json_path}")