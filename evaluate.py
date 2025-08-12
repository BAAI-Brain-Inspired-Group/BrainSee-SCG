import pyiqa
import os
import argparse
from pathlib import Path
import torch
from utils import util_image
import tqdm
import torch.nn.functional as F
import pandas as pd
import os
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def save_output_to_file(in_path, lr_path_list, result, output_file):
    with open(output_file, 'a') as f:
        in_path_suffix = os.path.join(str(in_path).split('/')[-4], str(in_path).split('/')[-2])
        f.write(f'Find {len(lr_path_list)} images in {in_path_suffix}   ')    #\n
        for key, res in result.items():
            f.write(f"{key}: {res/len(lr_path_list):.5f}   ")    #\n
        f.write('\n')

def save_output_to_excel(in_path, lr_path_list, result, output_file):
    in_path_suffix = os.path.join(str(in_path).split('/')[-4], str(in_path).split('/')[-2])
    
    data = {
        "In Path Suffix": [in_path_suffix],
        "Number of Images": [len(lr_path_list)]
    }
    for key, res in result.items():
        data[key] = [res / len(lr_path_list)]
    df = pd.DataFrame(data)
    # try:
    #     with pd.ExcelWriter(output_file, mode='a', engine='openpyxl', if_sheet_exists='overlay') as writer:
    #         df.to_excel(writer, sheet_name='Results', index=False, header=writer.sheets['Results'].max_row == 1)
    # except FileNotFoundError:
    #     df.to_excel(output_file, sheet_name='Results', index=False)
    if not os.path.exists(output_file):
        df.to_excel(output_file, sheet_name='Results', index=False)
    else:
        with pd.ExcelWriter(output_file, mode='a', engine='openpyxl', if_sheet_exists='overlay') as writer:
            if 'Results' in writer.sheets:
                startrow = writer.sheets['Results'].max_row
                df.to_excel(writer, sheet_name='Results', index=False, header=False, startrow=startrow)
            else:
                df.to_excel(writer, sheet_name='Results', index=False)

def modify_path(img_path):
    img_path = str(img_path)
    directory, filename = img_path.rsplit('/', 1)
    name, ext = filename.rsplit('.', 1) 
    
    new_directory = directory
    if 'inDoor_dehazied_' in directory:
        base_dir = directory.split('inDoor_dehazied_')[0]
        new_directory = f"{base_dir}GT"#GT_images"#GT"#
    if 'outDoor_dehazied_' in directory:
        base_dir = directory.split('outDoor_dehazied_')[0]
        new_directory = f"{base_dir}GT"#GT_images"#GT" #      
    if 'IandODoor_dehazied_' in directory:
        base_dir = directory.split('IandODoor_dehazied_')[0]
        new_directory = f"{base_dir}GT"#GT_images"#GT" #    
    
    name2 = name.replace('hazy', 'GT')
    new_directory = new_directory.replace('/three/few10p', '')  #/Zero-shot
    
    new_path = f"{new_directory}/{name2}.{ext}"
    
    return new_path

def evaluate(in_path, ref_path, ntest):
    metric_dict = {}
    metric_dict["clipiqa"] = pyiqa.create_metric('clipiqa').to(device)
    metric_dict["musiq"] = pyiqa.create_metric('musiq').to(device)
    metric_paired_dict = {}
    
    in_path = Path(in_path) if not isinstance(in_path, Path) else in_path
    assert in_path.is_dir()
    
    # ref_path_list = None
    # if ref_path is not None:
    #     ref_path = Path(ref_path) if not isinstance(ref_path, Path) else ref_path
    #     ref_path_list = sorted([x for x in ref_path.glob("*.[jpJP][pnPN]*[gG]")])
    #     if ntest is not None: ref_path_list = ref_path_list[:ntest]
    metric_paired_dict["lpips"]=pyiqa.create_metric('lpips').to(device)
    metric_paired_dict["psnr"]=pyiqa.create_metric('psnr', test_y_channel=True, color_space='ycbcr').to(device)
    metric_paired_dict["ssim"]=pyiqa.create_metric('ssim', test_y_channel=True, color_space='ycbcr' ).to(device)
        
    lr_path_list = sorted([x for x in in_path.glob("*.[jpJP][pnPN]*[gG]")])
    if ntest is not None: lr_path_list = lr_path_list[:ntest]
    
    print(f'Find {len(lr_path_list)} images in {in_path}')
    result = {}
    for i in tqdm.tqdm(range(len(lr_path_list))):
        in_path = lr_path_list[i]
        ref_path = modify_path(in_path)

        im_in = util_image.imread(in_path, chn='rgb', dtype='float32')  # h x w x c
        im_in_tensor = util_image.img2tensor(im_in).cuda()              # 1 x c x h x w
        

        for key, metric in metric_dict.items():
            with torch.cuda.amp.autocast():
                result[key] = result.get(key, 0) + metric(im_in_tensor).item()
        
        if ref_path is not None:
            im_ref = util_image.imread(ref_path, chn='rgb', dtype='float32')  # h x w x c
            im_ref_tensor = util_image.img2tensor(im_ref).cuda()    
            # im_ref_tensor = F.interpolate(im_ref_tensor, size=(512, 512))
            im_in_tensor = F.interpolate(im_in_tensor, size=(im_ref_tensor.shape[-2], im_ref_tensor.shape[-1]))
            for key, metric in metric_paired_dict.items():
                result[key] = result.get(key, 0) + metric(im_in_tensor, im_ref_tensor).item()
                
    for key, res in result.items():
        print(f"{key}: {res/len(lr_path_list):.5f}")

    # save_output_to_file(in_path, lr_path_list, result, 'I-HAZE_metrics.txt')
    save_output_to_excel(in_path, lr_path_list, result, 'Dense-HAZE_three_metrics.xlsx')
        
if __name__ == "__main__":
    # res_name = 'indoor/dehazied_10_1_0.5/'
    # parser.add_argument('-i',"--in_path", type=str, default=f'/mnt/dantongwu/SOTS/{res_name}')
    root_path = '/home/uchihawdt/control-net-main-v0.5/output/few1' #'/mnt/dantongwu/Dense_Haze_NTIRE19'#'/mnt/dantongwu/NH-HAZE/' #'/mnt/dantongwu/O-HAZE/# O-HAZY NTIRE 2018/' #'/mnt/dantongwu/I-HAZE/# I-HAZY NTIRE 2018'
    in_paths = []
    for folder_name in os.listdir(root_path):
        folder_path = os.path.join(root_path, folder_name)
        if os.path.isdir(folder_path) and folder_name.count('_') == 6:
            in_paths.append(folder_path)
    for in_path in in_paths:
        parser = argparse.ArgumentParser()
        parser.add_argument('-i',"--in_path", type=str, default=in_path)
        parser.add_argument("-r", "--ref_path", type=str, default=None)
        parser.add_argument("--ntest", type=int, default=None)
        args = parser.parse_args()
        evaluate(args.in_path, args.ref_path, args.ntest)