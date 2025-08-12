from share import *
import argparse
import ast 

import cv2
import einops
import numpy as np
import torch
import random
import json
import os
from PIL import Image
import PIL
from tqdm import tqdm

from pytorch_lightning import seed_everything
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler


def get_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument("--cur_config", type=str, default=None, required=True, help="path of config")
    parser.add_argument("--cur_ckpt", type=str, default=None, required=True, help="path to store ckp")
    parser.add_argument("--output_dir", type=str, default=None, required=True, help="path to store result")
    parser.add_argument("--test_json_path", type=str, required=True, help="path to store data")
    # parser.add_argument("--control_scale", type=lambda s: ast.literal_eval(s), default=None, required=True, help="Scaling factors for control, e.g., [1, 0.25]")
    parser.add_argument("--hyper_scale", type=lambda s: ast.literal_eval(s), default=None, required=True)
    return parser

parser = get_parser()
args = parser.parse_args()

cur_config = args.cur_config
cur_ckpt = args.cur_ckpt
output_dir = args.output_dir
test_json_path = args.test_json_path
# control_scale = args.control_scale
hyper_scale = args.hyper_scale

testdata = []
with open(test_json_path, 'rt') as f:
    for line in f:
        testdata.append(json.loads(line))

strength = 1.25
scale = 9.1
a_prompt = 'best quality, extremely detailed'
n_prompt = 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality'
num_samples = 1
image_resolution = 512
ddim_steps =50
seed, eta = 0, 0.0

# model = create_model(cur_config, sub_names='1_10epoch').cpu()
model = create_model(cur_config).cpu()
model.load_state_dict(load_state_dict(cur_ckpt, location='cuda'), strict=False)
model = model.cuda()
ddim_sampler = DDIMSampler(model)
ddim_sampler.make_schedule(50)

with torch.no_grad():
    for i in range(len(testdata)):
                        
        item = testdata[i]
        img_path = item['source']
        input_image = cv2.resize(cv2.imread(img_path),(512,512))
        prompt = 'high-quality,extremely detailed,4K,HQ'

        img = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        H, W, C = img.shape
        control = torch.from_numpy(img.copy()).float().cuda() / 255.0
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()

        seed_everything(seed)

        guess_mode = False
        cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]}
        un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]}
        shape = (4, H // 8, W // 8)

        model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([1.] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
        
        # output_path = f'{output_dir}/{i}_{control_scale[0]}+{control_scale[1]}.png'   
        output_path = img_path.replace('/DehazeFormer/data/D-HAZY/test/hazy/', '/control-net-main-v0.5/output/dhaze/few0_0.75_0.5/')     #f'{output_dir}/{i}.png'   
        if not os.path.exists(output_path):
            samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples,
                                                            shape, cond, verbose=False, eta=eta,
                                                            unconditional_guidance_scale=scale,
                                                            unconditional_conditioning=un_cond,
                                                            hyper_scale=hyper_scale, control_scale=[0.75, 0.5], is_train=False, multi=True) # control_scale=control_scale, multi=True)

            x_samples = model.decode_first_stage(samples)
            x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)
            results = [x_samples[i] for i in range(num_samples)]

            os.makedirs(os.path.dirname(output_path),exist_ok=True)
            Image.fromarray(results[0]).save(output_path)
