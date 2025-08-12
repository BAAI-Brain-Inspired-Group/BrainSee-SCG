from share import *

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
# os.environ['CUDA_VISIBLE_DEVICES'] = '3'
# torch.cuda.set_device(6)
testdata = []

configs = ['./models/cldm_v15_0_256_tune.yaml',]
ckpts = ['/home/uchihawdt/control-net-main-v0.5/multi_ckpt/epoch=10-step=325291.ckpt',]
with open('./data/testdata.json', 'rt') as f:
    for line in f:
        testdata.append(json.loads(line))


def get_input(xt):
    # encode prior 
    xt = np.array(xt.convert("RGB"))
    xt = xt[None].transpose(0, 3, 1, 2)
    xt = torch.from_numpy(xt).to(dtype=torch.float32) / 127.5 - 1.0
    xt = xt.float()
    return xt

@torch.no_grad()
def image2latent(image,model,device):
    with torch.no_grad():
        if type(image) is Image:
            image = np.array(image.convert("RGB"))
            # image = np.array(image).astype(np.float32) / 255.0
            image = image[None].transpose(0, 3, 1, 2)
            image = torch.from_numpy(image).to(device)
            image = torch.from_numpy(image).float() / 127.5 - 1
            latents = model.get_first_stage_encoding(model.encode_first_stage(image)).detach().float()
        if type(image) is torch.Tensor and image.dim() == 4:
            latents = image
        else:
            image = np.array(image.convert("RGB"))
            # image = np.array(image).astype(np.float32) / 255.0
            image = image[None].transpose(0, 3, 1, 2)
            image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0
            image = image.float().to(device)
            latents = model.get_first_stage_encoding(model.encode_first_stage(image)).detach().float()
    return latents

@torch.no_grad()
def ddim_loop(latent,sampler,model,context,num_samples,device):
    # uncond_embeddings, cond_embeddings = context.chunk(2)
    cond_embeddings = context
    all_latent = [latent]
    latent = latent.clone().detach()

    timesteps = sampler.ddim_timesteps
    time_range = timesteps
    total_steps = timesteps.shape[0]
    iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps)

    # for i in range(total_steps):
    for i, step in enumerate(iterator):
        index = total_steps - i - 1
        ts = torch.full((num_samples,), step, device=device, dtype=torch.long)
        _,_,noise_pred = sampler.p_sample_ddim(latent, cond_embeddings, ts, index=index)
        latent = next_step(noise_pred, ts, latent, model, device)
        all_latent.append(latent)
    return all_latent

def ddim_inversion(image,sampler,model,context,num_samples,device):
    latent = image2latent(image,model,device)
    # image_rec = latent2image(latent,model)
    ddim_latents = ddim_loop(latent,sampler,model,context,num_samples,device)
    # return image_rec, ddim_latents
    return ddim_latents

def next_step(model_output, timestep, sample, model, device):
    timestep, next_timestep = min(timestep - 1000 // NUM_DDIM_STEPS, 999), timestep
    alpha_prod_t = model.alphas_cumprod[timestep].to(device) if timestep >= 0 else model.alphas_cumprod[0].to(device)
    alpha_prod_t_next = model.alphas_cumprod[next_timestep].to(device)
    beta_prod_t = 1 - alpha_prod_t
    next_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
    next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
    next_sample = alpha_prod_t_next ** 0.5 * next_original_sample + next_sample_direction
    return next_sample

strength = 1.25
scale = 9.1

a_prompt = 'best quality, extremely detailed'
n_prompt = 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality'
num_samples = 1
image_resolution = 512
ddim_steps =50
seed, eta = 0, 0.0
NUM_DDIM_STEPS = 50
control_scale_sets = [[1,0.2], [1,0.5], [1,0.25], [0.8,0.2], [1,0.75], [1,1], [1,0], [0,1]]
output_dir = './output'

for k in range(0, 1): #len(ckpts)
    cur_config = configs[0]
    cur_ckpt = ckpts[k]

    model = create_model(cur_config).cpu()
    model.load_state_dict(load_state_dict(cur_ckpt, location='cuda'), strict=False)
    model = model.cuda()
    ddim_sampler = DDIMSampler(model)
    ddim_sampler.make_schedule(50)

    with torch.no_grad():

        for h in [5]:#range(len(control_scale_sets)):
            control_scale = control_scale_sets[h]

            for i in range(len(testdata)): #len(testdata)):#[1,32,8,10,12,14,16,23,28,36,41]:#range(len(testdata)):  #10,11): 
            # for i in range(0,100):
                for j in range(0,1):
                    for hyper_scale in [1]:
                        item = testdata[i]
                        img_path = item['source']
                        input_image = cv2.resize(cv2.imread(img_path),(512,512))
                        prompt = 'high-quality,extremely detailed,4K,HQ'  #item['prompt'] #if j==1 else ''

                        img = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
                        H, W, C = img.shape
                        control = torch.from_numpy(img.copy()).float().cuda() / 255.0
                        control = torch.stack([control for _ in range(num_samples)], dim=0)
                        control = einops.rearrange(control, 'b h w c -> b c h w').clone()

                        # seed = random.randint(0, 65535) if seed == -1 else seed
                        seed = 42+j
                        seed_everything(seed)

                        guess_mode = False #if j==1 or j==3 else False #True #False
                        cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([prompt] * num_samples)]}#[prompt + ', ' + a_prompt] * num_samples)]}
                        un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [model.get_learned_conditioning([""] * num_samples)]}#[n_prompt] * num_samples)]}
                        shape = (4, H // 8, W // 8)

                        model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([1.] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
                        
                        output_path = f'{output_dir}/{i}.png'   
                        # if not os.path.exists(output_path):
                        samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples,
                                                                        shape, cond, verbose=False, eta=eta,
                                                                        unconditional_guidance_scale=scale,
                                                                        unconditional_conditioning=un_cond,
                                                                        hyper_scale=hyper_scale, control_scale=control_scale, multi=False)

                        x_samples = model.decode_first_stage(samples)
                        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)
                        results = [x_samples[i] for i in range(num_samples)]

                        os.makedirs(os.path.dirname(output_path),exist_ok=True)
                        Image.fromarray(results[0]).save(output_path)
