import os
import torch

from omegaconf import OmegaConf
from ldm.util import instantiate_from_config


def get_state_dict(d):
    return d.get('state_dict', d)


def load_state_dict(ckpt_path, location='cpu'):
    _, extension = os.path.splitext(ckpt_path)
    if extension.lower() == ".safetensors":
        import safetensors.torch
        state_dict = safetensors.torch.load_file(ckpt_path, device=location)
    else:
        state_dict = get_state_dict(torch.load(ckpt_path, map_location=torch.device(location)))
    state_dict = get_state_dict(state_dict)
    print(f'Loaded state_dict from [{ckpt_path}]')
    return state_dict


def create_model(config_path, is_train=False, sub_names=None):
    config = OmegaConf.load(config_path)
    if is_train==True:
        config.model.params.control_stage_config.params['is_train']=True
    if sub_names!=None:
        config.model.params.control_stage_config.params['sub_names']=sub_names
    model = instantiate_from_config(config.model).cpu()
    print(f'Loaded model config from [{config_path}]')
    return model
