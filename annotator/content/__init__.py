import cv2
from PIL import Image

import torch
from transformers import AutoProcessor, CLIPModel
import sys
import os
sys.path.append(os.path.abspath(directory))
from annotator.util import annotator_ckpts_path

import einops
import torch
import torch as th
import torch.nn as nn
import torchvision.transforms as transforms
import sys
import numpy as np
import random
from einops import rearrange, repeat,reduce
from torchvision import transforms
import os
import numpy as np
from hypercolumn.vit_pytorch.train_V1_sep_new import Column_trans_rot_lgn
from basicsr import tensor2img
from einops import rearrange, repeat
import argparse


class ContentDetector:
    def __init__(self):

        model_name = "/home/uchihawdt/control-net-main-v0.5/openai/clip-vit-large-patch14"

        self.model = CLIPModel.from_pretrained(model_name, cache_dir=annotator_ckpts_path).cuda().eval()
        self.processor = AutoProcessor.from_pretrained(model_name, cache_dir=annotator_ckpts_path)

    def __call__(self, img):
        assert img.ndim == 3
        with torch.no_grad():
            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            inputs = self.processor(images=img, return_tensors="pt").to(self.model.device)  #.to('cuda')
            image_features = self.model.get_image_features(**inputs)
            image_feature = image_features[0].detach().cpu().numpy()
        return image_feature

def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self

class HyperColumnLGN(nn.Module):
    def __init__(self,key=0,hypercond=[0],size=512,restore_ckpt = './hypercolumn/checkpoint/imagenet/equ_nv16_vl4_rn1_Bipolar_norm.pth'):
        super().__init__()
        ckpt = torch.load(restore_ckpt)
        hc = Column_trans_rot_lgn(ckpt['arg'])
        hc.load_state_dict(ckpt['state_dict'], strict=False)
        self.lgn_ende = hc.lgn_ende[0].eval()
        self.lgn_ende.train = disabled_train
        for param in self.lgn_ende.parameters():
            param.requires_grad = False

        self.resize = transforms.Resize(size)
        if size == 128:
            self.pad = nn.ConstantPad2d((1,1,1,1),0.)
        else:
            self.pad = nn.Identity()
        
        self.groups = [[0,1,4,8,9,15],[2,3],[5,12],[10],[6,7],[11,13,14]]
        self.p = [0. for i in range(len(self.groups))]
        self.p[0] = 1.

        norm_mean = np.array([0.50705882, 0.48666667, 0.44078431])
        norm_std = np.array([0.26745098, 0.25568627, 0.27607843])
        self.norm = transforms.Normalize(norm_mean, norm_std)
        self.cond = hypercond
        self.slct = None

    def forward(self,x):
        x = torch.tensor(x)
        x = x.unsqueeze(0)
        x = x.permute(0, 3, 1, 2)
        x = self.resize(x)
        s = x.size()
        r = torch.zeros(1,16,1,1).to(x.device)

        if self.cond is None:
            c = [i for i in range(len(self.groups))]
            random.shuffle(c)
            # print(self.groups[c[0]])
            pa = random.random()
            for i in range(len(self.groups)):
                if pa < self.p[i]:
                    for j in self.groups[c[i]]:
                        r[:,j,:,:] = 1.
        else:
            for i in self.cond:
                for j in self.groups[i]:
                    r[:,j,:,:] = 1

        r = repeat(r,'n c h w -> n (c repeat) h w',repeat=4)
        out = self.lgn_ende(self.norm(x))*r
        out = self.pad(self.lgn_ende.deconv(out))
        return out
    
    def deconv(self,x):
        x = self.resize(x)
        s = x.size()
        r = torch.zeros(1,16,1,1).to(x.device)
        if self.cond is not None:
            for i in self.cond:
                for j in self.groups[i]:
                    r[:,j,:,:] = 1

        r = repeat(r,'n c h w -> n (c repeat) h w',repeat=4)
        conv = self.lgn_ende(self.norm(x))*r
        deconv = self.lgn_ende.deconv(conv)
        deconv = (deconv - reduce(deconv,'b c h w -> b c () ()', 'min'))/(reduce(deconv,'b c h w -> b c () ()', 'max')-reduce(deconv,'b c h w -> b c () ()', 'min'))*2.-1.

        return deconv

detector = ContentDetector()
hyper = HyperColumnLGN()

def get_features(image_path, hypercond=None):
    img = cv2.imread(image_path)
    img = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32),(512,512))
    img = img / 255.
    if hypercond!=None:
        hyper.hypercond = hypercond
    img = hyper(img)
    img = tensor2img(img)   #(img.permute(0, 2, 3, 1).squeeze(0).numpy()*255).astype(np.uint8)
    feature = detector(img[0])
    return feature