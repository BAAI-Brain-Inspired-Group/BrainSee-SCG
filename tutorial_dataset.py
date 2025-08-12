import json
import random
import numpy as np
import cv2
from cv2 import Canny
from pycocotools.coco import COCO

from torch.utils.data import Dataset
from annotator.util import HWC3
from annotator.hed import HEDdetector
from utils.data_augmentations import *

apply_hed = HEDdetector()


class MyDatasetCOCO(Dataset):
    def __init__(self, dirname=None, cond_ratio=0.5, scribble_ratio=0.4):
        self.data = []
        path = dirname+'/annotations/captions_train2017.json'
        self.dir_train = dirname+'/train2017'
        self.dir_val = dirname+'/val2017'
        self.coco = COCO(path)
        self.data = self.coco.loadImgs(self.coco.getImgIds())
        self.cond_ratio = cond_ratio
        self.scribble_ratio=scribble_ratio

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx, ):
        item = self.data[idx]

        target_filename = self.dir_train + '/' + item['file_name']
        target = cv2.imread(target_filename)
        prompt = self.coco.loadAnns(self.coco.getAnnIds(item['id']))[0]['caption'] if random.random() < self.cond_ratio else ''

        # Do not forget that OpenCV read images in BGR order.
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB).astype(np.float32)
        targ = cv2.resize(target,(512,512))
        cond = cv2.resize(target,(512,512))
        
        # Normalize target images to [-1, 1].
        targ = (targ / 127.5) - 1.0
        cond = cond / 255.

        return dict(jpg=targ, txt=prompt, hint=cond)

class MyDatasetCOCO_Scribble(MyDatasetCOCO):
    def __getitem__(self, idx, ):
        item = self.data[idx]
        # print(item)

        target_filename = self.dir_train + '/' + item['file_name']
        target = cv2.imread(target_filename)
        prompt = self.coco.loadAnns(self.coco.getAnnIds(item['id']))[0]['caption'] if random.random() < self.cond_ratio else ''

        # Do not forget that OpenCV read images in BGR order.
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB).astype(np.float32)
        targ = cv2.resize(target,(512,512))
        if random.random()<self.scribble_ratio:
            cond = apply_hed(target).astype(np.float32)
            cond = augment_edges(cond)
            cond = cv2.merge([cond, cond, cond])
            cond = cv2.resize(cond,(512,512))
        else:
            cond = cv2.resize(target,(512,512))

        # Normalize target images to [-1, 1].
        targ = (targ / 127.5) - 1.0
        cond = cond / 255.

        return dict(jpg=targ, txt=prompt, hint=cond)

class MyDatasetCOCO_val(MyDatasetCOCO):
    def __init__(self, dirname=None):
        path = dirname+'/annotations/captions_train2017.json'
        self.dir_train = dirname+'/val2017'
        self.coco = COCO(path)
        self.data = self.coco.loadImgs(self.coco.getImgIds())

class MyDatasetCOCO_canny(MyDatasetCOCO):
    def __getitem__(self,idx):
        item = self.data[idx]

        target_filename = self.dir_train + '/' + item['file_name']
        target = cv2.imread(target_filename)
        prompt = self.coco.loadAnns(self.coco.getAnnIds(item['id']))[0]['caption']

        # Do not forget that OpenCV read images in BGR order.
        detected_map = Canny(cv2.resize(target,(512,512)), 100, 200)
        cond = HWC3(detected_map).astype(np.float32)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB).astype(np.float32)
        targ = cv2.resize(target,(512,512))

        # Normalize target images to [-1, 1].
        targ = (targ / 127.5) - 1.0
        cond = cond / 255.

        return dict(jpg=targ, txt=prompt, hint=cond)
    
class MyDatasetTest(Dataset):
    def __init__(self, data_json):
        self.data = []
        with open(data_json, 'rt') as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        source_filename = item['source']
        target_filename = item['target']
        prompt = item['prompt']

        source = cv2.imread(source_filename)
        target = cv2.imread(target_filename)

        # Do not forget that OpenCV read images in BGR order.
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
        target = cv2.resize(target,(512,512))

        source = source.astype(np.float32) / 255.0

        targ = (target.astype(np.float32) / 127.5) - 1.0
        cond = target.astype(np.float32) / 255.0

        return dict(jpg=targ, txt=prompt, hint=cond)

class MyDatasetFewShot(Dataset):
    def __init__(self, data_json):
        self.data = []
        with open(data_json, 'rt') as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        cond_filename = item['source']
        target_filename = item['target']
        prompt = item['prompt']

        cond = cv2.imread(cond_filename)
        target = cv2.imread(target_filename)

        # Do not forget that OpenCV read images in BGR order.
        cond = cv2.cvtColor(cond, cv2.COLOR_BGR2RGB)
        cond = cv2.resize(cond,(512,512))
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
        target = cv2.resize(target,(512,512))

        # Normalize target images to [-1, 1].
        targ = (target.astype(np.float32) / 127.5) - 1.0
        cond = cond.astype(np.float32) / 255.0

        return dict(jpg=targ, txt=prompt, hint=cond)
    
class MyDataset_Hazy_FewShot(Dataset):
    def __init__(self):
        self.data = []
        with open('/home/uchihawdt/control-net-main-v0.5/data/train_1.json', 'rt') as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        cond_filename = item['source']
        target_filename = item['target']
        prompt = item['prompt']

        cond = cv2.imread(cond_filename)
        target = cv2.imread(target_filename)

        # Do not forget that OpenCV read images in BGR order.
        cond = cv2.cvtColor(cond, cv2.COLOR_BGR2RGB)
        cond = cv2.resize(cond,(512,512))
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
        target = cv2.resize(target,(512,512))

        # Normalize source images to [0, 1].

        # Normalize target images to [-1, 1].
        targ = (target.astype(np.float32) / 127.5) - 1.0
        cond = cond.astype(np.float32) / 255.0

        return dict(jpg=targ, txt=prompt, hint=cond)
    
class MyDataset_Light_FewShot(Dataset):
    def __init__(self):
        self.data = []
        with open('/home/uchihawdt/control-net-main-v0.5/data/low/train_1.json', 'rt') as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        cond_filename = item['source']
        target_filename = item['target']
        prompt = item['prompt']

        cond = cv2.imread(cond_filename)
        target = cv2.imread(target_filename)

        # Do not forget that OpenCV read images in BGR order.
        cond = cv2.cvtColor(cond, cv2.COLOR_BGR2RGB)
        cond = cv2.resize(cond,(512,512))
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
        target = cv2.resize(target,(512,512))

        # Normalize source images to [0, 1].

        # Normalize target images to [-1, 1].
        targ = (target.astype(np.float32) / 127.5) - 1.0
        cond = cond.astype(np.float32) / 255.0

        return dict(jpg=targ, txt=prompt, hint=cond)