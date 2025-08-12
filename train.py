from share import *

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from tutorial_dataset import *
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
from pytorch_lightning.callbacks import ModelCheckpoint
import os
import argparse

def get_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument("--modelarch_path", type=str, default=None, required=True, help="path of config")
    parser.add_argument("--checkpoint_path", type=str, default=None, required=True, help="path to store ckp")
    parser.add_argument("--logger_path", type=str, default=None, required=True, help="path to store result")
    # parser.add_argument("--dataset_path", type=str, required=True, help="path to store data")
    parser.add_argument("--resume_path", type=str, default=None, required=True, help="default path to store")
    parser.add_argument("--cond_ratio", type=int, default=0.5, required=False, help="default gpu to train")
    parser.add_argument("--dataset_name", type=str, default='MyDatasetCOCO')
    return parser

parser = get_parser()
args = parser.parse_args()

# Configs train
modelarch_path = args.modelarch_path
logger_path = args.logger_path
dataset_name = args.dataset_name
resume_path = args.resume_path
checkpoint_path = f'controlNet_ckpt/'+args.checkpoint_path+'/'

batch_size, logger_freq, learning_rate = 4, 3000, 1e-5
iftrain, sd_locked, only_mid_control = True, True, False

# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model(modelarch_path, is_train=True).cpu()
model.load_state_dict(load_state_dict(resume_path, location='cpu'), strict=False)
model.learning_rate = learning_rate
model.sd_locked = sd_locked

dataset = eval(f'{dataset_name}')()#(dirname={args.dataset_path})
dataloader = DataLoader(dataset, num_workers=0, batch_size=batch_size, shuffle=True)
logger = ImageLogger(batch_frequency=logger_freq,split=logger_path)
checkpoint_callback = ModelCheckpoint(dirpath=checkpoint_path, save_top_k=-1, save_last=True, save_weights_only=False, every_n_epochs=50,)

# Train!
trainer = pl.Trainer(strategy='ddp',accelerator='gpu',devices=[0], precision=32, callbacks=[logger, checkpoint_callback])
trainer.fit(model, dataloader)
