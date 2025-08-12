# # import einops
# # import torch
# # import torch as th
# # import torch.nn as nn
# # import torchvision.transforms as transforms
# # import sys
# # import numpy as np
# # import random
# # from einops import rearrange, repeat,reduce
# # from torchvision import transforms

# # # sys.path.append('/home/chenzhiqiang/code/')
# # sys.path.append('/home/uchihawdt/control-net-main-v0.5')
# # from hypercolumn.vit_pytorch.train_V1_sep_new import Column_trans_rot_lgn

# # from ldm.modules.diffusionmodules.util import (
# #     conv_nd,
# #     linear,
# #     zero_module,
# #     timestep_embedding,
# # )

# # from einops import rearrange, repeat
# # from torchvision.utils import make_grid
# # from ldm.modules.attention import SpatialTransformer
# # from ldm.modules.attention import FeedForward
# # from ldm.modules.diffusionmodules.openaimodel import UNetModel, TimestepEmbedSequential, ResBlock, Downsample, AttentionBlock
# # from ldm.models.diffusion.ddpm import LatentDiffusion
# # from ldm.util import log_txt_as_img, exists, instantiate_from_config
# # from ldm.models.diffusion.ddim import DDIMSampler


# # class ControlledUnetModel(UNetModel):
# #     def forward(self, x, timesteps=None, context=None, control=None, only_mid_control=False, **kwargs):
# #         hs = []
# #         with torch.no_grad():
# #             t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
# #             emb = self.time_embed(t_emb)
# #             h = x.type(self.dtype)
# #             for module in self.input_blocks:
# #                 h = module(h, emb, context)
# #                 hs.append(h)
# #             h = self.middle_block(h, emb, context)

# #         if control is not None:
# #             h += control.pop()

# #         for i, module in enumerate(self.output_blocks):
# #             if only_mid_control or control is None:
# #                 h = torch.cat([h, hs.pop()], dim=1)
# #             else:
# #                 h = torch.cat([h, hs.pop() + control.pop()], dim=1)
# #             h = module(h, emb, context)

# #         h = h.type(x.dtype)
# #         return self.out(h)

# # def disabled_train(self, mode=True):
# #     """Overwrite model.train with this function to make sure train/eval mode
# #     does not change anymore."""
# #     return self


# # class HyperColumnLGN(nn.Module):
# #     def __init__(self,key=0,hypercond=[0],size=512,restore_ckpt = '/home/uchihawdt/control-net-main-v0.5/hypercolumn/checkpoint/imagenet/equ_nv16_vl4_rn1_Bipolar_norm.pth',groups = [[0,1,4,8,9,15],[2,3],[5,12],[10],[6,7],[11,13,14],[5],[12],[2],[3]]):
# #         super().__init__()
# #         ckpt = torch.load(restore_ckpt)
# #         hc = Column_trans_rot_lgn(ckpt['arg'])
# #         hc.load_state_dict(ckpt['state_dict'], strict=False)
# #         self.lgn_ende = hc.lgn_ende[0].eval()
# #         self.lgn_ende.train = disabled_train
# #         for param in self.lgn_ende.parameters():
# #             param.requires_grad = False

# #         self.resize = transforms.Resize(size)
# #         if size == 128:
# #             self.pad = nn.ConstantPad2d((1,1,1,1),0.)
# #         else:
# #             self.pad = nn.Identity()
        
# #         self.groups = groups
# #         self.p = [0. for i in range(len(self.groups))]
# #         self.p[0] = 1.

# #         norm_mean = np.array([0.50705882, 0.48666667, 0.44078431])
# #         norm_std = np.array([0.26745098, 0.25568627, 0.27607843])
# #         self.norm = transforms.Normalize(norm_mean, norm_std)
# #         self.cond = hypercond
# #         self.slct = None
    
# #     def forward(self,x, hyper_scale=1, cond=None):
# #         x = self.resize(x)
# #         s = x.size()
# #         r = torch.zeros(1,self.lgn_ende.n_vector,1,1).to(x.device)
# #         cond = cond if cond!=None else self.cond

# #         if cond is None:
# #             c = [i for i in range(len(self.groups))]
# #             random.shuffle(c)
# #             # print(self.groups[c[0]])
# #             pa = random.random()
# #             for i in range(len(self.groups)):
# #                 if pa < self.p[i]:
# #                     for j in self.groups[c[i]]:
# #                         r[:,j,:,:] = 1.
# #         else:
# #             # for i in cond:
# #             #     for j in self.groups[i]:
# #             #         r[:,j,:,:] = 1
# #             for j in self.groups[cond]:
# #                 r[:,j,:,:] = 1

# #         r = repeat(r,'n c h w -> n (c repeat) h w',repeat=self.lgn_ende.vector_length)
# #         out = self.lgn_ende(self.norm(x))*r
# #         out = self.pad(self.lgn_ende.deconv(out))
# #         # out *= hyper_scale
# #         return out
    
# #     def deconv(self,x):
# #         x = self.resize(x)
# #         s = x.size()
# #         r = torch.zeros(1,self.lgn_ende.n_vector,1,1).to(x.device)
# #         if self.cond is not None:
# #             for i in self.cond:
# #                 for j in self.groups[i]:
# #                     r[:,j,:,:] = 1

# #         r = repeat(r,'n c h w -> n (c repeat) h w',repeat=self.lgn_ende.vector_length)
# #         conv = self.lgn_ende(self.norm(x))*r
# #         deconv = self.lgn_ende.deconv(conv)
# #         deconv = (deconv - reduce(deconv,'b c h w -> b c () ()', 'min'))/(reduce(deconv,'b c h w -> b c () ()', 'max')-reduce(deconv,'b c h w -> b c () ()', 'min'))*2.-1.

# #         return deconv

# # class HyperColumnLGNVisual(nn.Module):
# #     def __init__(self,key=0,hypercond=[0],size=512,restore_ckpt = '/home/uchihawdt/control-net-main-v0.5/hypercolumn/checkpoint/imagenet/equ_nv16_vl4_rn1_Bipolar_norm.pth'):
# #         super().__init__()
# #         ckpt = torch.load(restore_ckpt)
# #         hc = Column_trans_rot_lgn(ckpt['arg'])
# #         hc.load_state_dict(ckpt['state_dict'], strict=False)
# #         self.lgn_ende = hc.lgn_ende[0].eval()
# #         self.lgn_ende.train = disabled_train
# #         for param in self.lgn_ende.parameters():
# #             param.requires_grad = False

# #         self.resize = transforms.Resize(size)
# #         if size == 128:
# #             self.pad = nn.ConstantPad2d((1,1,1,1),0.)
# #         else:
# #             self.pad = nn.Identity()
        

# #         # self.groups = [[2,3],[0,1,4,8,9,15],[5,6,7,10,12],[11,13,14]]
# #         self.groups = [[0,1,4,8,9,15],[2,3],[5,12],[10],[6,7],[11,13,14],[5],[12],[2],[3]]
# #         # self.groups = [[2],[3],[5],[12]]
            
# #         # self.p = [1.,0.5,0.25,0.125]
# #         self.p = [0. for i in range(len(self.groups))]
# #         self.p[0] = 1.

# #         norm_mean = np.array([0.50705882, 0.48666667, 0.44078431])
# #         norm_std = np.array([0.26745098, 0.25568627, 0.27607843])
# #         self.norm = transforms.Normalize(norm_mean, norm_std)
# #         self.cond = hypercond
# #         self.slct = None

# #     # def forward(self,x):
# #     #     s = x.size()
# #     #     r = torch.zeros(1,16,1,1).to(x.device)

# #     #     if self.cond is None:
# #     #         c = [i for i in range(len(self.groups))]
# #     #         random.shuffle(c)
# #     #         # print(self.groups[c[0]])
# #     #         pa = random.random()
# #     #         for i in range(4):
# #     #             if pa < self.p[i]:
# #     #                 for j in self.groups[c[i]]:
# #     #                     r[:,j,:,:] = 1.
# #     #     else:
# #     #         for i in self.cond:
# #     #             for j in self.groups[i]:
# #     #                 r[:,j,:,:] = 1

# #     #     r = repeat(r,'n c h w -> n (c repeat) h w',repeat=4)
# #     #     return self.lgn_ende(self.norm(x))*r
    
# #     def forward(self,x):
# #         x = self.resize(x)
# #         s = x.size()
# #         r = torch.zeros(1,16,1,1).to(x.device)

# #         if self.cond is None:
# #             c = [i for i in range(len(self.groups))]
# #             random.shuffle(c)
# #             # print(self.groups[c[0]])
# #             pa = random.random()
# #             for i in range(len(self.groups)):
# #                 if pa < self.p[i]:
# #                     for j in self.groups[c[i]]:
# #                         r[:,j,:,:] = 1.
# #         else:
# #             for i in self.cond:
# #                 for j in self.groups[i]:
# #                     r[:,j,:,:] = 1

# #         # r = repeat(r,'n c h w -> n (c repeat) h w',repeat=4)
# #         # out = self.lgn_ende(self.norm(x))*r
# #         # # print('out_conv:',out.size())
# #         # out = self.pad(self.lgn_ende.deconv(out))
# #         # # out = self.lgn_ende.deconv(self.lgn_ende(self.norm(x))*r)
# #         # # print('out:',out.size())
# #         # return out

# #         out = self.lgn_ende(self.norm(x))
# #         # out = rearrange(out, 'b (c n) h w -> b c (n h) w',n=16)
# #         out = rearrange(out, 'b (n c) h w -> b c (n h) w', n=16)
# #         # out = self.pad(self.lgn_ende.deconv(out))
# #         return out
    
# #     def deconv(self,x):
# #         x = self.resize(x)
# #         s = x.size()
# #         r = torch.zeros(1,16,1,1).to(x.device)
# #         if self.cond is not None:
# #             for i in self.cond:
# #                 for j in self.groups[i]:
# #                     r[:,j,:,:] = 1

# #         r = repeat(r,'n c h w -> n (c repeat) h w',repeat=4)
# #         conv = self.lgn_ende(self.norm(x))*r
# #         deconv = self.lgn_ende.deconv(conv)
# #         deconv = (deconv - reduce(deconv,'b c h w -> b c () ()', 'min'))/(reduce(deconv,'b c h w -> b c () ()', 'max')-reduce(deconv,'b c h w -> b c () ()', 'min'))*2.-1.

# #         return deconv
    
# # class HyperColumnLGNFeature(nn.Module):
# #     def __init__(self,key=0,hypercond=[0],size=512,restore_ckpt = '/home/uchihawdt/control-net-main-v0.5/hypercolumn/checkpoint/imagenet/equ_nv16_vl4_rn1_Bipolar_norm.pth'):
# #         super().__init__()
# #         ckpt = torch.load(restore_ckpt)
# #         hc = Column_trans_rot_lgn(ckpt['arg'])
# #         hc.load_state_dict(ckpt['state_dict'], strict=False)
# #         self.lgn_ende = hc.lgn_ende[0].eval()
# #         self.lgn_ende.train = disabled_train
# #         for param in self.lgn_ende.parameters():
# #             param.requires_grad = False

# #         self.resize = transforms.Resize(size)
# #         if size == 128:
# #             self.pad = nn.ConstantPad2d((1,1,1,1),0.)
# #         else:
# #             self.pad = nn.Identity()
        

# #         # self.groups = [[2,3],[0,1,4,8,9,15],[5,6,7,10,12],[11,13,14]]
# #         self.groups = [[0,1,4,8,9,15],[2,3],[5,12],[10],[6,7],[11,13,14],[5],[12],[2],[3]]
# #         # self.groups = [[2],[3],[5],[12]]
            
# #         # self.p = [1.,0.5,0.25,0.125]
# #         self.p = [0. for i in range(len(self.groups))]
# #         self.p[0] = 1.

# #         norm_mean = np.array([0.50705882, 0.48666667, 0.44078431])
# #         norm_std = np.array([0.26745098, 0.25568627, 0.27607843])
# #         self.norm = transforms.Normalize(norm_mean, norm_std)
# #         self.cond = hypercond
# #         self.slct = None
    
# #     def forward(self,x):
# #         x = self.resize(x)
# #         s = x.size()
# #         # r = torch.zeros(1,16,1,1).to(x.device)

# #         # if self.cond is None:
# #         #     c = [i for i in range(len(self.groups))]
# #         #     random.shuffle(c)
# #         #     # print(self.groups[c[0]])
# #         #     pa = random.random()
# #         #     for i in range(len(self.groups)):
# #         #         if pa < self.p[i]:
# #         #             for j in self.groups[c[i]]:
# #         #                 r[:,j,:,:] = 1.
# #         # else:
# #         #     for i in self.cond:
# #         #         for j in self.groups[i]:
# #         #             r[:,j,:,:] = 1

# #         # r = repeat(r,'n c h w -> n (c repeat) h w',repeat=4)
# #         out = self.lgn_ende(self.norm(x))
# #         out = out[:,40:44,:,:]
# #         # out = self.pad(self.lgn_ende.deconv(out))
# #         return out

# #     def deconv(self,x):
# #         x = self.resize(x)
# #         s = x.size()
# #         r = torch.zeros(1,16,1,1).to(x.device)
# #         if self.cond is not None:
# #             for i in self.cond:
# #                 for j in self.groups[i]:
# #                     r[:,j,:,:] = 1

# #         r = repeat(r,'n c h w -> n (c repeat) h w',repeat=4)
# #         conv = self.lgn_ende(self.norm(x))*r
# #         deconv = self.lgn_ende.deconv(conv)
# #         deconv = (deconv - reduce(deconv,'b c h w -> b c () ()', 'min'))/(reduce(deconv,'b c h w -> b c () ()', 'max')-reduce(deconv,'b c h w -> b c () ()', 'min'))*2.-1.

# #         return deconv


# # class SemanticAdapter(nn.Module):
# #     def __init__(self, in_dim, channel_mult=[2, 4]):
# #         super().__init__()
# #         dim_out1, mult1 = in_dim*channel_mult[0], channel_mult[0]*2
# #         dim_out2, mult2 = in_dim*channel_mult[1], channel_mult[1]*2//channel_mult[0]
# #         self.in_dim = in_dim
# #         self.channel_mult = channel_mult
        
# #         self.ff1 = FeedForward(in_dim, dim_out=dim_out1, mult=mult1, glu=True, dropout=0.1)
# #         self.ff2 = FeedForward(dim_out1, dim_out=dim_out2, mult=mult2, glu=True, dropout=0.3)
# #         self.norm1 = nn.LayerNorm(in_dim)
# #         self.norm2 = nn.LayerNorm(dim_out1)

# #     def forward(self, x):
# #         x = self.ff1(self.norm1(x))
# #         x = self.ff2(self.norm2(x))
# #         x = rearrange(x, 'b (n d) -> b n d', n=self.channel_mult[-1], d=self.in_dim).contiguous()
# #         return x


# # class Canny(nn.Module):
# #     def __init__(self,para=None):
# #         super().__init__()
# #     def forward(self,x):
# #         return x
# #     def deconv(self,x):
# #         return x


# # class ConditionEmbedding(nn.Module):
# #     def __init__(self, num_conditions, embedding_dim):
# #         super(ConditionEmbedding, self).__init__()
# #         self.embedding = nn.Embedding(num_conditions, embedding_dim)

# #     def forward(self, condition_type):
# #         return self.embedding(condition_type)

# # class ControlSingle(nn.Module):
# #     def __init__(self,image_size, in_channels, model_channels, hint_channels, num_res_blocks, attention_resolutions, dropout=0, channel_mult=(1, 2, 4, 8), conv_resample=True, dims=2, use_checkpoint=False, use_fp16=False, num_heads=-1, num_head_channels=-1, num_heads_upsample=-1, use_scale_shift_norm=False, resblock_updown=False, use_new_attention_order=False, use_spatial_transformer=False, transformer_depth=1, context_dim=None, n_embed=None, legacy=True, disable_self_attentions=None, num_attention_blocks=None, disable_middle_self_attn=False, use_linear_in_transformer=False, hyperconfig=None, stride=[2,2,2], num_conditions=6, embedding_dim=128,hypercond=0,size=None,restore_ckpt=None,groups=None):
# #         super().__init__()
# #         self.dims = dims
# #         hyperconfig['params']['hypercond'] = [hypercond]
# #         if size and restore_ckpt and groups:
# #             hyperconfig['params']['size']=size #256
# #             hyperconfig['params']['restore_ckpt']=restore_ckpt  #'/home/chenzhiqiang/code/ControlNet-main/hypercolumn/checkpoint/imagenet/equ_nv128_vl1_rn8_Vanilla_ks17_norm.pth'
# #             hyperconfig['params']['groups']=groups  #: [[0],[1],[2],[3],[4],[5],[6],[7],[8],[9],[10],[11],[12],[13],[14],[15]]
# #         self.hypercolumn = instantiate_from_config(hyperconfig)
# #         self.input_blocks = nn.ModuleList(
# #             [
# #                 TimestepEmbedSequential(
# #                     conv_nd(dims, in_channels, model_channels, 3, padding=1)
# #                 )
# #             ]
# #         )
# #         self.zero_convs = nn.ModuleList([self.make_zero_conv(model_channels)])

# #         if isinstance(num_res_blocks, int):
# #             self.num_res_blocks = len(channel_mult) * [num_res_blocks]
# #         else:
# #             if len(num_res_blocks) != len(channel_mult):
# #                 raise ValueError("provide num_res_blocks either as an int (globally constant) or "
# #                                  "as a list/tuple (per-level) with the same length as channel_mult")
# #             self.num_res_blocks = num_res_blocks
        
# #         time_embed_dim = model_channels * 4
# #         self.input_hint_block_new = TimestepEmbedSequential(self.hypercolumn,
# #                                                             conv_nd(dims, hint_channels, 16, 3, padding=1),nn.SiLU(),
# #                                                             conv_nd(dims, 16, 16, 3, padding=1),nn.SiLU(),
# #                                                             conv_nd(dims, 16, 32, 3, padding=1, stride=stride[0]),nn.SiLU(),
# #                                                             conv_nd(dims, 32, 32, 3, padding=1),nn.SiLU(),
# #                                                             conv_nd(dims, 32, 96, 3, padding=1, stride=stride[1]),nn.SiLU(),
# #                                                             conv_nd(dims, 96, 96, 3, padding=1),nn.SiLU(),
# #                                                             conv_nd(dims, 96, 256, 3, padding=1, stride=stride[2]),nn.SiLU(),
# #                                                             zero_module(conv_nd(dims, 256, model_channels, 3, padding=1)))
# #         self._feature_size = model_channels
# #         input_block_chans = [model_channels]
# #         ch = model_channels
# #         ds = 1
# #         for level, mult in enumerate(channel_mult):
# #             for nr in range(self.num_res_blocks[level]):
# #                 layers = [
# #                     ResBlock(
# #                         ch,
# #                         time_embed_dim,
# #                         dropout,
# #                         out_channels=mult * model_channels,
# #                         dims=dims,
# #                         use_checkpoint=use_checkpoint,
# #                         use_scale_shift_norm=use_scale_shift_norm,
# #                     )
# #                 ]
# #                 ch = mult * model_channels
# #                 if ds in attention_resolutions:
# #                     if num_head_channels == -1:
# #                         dim_head = ch // num_heads
# #                     else:
# #                         num_heads = ch // num_head_channels
# #                         dim_head = num_head_channels
# #                     if legacy:
# #                         # num_heads = 1
# #                         dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
# #                     if exists(disable_self_attentions):
# #                         disabled_sa = disable_self_attentions[level]
# #                     else:
# #                         disabled_sa = False

# #                     if not exists(num_attention_blocks) or nr < num_attention_blocks[level]:
# #                         layers.append(
# #                             AttentionBlock(
# #                                 ch,
# #                                 use_checkpoint=use_checkpoint,
# #                                 num_heads=num_heads,
# #                                 num_head_channels=dim_head,
# #                                 use_new_attention_order=use_new_attention_order,
# #                             ) if not use_spatial_transformer else SpatialTransformer(
# #                                 ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
# #                                 disable_self_attn=disabled_sa, use_linear=use_linear_in_transformer,
# #                                 use_checkpoint=use_checkpoint
# #                             )
# #                         )
# #                 self.input_blocks.append(TimestepEmbedSequential(*layers))
# #                 self.zero_convs.append(self.make_zero_conv(ch))
# #                 self._feature_size += ch
# #                 input_block_chans.append(ch)
# #             if level != len(channel_mult) - 1:
# #                 out_ch = ch
# #                 self.input_blocks.append(
# #                     TimestepEmbedSequential(
# #                         ResBlock(
# #                             ch,
# #                             time_embed_dim,
# #                             dropout,
# #                             out_channels=out_ch,
# #                             dims=dims,
# #                             use_checkpoint=use_checkpoint,
# #                             use_scale_shift_norm=use_scale_shift_norm,
# #                             down=True,
# #                         )
# #                         if resblock_updown
# #                         else Downsample(
# #                             ch, conv_resample, dims=dims, out_channels=out_ch
# #                         )
# #                     )
# #                 )
# #                 ch = out_ch
# #                 input_block_chans.append(ch)
# #                 self.zero_convs.append(self.make_zero_conv(ch))
# #                 ds *= 2
# #                 self._feature_size += ch

# #         if num_head_channels == -1:
# #             dim_head = ch // num_heads
# #         else:
# #             num_heads = ch // num_head_channels
# #             dim_head = num_head_channels
# #         if legacy:
# #             # num_heads = 1
# #             dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
# #         self.middle_block = TimestepEmbedSequential(
# #             ResBlock(
# #                 ch,
# #                 time_embed_dim,
# #                 dropout,
# #                 dims=dims,
# #                 use_checkpoint=use_checkpoint,
# #                 use_scale_shift_norm=use_scale_shift_norm,
# #             ),
# #             AttentionBlock(
# #                 ch,
# #                 use_checkpoint=use_checkpoint,
# #                 num_heads=num_heads,
# #                 num_head_channels=dim_head,
# #                 use_new_attention_order=use_new_attention_order,
# #             ) if not use_spatial_transformer else SpatialTransformer(  # always uses a self-attn
# #                 ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
# #                 disable_self_attn=disable_middle_self_attn, use_linear=use_linear_in_transformer,
# #                 use_checkpoint=use_checkpoint
# #             ),
# #             ResBlock(
# #                 ch,
# #                 time_embed_dim,
# #                 dropout,
# #                 dims=dims,
# #                 use_checkpoint=use_checkpoint,
# #                 use_scale_shift_norm=use_scale_shift_norm,
# #             ),
# #         )
# #         self.middle_block_out = self.make_zero_conv(ch)
# #     def make_zero_conv(self, channels):
# #         return TimestepEmbedSequential(zero_module(conv_nd(self.dims, channels, channels, 1, padding=0)))

# # class ControlNet(nn.Module):
# #     def __init__(
# #             self,
# #             image_size,
# #             in_channels,
# #             model_channels,
# #             hint_channels,
# #             num_res_blocks,
# #             attention_resolutions,
# #             dropout=0,
# #             channel_mult=(1, 2, 4, 8),
# #             conv_resample=True,
# #             dims=2,
# #             use_checkpoint=False,
# #             use_fp16=False,
# #             num_heads=-1,
# #             num_head_channels=-1,
# #             num_heads_upsample=-1,
# #             use_scale_shift_norm=False,
# #             resblock_updown=False,
# #             use_new_attention_order=False,
# #             use_spatial_transformer=False,  # custom transformer support
# #             transformer_depth=1,  # custom transformer support
# #             context_dim=None,  # custom transformer support
# #             n_embed=None,  # custom support for prediction of discrete ids into codebook of first stage vq model
# #             legacy=True,
# #             disable_self_attentions=None,
# #             num_attention_blocks=None,
# #             disable_middle_self_attn=False,
# #             use_linear_in_transformer=False,
# #             hyperconfig=None,
# #             stride=[2,2,2],
# #             num_conditions=6, 
# #             embedding_dim=128,
# #             is_train=False,
# #             epoch_num=None,
# #     ):
# #         super().__init__()
# #         if use_spatial_transformer:
# #             assert context_dim is not None, 'Fool!! You forgot to include the dimension of your cross-attention conditioning...'

# #         if context_dim is not None:
# #             assert use_spatial_transformer, 'Fool!! You forgot to use the spatial transformer for your cross-attention conditioning...'
# #             from omegaconf.listconfig import ListConfig
# #             if type(context_dim) == ListConfig:
# #                 context_dim = list(context_dim)

# #         if num_heads_upsample == -1:
# #             num_heads_upsample = num_heads

# #         if num_heads == -1:
# #             assert num_head_channels != -1, 'Either num_heads or num_head_channels has to be set'

# #         if num_head_channels == -1:
# #             assert num_heads != -1, 'Either num_heads or num_head_channels has to be set'

# #         self.dims = dims
# #         self.image_size = image_size
# #         self.in_channels = in_channels
# #         self.model_channels = model_channels
# #         if isinstance(num_res_blocks, int):
# #             self.num_res_blocks = len(channel_mult) * [num_res_blocks]
# #         else:
# #             if len(num_res_blocks) != len(channel_mult):
# #                 raise ValueError("provide num_res_blocks either as an int (globally constant) or "
# #                                  "as a list/tuple (per-level) with the same length as channel_mult")
# #             self.num_res_blocks = num_res_blocks
# #         if disable_self_attentions is not None:
# #             # should be a list of booleans, indicating whether to disable self-attention in TransformerBlocks or not
# #             assert len(disable_self_attentions) == len(channel_mult)
# #         if num_attention_blocks is not None:
# #             assert len(num_attention_blocks) == len(self.num_res_blocks)
# #             assert all(map(lambda i: self.num_res_blocks[i] >= num_attention_blocks[i], range(len(num_attention_blocks))))
# #             print(f"Constructor of UNetModel received num_attention_blocks={num_attention_blocks}. "
# #                   f"This option has LESS priority than attention_resolutions {attention_resolutions}, "
# #                   f"i.e., in cases where num_attention_blocks[i] > 0 but 2**i not in attention_resolutions, "
# #                   f"attention will still not be set.")

# #         self.attention_resolutions = attention_resolutions
# #         self.dropout = dropout
# #         self.channel_mult = channel_mult
# #         self.conv_resample = conv_resample
# #         self.use_checkpoint = use_checkpoint
# #         self.dtype = th.float16 if use_fp16 else th.float32
# #         self.num_heads = num_heads
# #         self.num_head_channels = num_head_channels
# #         self.num_heads_upsample = num_heads_upsample
# #         self.predict_codebook_ids = n_embed is not None
# #         self.hyperconfig = hyperconfig
# #         self.hypercolumn = instantiate_from_config(hyperconfig)

# #         time_embed_dim = model_channels * 4
# #         self.time_embed = nn.Sequential(
# #             linear(model_channels, time_embed_dim),
# #             nn.SiLU(),
# #             linear(time_embed_dim, time_embed_dim),
# #         )
# #         self.group_embedding = ConditionEmbedding(num_conditions, embedding_dim)

# #         self.input_blocks = nn.ModuleList(
# #             [
# #                 TimestepEmbedSequential(
# #                     conv_nd(dims, in_channels, model_channels, 3, padding=1)
# #                 )
# #             ]
# #         )
# #         self.zero_convs = nn.ModuleList([self.make_zero_conv(model_channels)])

# #         self.input_hint_block_new = TimestepEmbedSequential(
# #             # HyperColumnLGN(hypercond=hypercond),
# #             # Canny(),
# #             self.hypercolumn,
# #             conv_nd(dims, hint_channels, 16, 3, padding=1),
# #             nn.SiLU(),
# #             conv_nd(dims, 16, 16, 3, padding=1),
# #             nn.SiLU(),
# #             conv_nd(dims, 16, 32, 3, padding=1, stride=stride[0]),
# #             nn.SiLU(),
# #             conv_nd(dims, 32, 32, 3, padding=1),
# #             nn.SiLU(),
# #             conv_nd(dims, 32, 96, 3, padding=1, stride=stride[1]),
# #             nn.SiLU(),
# #             conv_nd(dims, 96, 96, 3, padding=1),
# #             nn.SiLU(),
# #             conv_nd(dims, 96, 256, 3, padding=1, stride=stride[2]),
# #             nn.SiLU(),
# #             zero_module(conv_nd(dims, 256, model_channels, 3, padding=1))
# #         )

# #         self._feature_size = model_channels
# #         input_block_chans = [model_channels]
# #         ch = model_channels
# #         ds = 1
# #         for level, mult in enumerate(channel_mult):
# #             for nr in range(self.num_res_blocks[level]):
# #                 layers = [
# #                     ResBlock(
# #                         ch,
# #                         time_embed_dim,
# #                         dropout,
# #                         out_channels=mult * model_channels,
# #                         dims=dims,
# #                         use_checkpoint=use_checkpoint,
# #                         use_scale_shift_norm=use_scale_shift_norm,
# #                     )
# #                 ]
# #                 ch = mult * model_channels
# #                 if ds in attention_resolutions:
# #                     if num_head_channels == -1:
# #                         dim_head = ch // num_heads
# #                     else:
# #                         num_heads = ch // num_head_channels
# #                         dim_head = num_head_channels
# #                     if legacy:
# #                         # num_heads = 1
# #                         dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
# #                     if exists(disable_self_attentions):
# #                         disabled_sa = disable_self_attentions[level]
# #                     else:
# #                         disabled_sa = False

# #                     if not exists(num_attention_blocks) or nr < num_attention_blocks[level]:
# #                         layers.append(
# #                             AttentionBlock(
# #                                 ch,
# #                                 use_checkpoint=use_checkpoint,
# #                                 num_heads=num_heads,
# #                                 num_head_channels=dim_head,
# #                                 use_new_attention_order=use_new_attention_order,
# #                             ) if not use_spatial_transformer else SpatialTransformer(
# #                                 ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
# #                                 disable_self_attn=disabled_sa, use_linear=use_linear_in_transformer,
# #                                 use_checkpoint=use_checkpoint
# #                             )
# #                         )
# #                 self.input_blocks.append(TimestepEmbedSequential(*layers))
# #                 self.zero_convs.append(self.make_zero_conv(ch))
# #                 self._feature_size += ch
# #                 input_block_chans.append(ch)
# #             if level != len(channel_mult) - 1:
# #                 out_ch = ch
# #                 self.input_blocks.append(
# #                     TimestepEmbedSequential(
# #                         ResBlock(
# #                             ch,
# #                             time_embed_dim,
# #                             dropout,
# #                             out_channels=out_ch,
# #                             dims=dims,
# #                             use_checkpoint=use_checkpoint,
# #                             use_scale_shift_norm=use_scale_shift_norm,
# #                             down=True,
# #                         )
# #                         if resblock_updown
# #                         else Downsample(
# #                             ch, conv_resample, dims=dims, out_channels=out_ch
# #                         )
# #                     )
# #                 )
# #                 ch = out_ch
# #                 input_block_chans.append(ch)
# #                 self.zero_convs.append(self.make_zero_conv(ch))
# #                 ds *= 2
# #                 self._feature_size += ch

# #         if num_head_channels == -1:
# #             dim_head = ch // num_heads
# #         else:
# #             num_heads = ch // num_head_channels
# #             dim_head = num_head_channels
# #         if legacy:
# #             # num_heads = 1
# #             dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
# #         self.middle_block = TimestepEmbedSequential(
# #             ResBlock(
# #                 ch,
# #                 time_embed_dim,
# #                 dropout,
# #                 dims=dims,
# #                 use_checkpoint=use_checkpoint,
# #                 use_scale_shift_norm=use_scale_shift_norm,
# #             ),
# #             AttentionBlock(
# #                 ch,
# #                 use_checkpoint=use_checkpoint,
# #                 num_heads=num_heads,
# #                 num_head_channels=dim_head,
# #                 use_new_attention_order=use_new_attention_order,
# #             ) if not use_spatial_transformer else SpatialTransformer(  # always uses a self-attn
# #                 ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
# #                 disable_self_attn=disable_middle_self_attn, use_linear=use_linear_in_transformer,
# #                 use_checkpoint=use_checkpoint
# #             ),
# #             ResBlock(
# #                 ch,
# #                 time_embed_dim,
# #                 dropout,
# #                 dims=dims,
# #                 use_checkpoint=use_checkpoint,
# #                 use_scale_shift_norm=use_scale_shift_norm,
# #             ),
# #         )
# #         self.middle_block_out = self.make_zero_conv(ch)
# #         self._feature_size += ch
# #         # control_net_args = {'image_size': image_size, 'in_channels': in_channels, 'model_channels': model_channels, 'hint_channels': hint_channels, 'num_res_blocks': num_res_blocks, 'attention_resolutions': attention_resolutions,
# #         #                     'dropout': dropout, 'channel_mult': channel_mult, 'conv_resample': conv_resample, 'dims': dims, 'use_checkpoint': use_checkpoint, 'use_fp16': use_fp16, 'num_heads': num_heads, 'num_head_channels': num_head_channels,
# #         #                     'num_heads_upsample': num_heads_upsample, 'use_scale_shift_norm': use_scale_shift_norm,'resblock_updown': resblock_updown,'use_new_attention_order': use_new_attention_order, 'use_spatial_transformer': use_spatial_transformer,
# #         #                     'transformer_depth': transformer_depth,'context_dim': context_dim,'n_embed': n_embed,'legacy': legacy,'disable_self_attentions': None,'num_attention_blocks': num_attention_blocks,'disable_middle_self_attn': disable_middle_self_attn,
# #         #                     'use_linear_in_transformer': use_linear_in_transformer,'hyperconfig': hyperconfig,'stride': stride,'num_conditions': num_conditions,'embedding_dim': embedding_dim,}
# #         # self.new_controlNet = ControlSingle(self.image_size, self.in_channels, self.model_channels, hint_channels, self.num_res_blocks, self.attention_resolutions, hyperconfig=hyperconfig, dims = self.dims, channel_mult=self.channel_mult,
# #         #                                     num_heads = self.num_heads, num_heads_upsample=self.num_heads_upsample, stride=stride, legacy=legacy)
# #         if is_train==False and epoch_num:
# #             import omegaconf
# #             if isinstance(epoch_num,omegaconf.listconfig.ListConfig):
# #                 # for epoch_num_single in epoch_num_single: 
# #                 cur_hypercond=int(epoch_num[0].split('_')[1])
# #                 self.new_controlNet_0 = ControlSingle(image_size=image_size, in_channels=in_channels, model_channels=model_channels, hint_channels=hint_channels, num_res_blocks=num_res_blocks, attention_resolutions=attention_resolutions,
# #                                                 dropout=dropout, channel_mult=channel_mult, conv_resample=conv_resample, dims=dims, use_checkpoint=use_checkpoint, use_fp16=use_fp16, num_heads=num_heads, num_head_channels=num_head_channels,
# #                                                 num_heads_upsample=num_heads_upsample, use_scale_shift_norm=use_scale_shift_norm,resblock_updown=resblock_updown,use_new_attention_order=use_new_attention_order, use_spatial_transformer=use_spatial_transformer,
# #                                                 transformer_depth=transformer_depth,context_dim=context_dim,n_embed=n_embed,legacy=legacy,disable_self_attentions=None,num_attention_blocks=num_attention_blocks,disable_middle_self_attn=disable_middle_self_attn,
# #                                                 use_linear_in_transformer=use_linear_in_transformer,hyperconfig=hyperconfig,stride=stride,num_conditions=num_conditions,embedding_dim=embedding_dim,hypercond=cur_hypercond)        
# #                 self.load_model_parts(self.new_controlNet_0, f'multi_ckpt/{epoch_num[0]}.pt')
# #                 cur_hypercond=int(epoch_num[1].split('_')[1])
# #                 stride, size, restore_ckpt, groups = [1,2,2], 256, '/home/chenzhiqiang/code/ControlNet-main/hypercolumn/checkpoint/imagenet/equ_nv128_vl1_rn8_Vanilla_ks17_norm.pth', [[0],[1],[2],[3],[4],[5],[6],[7],[8],[9],[10],[11],[12],[13],[14],[15]]
# #                 self.new_controlNet_1 = ControlSingle(image_size=image_size, in_channels=in_channels, model_channels=model_channels, hint_channels=hint_channels, num_res_blocks=num_res_blocks, attention_resolutions=attention_resolutions,
# #                                                 dropout=dropout, channel_mult=channel_mult, conv_resample=conv_resample, dims=dims, use_checkpoint=use_checkpoint, use_fp16=use_fp16, num_heads=num_heads, num_head_channels=num_head_channels,
# #                                                 num_heads_upsample=num_heads_upsample, use_scale_shift_norm=use_scale_shift_norm,resblock_updown=resblock_updown,use_new_attention_order=use_new_attention_order, use_spatial_transformer=use_spatial_transformer,
# #                                                 transformer_depth=transformer_depth,context_dim=context_dim,n_embed=n_embed,legacy=legacy,disable_self_attentions=None,num_attention_blocks=num_attention_blocks,disable_middle_self_attn=disable_middle_self_attn,
# #                                                 use_linear_in_transformer=use_linear_in_transformer,hyperconfig=hyperconfig,stride=stride,num_conditions=num_conditions,embedding_dim=embedding_dim,hypercond=cur_hypercond,size=size, restore_ckpt=restore_ckpt, groups=groups)        
# #                 self.load_model_parts(self.new_controlNet_1, f'multi_ckpt/{epoch_num[1]}.pt')
# #                 self.new_controlNets = nn.ModuleList([self.new_controlNet_0, self.new_controlNet_1])
# #             else: #isinstance(str):
# #                 cur_hypercond=int(epoch_num.split('_')[1])
# #                 self.new_controlNet_1 = ControlSingle(image_size=image_size, in_channels=in_channels, model_channels=model_channels, hint_channels=hint_channels, num_res_blocks=num_res_blocks, attention_resolutions=attention_resolutions,
# #                                                     dropout=dropout, channel_mult=channel_mult, conv_resample=conv_resample, dims=dims, use_checkpoint=use_checkpoint, use_fp16=use_fp16, num_heads=num_heads, num_head_channels=num_head_channels,
# #                                                     num_heads_upsample=num_heads_upsample, use_scale_shift_norm=use_scale_shift_norm,resblock_updown=resblock_updown,use_new_attention_order=use_new_attention_order, use_spatial_transformer=use_spatial_transformer,
# #                                                     transformer_depth=transformer_depth,context_dim=context_dim,n_embed=n_embed,legacy=legacy,disable_self_attentions=None,num_attention_blocks=num_attention_blocks,disable_middle_self_attn=disable_middle_self_attn,
# #                                                     use_linear_in_transformer=use_linear_in_transformer,hyperconfig=hyperconfig,stride=stride,num_conditions=num_conditions,embedding_dim=embedding_dim,hypercond=cur_hypercond)        
# #                 self.load_model_parts(self.new_controlNet_1, f'multi_ckpt/{epoch_num}.pt')  
# #                 self.new_controlNets = nn.ModuleList([self.new_controlNet_1])

# #     def make_zero_conv(self, channels):
# #         return TimestepEmbedSequential(zero_module(conv_nd(self.dims, channels, channels, 1, padding=0)))

# #     def save_model_parts(self, model, filepath):
# #         state_dict = {
# #             'input_hint_block_new': model.input_hint_block_new.state_dict(),
# #             'input_blocks': model.input_blocks.state_dict(),
# #             'zero_convs': model.zero_convs.state_dict(),
# #             'middle_block': model.middle_block.state_dict(),
# #             'middle_block_out': model.middle_block_out.state_dict()
# #         }
# #         torch.save(state_dict, filepath)

# #     def load_model_parts(self, model, filepath):
# #         state_dict = torch.load(filepath)
# #         model.input_hint_block_new.load_state_dict(state_dict['input_hint_block_new'])
# #         model.input_blocks.load_state_dict(state_dict['input_blocks'])
# #         model.zero_convs.load_state_dict(state_dict['zero_convs'])
# #         model.middle_block.load_state_dict(state_dict['middle_block'])
# #         model.middle_block_out.load_state_dict(state_dict['middle_block_out'])

# #     def forward(self, x, hint, timesteps, context ,hyper_scales=None,PCA=None, control_scale=None,training=False,multi=False,**kwargs):
# #         t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
# #         emb = self.time_embed(t_emb)
# #         if training==False:
# #             if multi==True:
# #                 self.hypercolumn.cond = [0, 1, 0]
# #                 hs = [[] for _ in range(len(self.input_blocks) + 1)]
# #                 outs = []
# #                 for i in range(3):
# #                     h = x.type(self.dtype)
# #                     if i==0:
# #                         guided_hint = self.input_hint_block_new(hint, emb, context,hyper_scale=hyper_scales[i], cond=self.hypercolumn.cond[i]) #self.hypercolumn.cond.index(i)
# #                         for j, (module, zero_conv) in enumerate(zip(self.input_blocks, self.zero_convs)):
# #                             if guided_hint is not None:
# #                                 h = module(h, emb, context)
# #                                 h += guided_hint
# #                                 guided_hint = None
# #                             else:
# #                                 h = module(h, emb, context)
# #                             hs[j].append(zero_conv(h, emb, context))
# #                         h = self.middle_block(h, emb, context)    
# #                         hs[-1].append(self.middle_block_out(h, emb, context))
# #                     else:
# #                         model = self.new_controlNets[i-1] #self.hypercolumn.cond.index(i)-1] #[i-1]
# #                         guided_hint = model.input_hint_block_new(hint, emb, context,hyper_scale=hyper_scales[i], cond=self.hypercolumn.cond[i]) #hyper_scales[self.hypercolumn.cond.index(i)], cond=i)
# #                         for j, (module, zero_conv) in enumerate(zip(model.input_blocks, model.zero_convs)):
# #                             if guided_hint is not None:
# #                                 h = module(h, emb, context)
# #                                 h += guided_hint
# #                                 guided_hint = None
# #                             else:
# #                                 h = module(h, emb, context)
# #                             hs[j].append(zero_conv(h, emb, context))
# #                         h = model.middle_block(h, emb, context)
# #                         hs[-1].append(self.middle_block_out(h, emb, context))
# #                 for i in range(len(hs)):
# #                     outs.append(control_scale[0]*hs[i][0]+ control_scale[1]*hs[i][1]+ control_scale[2]*hs[i][2])
# #             else:
                
# #                 guided_hint = self.input_hint_block_new(hint, emb, context, hyper_scale=hyper_scales, cond=self.hypercolumn.cond[0])
# #                 outs = []
# #                 h = x.type(self.dtype)
# #                 for j, (module, zero_conv) in enumerate(zip(self.input_blocks, self.zero_convs)):
# #                     if guided_hint is not None:
# #                         h = module(h, emb, context)
# #                         h += guided_hint
# #                         guided_hint = None
# #                     else:
# #                         h = module(h, emb, context)
# #                     outs.append(zero_conv(h, emb, context))
# #                 h = self.middle_block(h, emb, context)
# #                 outs.append(self.middle_block_out(h, emb, context))
# #         else:
# #             guided_hint = self.input_hint_block_new(hint, emb, context, hyper_scale=1, cond=self.hypercolumn.cond[0])
# #             outs = []
# #             h = x.type(self.dtype)
# #             for j, (module, zero_conv) in enumerate(zip(self.input_blocks, self.zero_convs)):
# #                 if guided_hint is not None:
# #                     h = module(h, emb, context)
# #                     h += guided_hint
# #                     guided_hint = None
# #                 else:
# #                     h = module(h, emb, context)
# #                 outs.append(zero_conv(h, emb, context))
# #             h = self.middle_block(h, emb, context)
# #             outs.append(self.middle_block_out(h, emb, context))
# #         return outs   

# # class ControlLDM(LatentDiffusion):

# #     def __init__(self, control_stage_config, control_key, only_mid_control,control_step=0, use_semantic=False,semantic_strength=1,semantic_control_config=None, model_path=None, *args, **kwargs):
# #         super().__init__(*args, **kwargs)
# #         self.control_model = instantiate_from_config(control_stage_config)
# #         self.control_key = control_key
# #         self.only_mid_control = only_mid_control
# #         self.control_scales = [1.] * 13      
# #         self.control_step = control_step
# #         self.use_semantic = use_semantic
# #         self.semantic_strength = semantic_strength
# #         self.semantic_control_config = semantic_control_config
# #         if semantic_control_config:
# #             self.model_path = model_path
# #             self.semantic_adapter = instantiate_from_config(self.semantic_control_config)
# #             semantic_adapter_state_dict = torch.load(self.model_path)
# #             self.semantic_adapter.load_state_dict(semantic_adapter_state_dict)
        
# #         if isinstance(self.control_model.hypercolumn,HyperColumnLGN):
# #             self.hypercond = control_stage_config['params']['hyperconfig']['params']['hypercond']
# #         else:
# #             self.hypercond = [0]
# #         self.select()
    
# #     def training_step(self, batch, batch_idx):
# #         return super().training_step(batch, batch_idx)
    
    
# #     # def validation_step(self, batch, batch_idx):
# #     #     return super().training_step(batch, batch_idx)

# #     @torch.no_grad()
# #     def get_input(self, batch, k, bs=None, *args, **kwargs):
# #         x, c = super().get_input(batch, self.first_stage_key, *args, **kwargs)
# #         control = batch[self.control_key]
# #         # print('c:',c)
        
# #         if bs is not None:
# #             control = control[:bs]
# #         control = control.to(self.device)
# #         control = einops.rearrange(control, 'b h w c -> b c h w')
# #         control = control.to(memory_format=torch.contiguous_format).float()
# #         return x, dict(c_crossattn=[c], c_concat=[control])

# #     def apply_model(self, x_noisy, t, cond, hyper_scales=None, PCA=None, control_scale=None,training=False,multi=False,*args, **kwargs):
# #         assert isinstance(cond, dict)
# #         diffusion_model = self.model.diffusion_model
# #         # print('t:',t)

# #         cond_txt = torch.cat(cond['c_crossattn'], 1)

# #         if cond['c_concat'] is None:
# #             eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=None, only_mid_control=self.only_mid_control)
# #         else:
# #             if self.use_semantic:
# #                 semantic_control = self.semantic_adapter(cond['semantic_control'][0])
# #                 cond_txt = torch.cat([cond_txt, self.semantic_strength*semantic_control], dim=1)
# #             control = self.control_model(x=x_noisy, hint=torch.cat(cond['c_concat'], 1), timesteps=t, context=cond_txt, group_num=0, hyper_scales=hyper_scales,control_scale=control_scale,training=training,multi=multi)
# #             if t[0]<self.control_step:
# #                 control = [c * 0. for c, scale in zip(control, self.control_scales)]
# #             else:
# #                 control = [c * scale for c, scale in zip(control, self.control_scales)]
# #             eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=control, only_mid_control=self.only_mid_control)            

# #         return eps
    
# #     def apply_model_train(self, x_noisy, t, cond, *args, **kwargs):
# #         assert isinstance(cond, dict)
# #         diffusion_model = self.model.diffusion_model
# #         # print('t:',t)

# #         cond_txt = torch.cat(cond['c_crossattn'], 1)

# #         if cond['c_concat'] is None:
# #             eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=None, only_mid_control=self.only_mid_control)
# #         else:
# #             control = self.control_model(x=x_noisy, hint=torch.cat(cond['c_concat'], 1), timesteps=t, context=cond_txt,training=True)
# #             if t[0]<self.control_step:
# #                 control = [c * 0. for c, scale in zip(control, self.control_scales)]
# #             else:
# #                 control = [c * scale for c, scale in zip(control, self.control_scales)]
# #             eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=control, only_mid_control=self.only_mid_control)
# #         return eps    

# #     @torch.no_grad()
# #     def get_unconditional_conditioning(self, N):
# #         return self.get_learned_conditioning([""] * N)

# #     @torch.no_grad()
# #     def log_images(self, batch, N=4, n_row=2, sample=False, ddim_steps=50, ddim_eta=0.0, return_keys=None,
# #                    quantize_denoised=True, inpaint=True, plot_denoise_rows=False, plot_progressive_rows=True,
# #                    plot_diffusion_rows=False, unconditional_guidance_scale=7.5, unconditional_guidance_label=None,
# #                    use_ema_scope=True,
# #                    **kwargs):
# #         use_ddim = ddim_steps is not None

# #         log = dict()
# #         z, c = self.get_input(batch, self.first_stage_key, bs=N)
# #         c_cat, c = c["c_concat"][0][:N], c["c_crossattn"][0][:N]
# #         N = min(z.shape[0], N)
# #         n_row = min(z.shape[0], n_row)
# #         log["reconstruction"] = self.decode_first_stage(z)
# #         log["control"] = c_cat * 2.0 - 1.0
# #         log["conditioning"] = log_txt_as_img((512, 512), batch[self.cond_stage_key], size=16)

# #         if plot_diffusion_rows:
# #             # get diffusion row
# #             diffusion_row = list()
# #             z_start = z[:n_row]
# #             for t in range(self.num_timesteps):
# #                 if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
# #                     t = repeat(torch.tensor([t]), '1 -> b', b=n_row)
# #                     t = t.to(self.device).long()
# #                     noise = torch.randn_like(z_start)
# #                     z_noisy = self.q_sample(x_start=z_start, t=t, noise=noise)
# #                     diffusion_row.append(self.decode_first_stage(z_noisy))

# #             diffusion_row = torch.stack(diffusion_row)  # n_log_step, n_row, C, H, W
# #             diffusion_grid = rearrange(diffusion_row, 'n b c h w -> b n c h w')
# #             diffusion_grid = rearrange(diffusion_grid, 'b n c h w -> (b n) c h w')
# #             diffusion_grid = make_grid(diffusion_grid, nrow=diffusion_row.shape[0])
# #             log["diffusion_row"] = diffusion_grid

# #         if sample:
# #             # get denoise row
# #             samples, z_denoise_row = self.sample_log(cond={"c_concat": [c_cat], "c_crossattn": [c]},
# #                                                      batch_size=N, ddim=use_ddim,
# #                                                      ddim_steps=ddim_steps, eta=ddim_eta)
# #             x_samples = self.decode_first_stage(samples)
# #             log["samples"] = x_samples
# #             if plot_denoise_rows:
# #                 denoise_grid = self._get_denoise_row_from_list(z_denoise_row)
# #                 log["denoise_row"] = denoise_grid

# #         if unconditional_guidance_scale >= -1.0:
# #             uc_cross = self.get_unconditional_conditioning(N)
# #             uc_cat = c_cat  # torch.zeros_like(c_cat)
# #             uc_full = {"c_concat": [uc_cat], "c_crossattn": [uc_cross]}
# #             for slct in self.slct:
# #                 self.control_model.input_hint_block_new[0].cond = slct['cond']
# #                 suffix = slct['suffix']
# #                 print(suffix)
# #                 samples_cfg, _ = self.sample_log(cond={"c_concat": [c_cat], "c_crossattn": [c]},
# #                                                 batch_size=N, ddim=use_ddim,
# #                                                 ddim_steps=ddim_steps, eta=ddim_eta,
# #                                                 unconditional_guidance_scale=unconditional_guidance_scale,
# #                                                 unconditional_conditioning=uc_full,training=True
# #                                                 )
# #                 x_samples_cfg = self.decode_first_stage(samples_cfg)
# #                 deconv = self.control_model.input_hint_block_new[0].deconv(c_cat)
# #                 log[f"{suffix}_samples_cfg_scale_{unconditional_guidance_scale:.2f}"] = x_samples_cfg
# #                 log[f"{suffix}_deconv_samples_cfg_scale_{unconditional_guidance_scale:.2f}"] = deconv
# #                 # self.control_model.input_hint_block_new[0].cond = None

# #         return log

# #     @torch.no_grad()
# #     def sample_log(self, cond, batch_size, ddim, ddim_steps, training=False,**kwargs):
# #         ddim_sampler = DDIMSampler(self)
# #         b, c, h, w = cond["c_concat"][0].shape
# #         shape = (self.channels, h // 8, w // 8)
# #         samples, intermediates = ddim_sampler.sample(ddim_steps, batch_size, shape, cond, verbose=False,training=training, **kwargs)
# #         return samples, intermediates

# #     def configure_optimizers(self):
# #         lr = self.learning_rate
# #         params = list(self.control_model.parameters())
# #         if not self.sd_locked:
# #             params += list(self.model.diffusion_model.output_blocks.parameters())
# #             params += list(self.model.diffusion_model.out.parameters())
# #         opt = torch.optim.AdamW(params, lr=lr)
# #         return opt

# #     def low_vram_shift(self, is_diffusing):
# #         if is_diffusing:
# #             self.model = self.model.cuda()
# #             self.control_model = self.control_model.cuda()
# #             self.first_stage_model = self.first_stage_model.cpu()
# #             self.cond_stage_model = self.cond_stage_model.cpu()
# #         else:
# #             self.model = self.model.cpu()
# #             self.control_model = self.control_model.cpu()
# #             self.first_stage_model = self.first_stage_model.cuda()
# #             self.cond_stage_model = self.cond_stage_model.cuda()

# #     @torch.no_grad()
# #     def select(self):
# #         # self.slct = [{'cond':[0],'suffix':'0'},{'cond':[1],'suffix':'1'},{'cond':[2],'suffix':'2'},{'cond':[3],'suffix':'3'},{'cond':[0,1],'suffix':'01'},
# #         #              {'cond':[0,2],'suffix':'02'},{'cond':[0,3],'suffix':'03'},{'cond':[1,2],'suffix':'12'},{'cond':[1,3],'suffix':'13'},{'cond':[2,3],'suffix':'23'},
# #         #              {'cond':[0,1,2],'suffix':'012'},{'cond':[0,1,3],'suffix':'013'},{'cond':[1,2,3],'suffix':'123'},
# #         #              {'cond':[0,1,2,3],'suffix':'0123'}]
# #         self.slct = [{'cond':[i],'suffix':f'{i}'} for i in self.hypercond]

# # # import einops
# # # import torch
# # # import torch as th
# # # import torch.nn as nn
# # # import torchvision.transforms as transforms
# # # import sys
# # # import numpy as np
# # # import random
# # # from einops import rearrange, repeat,reduce
# # # from torchvision import transforms

# # # # sys.path.append('/home/chenzhiqiang/code/')
# # # sys.path.append('/home/uchihawdt/control-net-main-v0.5')
# # # from hypercolumn.vit_pytorch.train_V1_sep_new import Column_trans_rot_lgn

# # # from ldm.modules.diffusionmodules.util import (
# # #     conv_nd,
# # #     linear,
# # #     zero_module,
# # #     timestep_embedding,
# # # )

# # # from einops import rearrange, repeat
# # # from torchvision.utils import make_grid
# # # from ldm.modules.attention import SpatialTransformer
# # # from ldm.modules.diffusionmodules.openaimodel import UNetModel, TimestepEmbedSequential, ResBlock, Downsample, AttentionBlock
# # # from ldm.models.diffusion.ddpm import LatentDiffusion
# # # from ldm.util import log_txt_as_img, exists, instantiate_from_config
# # # from ldm.models.diffusion.ddim import DDIMSampler


# # # class ControlledUnetModel(UNetModel):
# # #     def forward(self, x, timesteps=None, context=None, control=None, only_mid_control=False, **kwargs):
# # #         hs = []
# # #         with torch.no_grad():
# # #             t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
# # #             emb = self.time_embed(t_emb)
# # #             h = x.type(self.dtype)
# # #             for module in self.input_blocks:
# # #                 h = module(h, emb, context)
# # #                 hs.append(h)
# # #             h = self.middle_block(h, emb, context)

# # #         if control is not None:
# # #             h += control.pop()

# # #         for i, module in enumerate(self.output_blocks):
# # #             if only_mid_control or control is None:
# # #                 h = torch.cat([h, hs.pop()], dim=1)
# # #             else:
# # #                 h = torch.cat([h, hs.pop() + control.pop()], dim=1)
# # #             h = module(h, emb, context)

# # #         h = h.type(x.dtype)
# # #         return self.out(h)

# # # def disabled_train(self, mode=True):
# # #     """Overwrite model.train with this function to make sure train/eval mode
# # #     does not change anymore."""
# # #     return self


# # # class HyperColumnLGN(nn.Module):
# # #     def __init__(self,key=0,hypercond=[0],size=512,restore_ckpt = '/home/uchihawdt/control-net-main-v0.5/hypercolumn/checkpoint/imagenet/equ_nv16_vl4_rn1_Bipolar_norm.pth'):
# # #         super().__init__()
# # #         ckpt = torch.load(restore_ckpt)
# # #         hc = Column_trans_rot_lgn(ckpt['arg'])
# # #         hc.load_state_dict(ckpt['state_dict'], strict=False)
# # #         self.lgn_ende = hc.lgn_ende[0].eval()
# # #         self.lgn_ende.train = disabled_train
# # #         for param in self.lgn_ende.parameters():
# # #             param.requires_grad = False

# # #         self.resize = transforms.Resize(size)
# # #         if size == 128:
# # #             self.pad = nn.ConstantPad2d((1,1,1,1),0.)
# # #         else:
# # #             self.pad = nn.Identity()
        
# # #         self.groups = [[0,1,4,8,9,15],[2,3],[5,12],[10],[6,7],[11,13,14],[5],[12],[2],[3]]
# # #         self.p = [0. for i in range(len(self.groups))]
# # #         self.p[0] = 1.

# # #         norm_mean = np.array([0.50705882, 0.48666667, 0.44078431])
# # #         norm_std = np.array([0.26745098, 0.25568627, 0.27607843])
# # #         self.norm = transforms.Normalize(norm_mean, norm_std)
# # #         self.cond = hypercond
# # #         self.slct = None
    
# # #     def forward(self,x, hyper_scale=1, cond=None):
# # #         x = self.resize(x)
# # #         s = x.size()
# # #         r = torch.zeros(1,16,1,1).to(x.device)
# # #         cond = cond if cond!=None else self.cond

# # #         if cond is None:
# # #             c = [i for i in range(len(self.groups))]
# # #             random.shuffle(c)
# # #             # print(self.groups[c[0]])
# # #             pa = random.random()
# # #             for i in range(len(self.groups)):
# # #                 if pa < self.p[i]:
# # #                     for j in self.groups[c[i]]:
# # #                         r[:,j,:,:] = 1.
# # #         else:
# # #             # for i in cond:
# # #             #     for j in self.groups[i]:
# # #             #         r[:,j,:,:] = 1
# # #             for j in self.groups[cond]:
# # #                 r[:,j,:,:] = 1

# # #         r = repeat(r,'n c h w -> n (c repeat) h w',repeat=4)
# # #         out = self.lgn_ende(self.norm(x))*r
# # #         out = self.pad(self.lgn_ende.deconv(out))
# # #         # out *= hyper_scale
# # #         return out
    
# # #     def deconv(self,x):
# # #         x = self.resize(x)
# # #         s = x.size()
# # #         r = torch.zeros(1,16,1,1).to(x.device)
# # #         if self.cond is not None:
# # #             for i in self.cond:
# # #                 for j in self.groups[i]:
# # #                     r[:,j,:,:] = 1

# # #         r = repeat(r,'n c h w -> n (c repeat) h w',repeat=4)
# # #         conv = self.lgn_ende(self.norm(x))*r
# # #         deconv = self.lgn_ende.deconv(conv)
# # #         deconv = (deconv - reduce(deconv,'b c h w -> b c () ()', 'min'))/(reduce(deconv,'b c h w -> b c () ()', 'max')-reduce(deconv,'b c h w -> b c () ()', 'min'))*2.-1.

# # #         return deconv

# # # class HyperColumnLGNVisual(nn.Module):
# # #     def __init__(self,key=0,hypercond=[0],size=512,restore_ckpt = '/home/uchihawdt/control-net-main-v0.5/hypercolumn/checkpoint/imagenet/equ_nv16_vl4_rn1_Bipolar_norm.pth'):
# # #         super().__init__()
# # #         ckpt = torch.load(restore_ckpt)
# # #         hc = Column_trans_rot_lgn(ckpt['arg'])
# # #         hc.load_state_dict(ckpt['state_dict'], strict=False)
# # #         self.lgn_ende = hc.lgn_ende[0].eval()
# # #         self.lgn_ende.train = disabled_train
# # #         for param in self.lgn_ende.parameters():
# # #             param.requires_grad = False

# # #         self.resize = transforms.Resize(size)
# # #         if size == 128:
# # #             self.pad = nn.ConstantPad2d((1,1,1,1),0.)
# # #         else:
# # #             self.pad = nn.Identity()
        

# # #         # self.groups = [[2,3],[0,1,4,8,9,15],[5,6,7,10,12],[11,13,14]]
# # #         self.groups = [[0,1,4,8,9,15],[2,3],[5,12],[10],[6,7],[11,13,14],[5],[12],[2],[3]]
# # #         # self.groups = [[2],[3],[5],[12]]
            
# # #         # self.p = [1.,0.5,0.25,0.125]
# # #         self.p = [0. for i in range(len(self.groups))]
# # #         self.p[0] = 1.

# # #         norm_mean = np.array([0.50705882, 0.48666667, 0.44078431])
# # #         norm_std = np.array([0.26745098, 0.25568627, 0.27607843])
# # #         self.norm = transforms.Normalize(norm_mean, norm_std)
# # #         self.cond = hypercond
# # #         self.slct = None

# # #     # def forward(self,x):
# # #     #     s = x.size()
# # #     #     r = torch.zeros(1,16,1,1).to(x.device)

# # #     #     if self.cond is None:
# # #     #         c = [i for i in range(len(self.groups))]
# # #     #         random.shuffle(c)
# # #     #         # print(self.groups[c[0]])
# # #     #         pa = random.random()
# # #     #         for i in range(4):
# # #     #             if pa < self.p[i]:
# # #     #                 for j in self.groups[c[i]]:
# # #     #                     r[:,j,:,:] = 1.
# # #     #     else:
# # #     #         for i in self.cond:
# # #     #             for j in self.groups[i]:
# # #     #                 r[:,j,:,:] = 1

# # #     #     r = repeat(r,'n c h w -> n (c repeat) h w',repeat=4)
# # #     #     return self.lgn_ende(self.norm(x))*r
    
# # #     def forward(self,x):
# # #         x = self.resize(x)
# # #         s = x.size()
# # #         r = torch.zeros(1,16,1,1).to(x.device)

# # #         if self.cond is None:
# # #             c = [i for i in range(len(self.groups))]
# # #             random.shuffle(c)
# # #             # print(self.groups[c[0]])
# # #             pa = random.random()
# # #             for i in range(len(self.groups)):
# # #                 if pa < self.p[i]:
# # #                     for j in self.groups[c[i]]:
# # #                         r[:,j,:,:] = 1.
# # #         else:
# # #             for i in self.cond:
# # #                 for j in self.groups[i]:
# # #                     r[:,j,:,:] = 1

# # #         # r = repeat(r,'n c h w -> n (c repeat) h w',repeat=4)
# # #         # out = self.lgn_ende(self.norm(x))*r
# # #         # # print('out_conv:',out.size())
# # #         # out = self.pad(self.lgn_ende.deconv(out))
# # #         # # out = self.lgn_ende.deconv(self.lgn_ende(self.norm(x))*r)
# # #         # # print('out:',out.size())
# # #         # return out

# # #         out = self.lgn_ende(self.norm(x))
# # #         # out = rearrange(out, 'b (c n) h w -> b c (n h) w',n=16)
# # #         out = rearrange(out, 'b (n c) h w -> b c (n h) w', n=16)
# # #         # out = self.pad(self.lgn_ende.deconv(out))
# # #         return out
    
# # #     def deconv(self,x):
# # #         x = self.resize(x)
# # #         s = x.size()
# # #         r = torch.zeros(1,16,1,1).to(x.device)
# # #         if self.cond is not None:
# # #             for i in self.cond:
# # #                 for j in self.groups[i]:
# # #                     r[:,j,:,:] = 1

# # #         r = repeat(r,'n c h w -> n (c repeat) h w',repeat=4)
# # #         conv = self.lgn_ende(self.norm(x))*r
# # #         deconv = self.lgn_ende.deconv(conv)
# # #         deconv = (deconv - reduce(deconv,'b c h w -> b c () ()', 'min'))/(reduce(deconv,'b c h w -> b c () ()', 'max')-reduce(deconv,'b c h w -> b c () ()', 'min'))*2.-1.

# # #         return deconv
    
# # # class HyperColumnLGNFeature(nn.Module):
# # #     def __init__(self,key=0,hypercond=[0],size=512,restore_ckpt = '/home/uchihawdt/control-net-main-v0.5/hypercolumn/checkpoint/imagenet/equ_nv16_vl4_rn1_Bipolar_norm.pth'):
# # #         super().__init__()
# # #         ckpt = torch.load(restore_ckpt)
# # #         hc = Column_trans_rot_lgn(ckpt['arg'])
# # #         hc.load_state_dict(ckpt['state_dict'], strict=False)
# # #         self.lgn_ende = hc.lgn_ende[0].eval()
# # #         self.lgn_ende.train = disabled_train
# # #         for param in self.lgn_ende.parameters():
# # #             param.requires_grad = False

# # #         self.resize = transforms.Resize(size)
# # #         if size == 128:
# # #             self.pad = nn.ConstantPad2d((1,1,1,1),0.)
# # #         else:
# # #             self.pad = nn.Identity()
        

# # #         # self.groups = [[2,3],[0,1,4,8,9,15],[5,6,7,10,12],[11,13,14]]
# # #         self.groups = [[0,1,4,8,9,15],[2,3],[5,12],[10],[6,7],[11,13,14],[5],[12],[2],[3]]
# # #         # self.groups = [[2],[3],[5],[12]]
            
# # #         # self.p = [1.,0.5,0.25,0.125]
# # #         self.p = [0. for i in range(len(self.groups))]
# # #         self.p[0] = 1.

# # #         norm_mean = np.array([0.50705882, 0.48666667, 0.44078431])
# # #         norm_std = np.array([0.26745098, 0.25568627, 0.27607843])
# # #         self.norm = transforms.Normalize(norm_mean, norm_std)
# # #         self.cond = hypercond
# # #         self.slct = None
    
# # #     def forward(self,x):
# # #         x = self.resize(x)
# # #         s = x.size()
# # #         # r = torch.zeros(1,16,1,1).to(x.device)

# # #         # if self.cond is None:
# # #         #     c = [i for i in range(len(self.groups))]
# # #         #     random.shuffle(c)
# # #         #     # print(self.groups[c[0]])
# # #         #     pa = random.random()
# # #         #     for i in range(len(self.groups)):
# # #         #         if pa < self.p[i]:
# # #         #             for j in self.groups[c[i]]:
# # #         #                 r[:,j,:,:] = 1.
# # #         # else:
# # #         #     for i in self.cond:
# # #         #         for j in self.groups[i]:
# # #         #             r[:,j,:,:] = 1

# # #         # r = repeat(r,'n c h w -> n (c repeat) h w',repeat=4)
# # #         out = self.lgn_ende(self.norm(x))
# # #         out = out[:,40:44,:,:]
# # #         # out = self.pad(self.lgn_ende.deconv(out))
# # #         return out

# # #     def deconv(self,x):
# # #         x = self.resize(x)
# # #         s = x.size()
# # #         r = torch.zeros(1,16,1,1).to(x.device)
# # #         if self.cond is not None:
# # #             for i in self.cond:
# # #                 for j in self.groups[i]:
# # #                     r[:,j,:,:] = 1

# # #         r = repeat(r,'n c h w -> n (c repeat) h w',repeat=4)
# # #         conv = self.lgn_ende(self.norm(x))*r
# # #         deconv = self.lgn_ende.deconv(conv)
# # #         deconv = (deconv - reduce(deconv,'b c h w -> b c () ()', 'min'))/(reduce(deconv,'b c h w -> b c () ()', 'max')-reduce(deconv,'b c h w -> b c () ()', 'min'))*2.-1.

# # #         return deconv


# # # class Canny(nn.Module):
# # #     def __init__(self,para=None):
# # #         super().__init__()
# # #     def forward(self,x):
# # #         return x
# # #     def deconv(self,x):
# # #         return x


# # # class ConditionEmbedding(nn.Module):
# # #     def __init__(self, num_conditions, embedding_dim):
# # #         super(ConditionEmbedding, self).__init__()
# # #         self.embedding = nn.Embedding(num_conditions, embedding_dim)

# # #     def forward(self, condition_type):
# # #         return self.embedding(condition_type)

# # # class ControlSingle(nn.Module):
# # #     def __init__(self,image_size, in_channels, model_channels, hint_channels, num_res_blocks, attention_resolutions, dropout=0, channel_mult=(1, 2, 4, 8), conv_resample=True, dims=2, use_checkpoint=False, use_fp16=False, num_heads=-1, num_head_channels=-1, num_heads_upsample=-1, use_scale_shift_norm=False, resblock_updown=False, use_new_attention_order=False, use_spatial_transformer=False, transformer_depth=1, context_dim=None, n_embed=None, legacy=True, disable_self_attentions=None, num_attention_blocks=None, disable_middle_self_attn=False, use_linear_in_transformer=False, hyperconfig=None, stride=[2,2,2], num_conditions=6, embedding_dim=128,hypercond=0):
# # #         super().__init__()
# # #         self.dims = dims
# # #         hyperconfig['params']['hypercond'] = [hypercond]
# # #         self.hypercolumn = instantiate_from_config(hyperconfig)
# # #         self.input_blocks = nn.ModuleList(
# # #             [
# # #                 TimestepEmbedSequential(
# # #                     conv_nd(dims, in_channels, model_channels, 3, padding=1)
# # #                 )
# # #             ]
# # #         )
# # #         self.zero_convs = nn.ModuleList([self.make_zero_conv(model_channels)])

# # #         if isinstance(num_res_blocks, int):
# # #             self.num_res_blocks = len(channel_mult) * [num_res_blocks]
# # #         else:
# # #             if len(num_res_blocks) != len(channel_mult):
# # #                 raise ValueError("provide num_res_blocks either as an int (globally constant) or "
# # #                                  "as a list/tuple (per-level) with the same length as channel_mult")
# # #             self.num_res_blocks = num_res_blocks
        
# # #         time_embed_dim = model_channels * 4
# # #         self.input_hint_block_new = TimestepEmbedSequential(self.hypercolumn,
# # #                                                             conv_nd(dims, hint_channels, 16, 3, padding=1),nn.SiLU(),
# # #                                                             conv_nd(dims, 16, 16, 3, padding=1),nn.SiLU(),
# # #                                                             conv_nd(dims, 16, 32, 3, padding=1, stride=stride[0]),nn.SiLU(),
# # #                                                             conv_nd(dims, 32, 32, 3, padding=1),nn.SiLU(),
# # #                                                             conv_nd(dims, 32, 96, 3, padding=1, stride=stride[1]),nn.SiLU(),
# # #                                                             conv_nd(dims, 96, 96, 3, padding=1),nn.SiLU(),
# # #                                                             conv_nd(dims, 96, 256, 3, padding=1, stride=stride[2]),nn.SiLU(),
# # #                                                             zero_module(conv_nd(dims, 256, model_channels, 3, padding=1)))
# # #         self._feature_size = model_channels
# # #         input_block_chans = [model_channels]
# # #         ch = model_channels
# # #         ds = 1
# # #         for level, mult in enumerate(channel_mult):
# # #             for nr in range(self.num_res_blocks[level]):
# # #                 layers = [
# # #                     ResBlock(
# # #                         ch,
# # #                         time_embed_dim,
# # #                         dropout,
# # #                         out_channels=mult * model_channels,
# # #                         dims=dims,
# # #                         use_checkpoint=use_checkpoint,
# # #                         use_scale_shift_norm=use_scale_shift_norm,
# # #                     )
# # #                 ]
# # #                 ch = mult * model_channels
# # #                 if ds in attention_resolutions:
# # #                     if num_head_channels == -1:
# # #                         dim_head = ch // num_heads
# # #                     else:
# # #                         num_heads = ch // num_head_channels
# # #                         dim_head = num_head_channels
# # #                     if legacy:
# # #                         # num_heads = 1
# # #                         dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
# # #                     if exists(disable_self_attentions):
# # #                         disabled_sa = disable_self_attentions[level]
# # #                     else:
# # #                         disabled_sa = False

# # #                     if not exists(num_attention_blocks) or nr < num_attention_blocks[level]:
# # #                         layers.append(
# # #                             AttentionBlock(
# # #                                 ch,
# # #                                 use_checkpoint=use_checkpoint,
# # #                                 num_heads=num_heads,
# # #                                 num_head_channels=dim_head,
# # #                                 use_new_attention_order=use_new_attention_order,
# # #                             ) if not use_spatial_transformer else SpatialTransformer(
# # #                                 ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
# # #                                 disable_self_attn=disabled_sa, use_linear=use_linear_in_transformer,
# # #                                 use_checkpoint=use_checkpoint
# # #                             )
# # #                         )
# # #                 self.input_blocks.append(TimestepEmbedSequential(*layers))
# # #                 self.zero_convs.append(self.make_zero_conv(ch))
# # #                 self._feature_size += ch
# # #                 input_block_chans.append(ch)
# # #             if level != len(channel_mult) - 1:
# # #                 out_ch = ch
# # #                 self.input_blocks.append(
# # #                     TimestepEmbedSequential(
# # #                         ResBlock(
# # #                             ch,
# # #                             time_embed_dim,
# # #                             dropout,
# # #                             out_channels=out_ch,
# # #                             dims=dims,
# # #                             use_checkpoint=use_checkpoint,
# # #                             use_scale_shift_norm=use_scale_shift_norm,
# # #                             down=True,
# # #                         )
# # #                         if resblock_updown
# # #                         else Downsample(
# # #                             ch, conv_resample, dims=dims, out_channels=out_ch
# # #                         )
# # #                     )
# # #                 )
# # #                 ch = out_ch
# # #                 input_block_chans.append(ch)
# # #                 self.zero_convs.append(self.make_zero_conv(ch))
# # #                 ds *= 2
# # #                 self._feature_size += ch

# # #         if num_head_channels == -1:
# # #             dim_head = ch // num_heads
# # #         else:
# # #             num_heads = ch // num_head_channels
# # #             dim_head = num_head_channels
# # #         if legacy:
# # #             # num_heads = 1
# # #             dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
# # #         self.middle_block = TimestepEmbedSequential(
# # #             ResBlock(
# # #                 ch,
# # #                 time_embed_dim,
# # #                 dropout,
# # #                 dims=dims,
# # #                 use_checkpoint=use_checkpoint,
# # #                 use_scale_shift_norm=use_scale_shift_norm,
# # #             ),
# # #             AttentionBlock(
# # #                 ch,
# # #                 use_checkpoint=use_checkpoint,
# # #                 num_heads=num_heads,
# # #                 num_head_channels=dim_head,
# # #                 use_new_attention_order=use_new_attention_order,
# # #             ) if not use_spatial_transformer else SpatialTransformer(  # always uses a self-attn
# # #                 ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
# # #                 disable_self_attn=disable_middle_self_attn, use_linear=use_linear_in_transformer,
# # #                 use_checkpoint=use_checkpoint
# # #             ),
# # #             ResBlock(
# # #                 ch,
# # #                 time_embed_dim,
# # #                 dropout,
# # #                 dims=dims,
# # #                 use_checkpoint=use_checkpoint,
# # #                 use_scale_shift_norm=use_scale_shift_norm,
# # #             ),
# # #         )
# # #         self.middle_block_out = self.make_zero_conv(ch)
# # #     def make_zero_conv(self, channels):
# # #         return TimestepEmbedSequential(zero_module(conv_nd(self.dims, channels, channels, 1, padding=0)))

# # # class ControlNet(nn.Module):
# # #     def __init__(
# # #             self,
# # #             image_size,
# # #             in_channels,
# # #             model_channels,
# # #             hint_channels,
# # #             num_res_blocks,
# # #             attention_resolutions,
# # #             dropout=0,
# # #             channel_mult=(1, 2, 4, 8),
# # #             conv_resample=True,
# # #             dims=2,
# # #             use_checkpoint=False,
# # #             use_fp16=False,
# # #             num_heads=-1,
# # #             num_head_channels=-1,
# # #             num_heads_upsample=-1,
# # #             use_scale_shift_norm=False,
# # #             resblock_updown=False,
# # #             use_new_attention_order=False,
# # #             use_spatial_transformer=False,  # custom transformer support
# # #             transformer_depth=1,  # custom transformer support
# # #             context_dim=None,  # custom transformer support
# # #             n_embed=None,  # custom support for prediction of discrete ids into codebook of first stage vq model
# # #             legacy=True,
# # #             disable_self_attentions=None,
# # #             num_attention_blocks=None,
# # #             disable_middle_self_attn=False,
# # #             use_linear_in_transformer=False,
# # #             hyperconfig=None,
# # #             stride=[2,2,2],
# # #             num_conditions=6, 
# # #             embedding_dim=128,
# # #             is_train=False,
# # #             epoch_num=None,
# # #     ):
# # #         super().__init__()
# # #         if use_spatial_transformer:
# # #             assert context_dim is not None, 'Fool!! You forgot to include the dimension of your cross-attention conditioning...'

# # #         if context_dim is not None:
# # #             assert use_spatial_transformer, 'Fool!! You forgot to use the spatial transformer for your cross-attention conditioning...'
# # #             from omegaconf.listconfig import ListConfig
# # #             if type(context_dim) == ListConfig:
# # #                 context_dim = list(context_dim)

# # #         if num_heads_upsample == -1:
# # #             num_heads_upsample = num_heads

# # #         if num_heads == -1:
# # #             assert num_head_channels != -1, 'Either num_heads or num_head_channels has to be set'

# # #         if num_head_channels == -1:
# # #             assert num_heads != -1, 'Either num_heads or num_head_channels has to be set'

# # #         self.dims = dims
# # #         self.image_size = image_size
# # #         self.in_channels = in_channels
# # #         self.model_channels = model_channels
# # #         if isinstance(num_res_blocks, int):
# # #             self.num_res_blocks = len(channel_mult) * [num_res_blocks]
# # #         else:
# # #             if len(num_res_blocks) != len(channel_mult):
# # #                 raise ValueError("provide num_res_blocks either as an int (globally constant) or "
# # #                                  "as a list/tuple (per-level) with the same length as channel_mult")
# # #             self.num_res_blocks = num_res_blocks
# # #         if disable_self_attentions is not None:
# # #             # should be a list of booleans, indicating whether to disable self-attention in TransformerBlocks or not
# # #             assert len(disable_self_attentions) == len(channel_mult)
# # #         if num_attention_blocks is not None:
# # #             assert len(num_attention_blocks) == len(self.num_res_blocks)
# # #             assert all(map(lambda i: self.num_res_blocks[i] >= num_attention_blocks[i], range(len(num_attention_blocks))))
# # #             print(f"Constructor of UNetModel received num_attention_blocks={num_attention_blocks}. "
# # #                   f"This option has LESS priority than attention_resolutions {attention_resolutions}, "
# # #                   f"i.e., in cases where num_attention_blocks[i] > 0 but 2**i not in attention_resolutions, "
# # #                   f"attention will still not be set.")

# # #         self.attention_resolutions = attention_resolutions
# # #         self.dropout = dropout
# # #         self.channel_mult = channel_mult
# # #         self.conv_resample = conv_resample
# # #         self.use_checkpoint = use_checkpoint
# # #         self.dtype = th.float16 if use_fp16 else th.float32
# # #         self.num_heads = num_heads
# # #         self.num_head_channels = num_head_channels
# # #         self.num_heads_upsample = num_heads_upsample
# # #         self.predict_codebook_ids = n_embed is not None
# # #         self.hyperconfig = hyperconfig
# # #         self.hypercolumn = instantiate_from_config(hyperconfig)

# # #         time_embed_dim = model_channels * 4
# # #         self.time_embed = nn.Sequential(
# # #             linear(model_channels, time_embed_dim),
# # #             nn.SiLU(),
# # #             linear(time_embed_dim, time_embed_dim),
# # #         )
# # #         self.group_embedding = ConditionEmbedding(num_conditions, embedding_dim)

# # #         self.input_blocks = nn.ModuleList(
# # #             [
# # #                 TimestepEmbedSequential(
# # #                     conv_nd(dims, in_channels, model_channels, 3, padding=1)
# # #                 )
# # #             ]
# # #         )
# # #         self.zero_convs = nn.ModuleList([self.make_zero_conv(model_channels)])

# # #         self.input_hint_block_new = TimestepEmbedSequential(
# # #             # HyperColumnLGN(hypercond=hypercond),
# # #             # Canny(),
# # #             self.hypercolumn,
# # #             conv_nd(dims, hint_channels, 16, 3, padding=1),
# # #             nn.SiLU(),
# # #             conv_nd(dims, 16, 16, 3, padding=1),
# # #             nn.SiLU(),
# # #             conv_nd(dims, 16, 32, 3, padding=1, stride=stride[0]),
# # #             nn.SiLU(),
# # #             conv_nd(dims, 32, 32, 3, padding=1),
# # #             nn.SiLU(),
# # #             conv_nd(dims, 32, 96, 3, padding=1, stride=stride[1]),
# # #             nn.SiLU(),
# # #             conv_nd(dims, 96, 96, 3, padding=1),
# # #             nn.SiLU(),
# # #             conv_nd(dims, 96, 256, 3, padding=1, stride=stride[2]),
# # #             nn.SiLU(),
# # #             zero_module(conv_nd(dims, 256, model_channels, 3, padding=1))
# # #         )

# # #         self._feature_size = model_channels
# # #         input_block_chans = [model_channels]
# # #         ch = model_channels
# # #         ds = 1
# # #         for level, mult in enumerate(channel_mult):
# # #             for nr in range(self.num_res_blocks[level]):
# # #                 layers = [
# # #                     ResBlock(
# # #                         ch,
# # #                         time_embed_dim,
# # #                         dropout,
# # #                         out_channels=mult * model_channels,
# # #                         dims=dims,
# # #                         use_checkpoint=use_checkpoint,
# # #                         use_scale_shift_norm=use_scale_shift_norm,
# # #                     )
# # #                 ]
# # #                 ch = mult * model_channels
# # #                 if ds in attention_resolutions:
# # #                     if num_head_channels == -1:
# # #                         dim_head = ch // num_heads
# # #                     else:
# # #                         num_heads = ch // num_head_channels
# # #                         dim_head = num_head_channels
# # #                     if legacy:
# # #                         # num_heads = 1
# # #                         dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
# # #                     if exists(disable_self_attentions):
# # #                         disabled_sa = disable_self_attentions[level]
# # #                     else:
# # #                         disabled_sa = False

# # #                     if not exists(num_attention_blocks) or nr < num_attention_blocks[level]:
# # #                         layers.append(
# # #                             AttentionBlock(
# # #                                 ch,
# # #                                 use_checkpoint=use_checkpoint,
# # #                                 num_heads=num_heads,
# # #                                 num_head_channels=dim_head,
# # #                                 use_new_attention_order=use_new_attention_order,
# # #                             ) if not use_spatial_transformer else SpatialTransformer(
# # #                                 ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
# # #                                 disable_self_attn=disabled_sa, use_linear=use_linear_in_transformer,
# # #                                 use_checkpoint=use_checkpoint
# # #                             )
# # #                         )
# # #                 self.input_blocks.append(TimestepEmbedSequential(*layers))
# # #                 self.zero_convs.append(self.make_zero_conv(ch))
# # #                 self._feature_size += ch
# # #                 input_block_chans.append(ch)
# # #             if level != len(channel_mult) - 1:
# # #                 out_ch = ch
# # #                 self.input_blocks.append(
# # #                     TimestepEmbedSequential(
# # #                         ResBlock(
# # #                             ch,
# # #                             time_embed_dim,
# # #                             dropout,
# # #                             out_channels=out_ch,
# # #                             dims=dims,
# # #                             use_checkpoint=use_checkpoint,
# # #                             use_scale_shift_norm=use_scale_shift_norm,
# # #                             down=True,
# # #                         )
# # #                         if resblock_updown
# # #                         else Downsample(
# # #                             ch, conv_resample, dims=dims, out_channels=out_ch
# # #                         )
# # #                     )
# # #                 )
# # #                 ch = out_ch
# # #                 input_block_chans.append(ch)
# # #                 self.zero_convs.append(self.make_zero_conv(ch))
# # #                 ds *= 2
# # #                 self._feature_size += ch

# # #         if num_head_channels == -1:
# # #             dim_head = ch // num_heads
# # #         else:
# # #             num_heads = ch // num_head_channels
# # #             dim_head = num_head_channels
# # #         if legacy:
# # #             # num_heads = 1
# # #             dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
# # #         self.middle_block = TimestepEmbedSequential(
# # #             ResBlock(
# # #                 ch,
# # #                 time_embed_dim,
# # #                 dropout,
# # #                 dims=dims,
# # #                 use_checkpoint=use_checkpoint,
# # #                 use_scale_shift_norm=use_scale_shift_norm,
# # #             ),
# # #             AttentionBlock(
# # #                 ch,
# # #                 use_checkpoint=use_checkpoint,
# # #                 num_heads=num_heads,
# # #                 num_head_channels=dim_head,
# # #                 use_new_attention_order=use_new_attention_order,
# # #             ) if not use_spatial_transformer else SpatialTransformer(  # always uses a self-attn
# # #                 ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
# # #                 disable_self_attn=disable_middle_self_attn, use_linear=use_linear_in_transformer,
# # #                 use_checkpoint=use_checkpoint
# # #             ),
# # #             ResBlock(
# # #                 ch,
# # #                 time_embed_dim,
# # #                 dropout,
# # #                 dims=dims,
# # #                 use_checkpoint=use_checkpoint,
# # #                 use_scale_shift_norm=use_scale_shift_norm,
# # #             ),
# # #         )
# # #         self.middle_block_out = self.make_zero_conv(ch)
# # #         self._feature_size += ch
# # #         # control_net_args = {'image_size': image_size, 'in_channels': in_channels, 'model_channels': model_channels, 'hint_channels': hint_channels, 'num_res_blocks': num_res_blocks, 'attention_resolutions': attention_resolutions,
# # #         #                     'dropout': dropout, 'channel_mult': channel_mult, 'conv_resample': conv_resample, 'dims': dims, 'use_checkpoint': use_checkpoint, 'use_fp16': use_fp16, 'num_heads': num_heads, 'num_head_channels': num_head_channels,
# # #         #                     'num_heads_upsample': num_heads_upsample, 'use_scale_shift_norm': use_scale_shift_norm,'resblock_updown': resblock_updown,'use_new_attention_order': use_new_attention_order, 'use_spatial_transformer': use_spatial_transformer,
# # #         #                     'transformer_depth': transformer_depth,'context_dim': context_dim,'n_embed': n_embed,'legacy': legacy,'disable_self_attentions': None,'num_attention_blocks': num_attention_blocks,'disable_middle_self_attn': disable_middle_self_attn,
# # #         #                     'use_linear_in_transformer': use_linear_in_transformer,'hyperconfig': hyperconfig,'stride': stride,'num_conditions': num_conditions,'embedding_dim': embedding_dim,}
# # #         # self.new_controlNet = ControlSingle(self.image_size, self.in_channels, self.model_channels, hint_channels, self.num_res_blocks, self.attention_resolutions, hyperconfig=hyperconfig, dims = self.dims, channel_mult=self.channel_mult,
# # #         #                                     num_heads = self.num_heads, num_heads_upsample=self.num_heads_upsample, stride=stride, legacy=legacy)
# # #         # if is_train==False: #and epoch_num:
# # #         cur_hypercond=0
# # #         self.new_controlNet_1 = ControlSingle(image_size=image_size, in_channels=in_channels, model_channels=model_channels, hint_channels=hint_channels, num_res_blocks=num_res_blocks, attention_resolutions=attention_resolutions,
# # #                                             dropout=dropout, channel_mult=channel_mult, conv_resample=conv_resample, dims=dims, use_checkpoint=use_checkpoint, use_fp16=use_fp16, num_heads=num_heads, num_head_channels=num_head_channels,
# # #                                             num_heads_upsample=num_heads_upsample, use_scale_shift_norm=use_scale_shift_norm,resblock_updown=resblock_updown,use_new_attention_order=use_new_attention_order, use_spatial_transformer=use_spatial_transformer,
# # #                                             transformer_depth=transformer_depth,context_dim=context_dim,n_embed=n_embed,legacy=legacy,disable_self_attentions=None,num_attention_blocks=num_attention_blocks,disable_middle_self_attn=disable_middle_self_attn,
# # #                                             use_linear_in_transformer=use_linear_in_transformer,hyperconfig=hyperconfig,stride=stride,num_conditions=num_conditions,embedding_dim=embedding_dim,hypercond=cur_hypercond)        
# # #         self.load_model_parts(self.new_controlNet_1, f'./multi_ckpt/0_10epoch.pt') 
# # #         self.new_controlNets = nn.ModuleList([self.new_controlNet_1])
        
# # #             # self.load_model_parts(self.new_controlNet_1, f'multi_ckpt/10epoch_{cur_hypercond}_512_10.pt')  #{epoch_num}
# # #             # self.load_model_parts(self.new_controlNet_1, f'multi_ckpt/10epoch_{cur_hypercond}.pt')
# # #             # cur_hypercond=2
# # #             # self.new_controlNet_2 = ControlSingle(image_size=image_size, in_channels=in_channels, model_channels=model_channels, hint_channels=hint_channels, num_res_blocks=num_res_blocks, attention_resolutions=attention_resolutions,
# # #             #                                     dropout=dropout, channel_mult=channel_mult, conv_resample=conv_resample, dims=dims, use_checkpoint=use_checkpoint, use_fp16=use_fp16, num_heads=num_heads, num_head_channels=num_head_channels,
# # #             #                                     num_heads_upsample=num_heads_upsample, use_scale_shift_norm=use_scale_shift_norm,resblock_updown=resblock_updown,use_new_attention_order=use_new_attention_order, use_spatial_transformer=use_spatial_transformer,
# # #             #                                     transformer_depth=transformer_depth,context_dim=context_dim,n_embed=n_embed,legacy=legacy,disable_self_attentions=None,num_attention_blocks=num_attention_blocks,disable_middle_self_attn=disable_middle_self_attn,
# # #             #                                     use_linear_in_transformer=use_linear_in_transformer,hyperconfig=hyperconfig,stride=stride,num_conditions=num_conditions,embedding_dim=embedding_dim,hypercond=cur_hypercond)        
# # #             # self.load_model_parts(self.new_controlNet_2, f'multi_ckpt/10epoch_{cur_hypercond}.pt')
# # #             # cur_hypercond=3
# # #             # self.new_controlNet_3 = ControlSingle(image_size=image_size, in_channels=in_channels, model_channels=model_channels, hint_channels=hint_channels, num_res_blocks=num_res_blocks, attention_resolutions=attention_resolutions,
# # #             #                                     dropout=dropout, channel_mult=channel_mult, conv_resample=conv_resample, dims=dims, use_checkpoint=use_checkpoint, use_fp16=use_fp16, num_heads=num_heads, num_head_channels=num_head_channels,
# # #             #                                     num_heads_upsample=num_heads_upsample, use_scale_shift_norm=use_scale_shift_norm,resblock_updown=resblock_updown,use_new_attention_order=use_new_attention_order, use_spatial_transformer=use_spatial_transformer,
# # #             #                                     transformer_depth=transformer_depth,context_dim=context_dim,n_embed=n_embed,legacy=legacy,disable_self_attentions=None,num_attention_blocks=num_attention_blocks,disable_middle_self_attn=disable_middle_self_attn,
# # #             #                                     use_linear_in_transformer=use_linear_in_transformer,hyperconfig=hyperconfig,stride=stride,num_conditions=num_conditions,embedding_dim=embedding_dim,hypercond=cur_hypercond)        
# # #             # self.load_model_parts(self.new_controlNet_3, f'multi_ckpt/10epoch_{cur_hypercond}.pt')
# # #             # self.new_controlNets = nn.ModuleList([self.new_controlNet_1]) #, self.new_controlNet_2, self.new_controlNet_3])

# # #     def make_zero_conv(self, channels):
# # #         return TimestepEmbedSequential(zero_module(conv_nd(self.dims, channels, channels, 1, padding=0)))

# # #     def save_model_parts(self, model, filepath):
# # #         state_dict = {
# # #             'input_hint_block_new': model.input_hint_block_new.state_dict(),
# # #             'input_blocks': model.input_blocks.state_dict(),
# # #             'zero_convs': model.zero_convs.state_dict(),
# # #             'middle_block': model.middle_block.state_dict(),
# # #             'middle_block_out': model.middle_block_out.state_dict()
# # #         }
# # #         torch.save(state_dict, filepath)

# # #     def load_model_parts(self, model, filepath):
# # #         state_dict = torch.load(filepath)
# # #         model.input_hint_block_new.load_state_dict(state_dict['input_hint_block_new'])
# # #         model.input_blocks.load_state_dict(state_dict['input_blocks'])
# # #         model.zero_convs.load_state_dict(state_dict['zero_convs'])
# # #         model.middle_block.load_state_dict(state_dict['middle_block'])
# # #         model.middle_block_out.load_state_dict(state_dict['middle_block_out'])

# # #     def forward(self, x, hint, timesteps, context ,hyper_scales=None,PCA=None, control_scale=None,training=False,multi=False,**kwargs):
# # #         t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
# # #         emb = self.time_embed(t_emb)
# # #         if training==False:
# # #             if multi==True:
# # #                 self.hypercolumn.cond = [0, 1]
# # #                 hs = [[] for _ in range(len(self.input_blocks) + 1)]
# # #                 outs = []
# # #                 for i in self.hypercolumn.cond:
# # #                     h = x.type(self.dtype)
# # #                     if i == 0:
# # #                         guided_hint = self.input_hint_block_new(hint, emb, context,hyper_scale=hyper_scales[i], cond=i)
# # #                         for j, (module, zero_conv) in enumerate(zip(self.input_blocks, self.zero_convs)):
# # #                             if guided_hint is not None:
# # #                                 h = module(h, emb, context)
# # #                                 h += guided_hint
# # #                                 guided_hint = None
# # #                             else:
# # #                                 h = module(h, emb, context)
# # #                             hs[j].append(zero_conv(h, emb, context))
# # #                         h = self.middle_block(h, emb, context)    
# # #                         hs[-1].append(self.middle_block_out(h, emb, context))
# # #                     else:
# # #                         model = self.new_controlNets[i-1]
# # #                         guided_hint = model.input_hint_block_new(hint, emb, context,hyper_scale=hyper_scales[i], cond=i)
# # #                         for j, (module, zero_conv) in enumerate(zip(model.input_blocks, model.zero_convs)):
# # #                             if guided_hint is not None:
# # #                                 h = module(h, emb, context)
# # #                                 h += guided_hint
# # #                                 guided_hint = None
# # #                             else:
# # #                                 h = module(h, emb, context)
# # #                             hs[j].append(zero_conv(h, emb, context))
# # #                         h = model.middle_block(h, emb, context)
# # #                         hs[-1].append(self.middle_block_out(h, emb, context))
# # #                 for i in range(len(hs)):
# # #                     outs.append(control_scale[0]*hs[i][0]+ control_scale[1]*hs[i][1])
# # #             else:
                
# # #                 guided_hint = self.input_hint_block_new(hint, emb, context, hyper_scale=hyper_scales, cond=self.hypercolumn.cond[0])
# # #                 outs = []
# # #                 h = x.type(self.dtype)
# # #                 for j, (module, zero_conv) in enumerate(zip(self.input_blocks, self.zero_convs)):
# # #                     if guided_hint is not None:
# # #                         h = module(h, emb, context)
# # #                         h += guided_hint
# # #                         guided_hint = None
# # #                     else:
# # #                         h = module(h, emb, context)
# # #                     outs.append(zero_conv(h, emb, context))
# # #                 h = self.middle_block(h, emb, context)
# # #                 outs.append(self.middle_block_out(h, emb, context))
# # #         else:
# # #             guided_hint = self.input_hint_block_new(hint, emb, context, hyper_scale=1, cond=self.hypercolumn.cond[0])
# # #             outs = []
# # #             h = x.type(self.dtype)
# # #             for j, (module, zero_conv) in enumerate(zip(self.input_blocks, self.zero_convs)):
# # #                 if guided_hint is not None:
# # #                     h = module(h, emb, context)
# # #                     h += guided_hint
# # #                     guided_hint = None
# # #                 else:
# # #                     h = module(h, emb, context)
# # #                 outs.append(zero_conv(h, emb, context))
# # #             h = self.middle_block(h, emb, context)
# # #             outs.append(self.middle_block_out(h, emb, context))
# # #         return outs   

# # # class ControlLDM(LatentDiffusion):

# # #     def __init__(self, control_stage_config, control_key, only_mid_control,control_step=0, *args, **kwargs):
# # #         super().__init__(*args, **kwargs)
# # #         self.control_model = instantiate_from_config(control_stage_config)
# # #         self.control_key = control_key
# # #         self.only_mid_control = only_mid_control
# # #         self.control_scales = [1.] * 13 
# # #         self.control_step = control_step
# # #         if isinstance(self.control_model.hypercolumn,HyperColumnLGN):
# # #             self.hypercond = control_stage_config['params']['hyperconfig']['params']['hypercond']
# # #         else:
# # #             self.hypercond = [0]
# # #         self.select()
    
# # #     def training_step(self, batch, batch_idx):
# # #         return super().training_step(batch, batch_idx)

# # #     @torch.no_grad()
# # #     def get_input(self, batch, k, bs=None, *args, **kwargs):
# # #         x, c = super().get_input(batch, self.first_stage_key, *args, **kwargs)
# # #         control = batch[self.control_key]
# # #         # print('c:',c)
        
# # #         if bs is not None:
# # #             control = control[:bs]
# # #         control = control.to(self.device)
# # #         control = einops.rearrange(control, 'b h w c -> b c h w')
# # #         control = control.to(memory_format=torch.contiguous_format).float()
# # #         return x, dict(c_crossattn=[c], c_concat=[control])

# # #     def apply_model(self, x_noisy, t, cond, hyper_scales=None, PCA=None, control_scale=None,training=False,multi=False,*args, **kwargs):
# # #         assert isinstance(cond, dict)
# # #         diffusion_model = self.model.diffusion_model
# # #         # print('t:',t)

# # #         cond_txt = torch.cat(cond['c_crossattn'], 1)

# # #         if cond['c_concat'] is None:
# # #             eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=None, only_mid_control=self.only_mid_control)
# # #         else:
# # #             control = self.control_model(x=x_noisy, hint=torch.cat(cond['c_concat'], 1), timesteps=t, context=cond_txt, group_num=0, hyper_scales=hyper_scales,control_scale=control_scale,training=training,multi=multi)
# # #             if t[0]<self.control_step:
# # #                 control = [c * 0. for c, scale in zip(control, self.control_scales)]
# # #             else:
# # #                 control = [c * scale for c, scale in zip(control, self.control_scales)]
# # #             eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=control, only_mid_control=self.only_mid_control)
# # #             # controls = self.control_model(x=x_noisy, hint=torch.cat(cond['c_concat'], 1), timesteps=t, context=cond_txt, group_num=0)
# # #             # epss = []
# # #             # for i in range(len(controls[0])):
# # #             #     control = [item[i] for item in controls]
# # #             #     if t[0]<self.control_step:
# # #             #         control = [c * 0. for c, scale in zip(control, self.control_scales)]
# # #             #     else:
# # #             #         control = [c * scale for c, scale in zip(control, self.control_scales)]
# # #             #     eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=control, only_mid_control=self.only_mid_control)
# # #             #     epss.append(eps)
# # #             # eps = epss[0] #0.7*epss[0] + 0.1*epss[1] + 0.1*epss[2] + 0.1*epss[3]  #torch.mean(torch.stack(epss), dim=0)  #0.7*epss[0] + 0.1*epss[1] + 0.1*epss[2] + 0.1*epss[3]

# # #         return eps
    
# # #     def apply_model_train(self, x_noisy, t, cond, *args, **kwargs):
# # #         assert isinstance(cond, dict)
# # #         diffusion_model = self.model.diffusion_model
# # #         # print('t:',t)

# # #         cond_txt = torch.cat(cond['c_crossattn'], 1)

# # #         if cond['c_concat'] is None:
# # #             eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=None, only_mid_control=self.only_mid_control)
# # #         else:
# # #             control = self.control_model(x=x_noisy, hint=torch.cat(cond['c_concat'], 1), timesteps=t, context=cond_txt,training=True)
# # #             if t[0]<self.control_step:
# # #                 control = [c * 0. for c, scale in zip(control, self.control_scales)]
# # #             else:
# # #                 control = [c * scale for c, scale in zip(control, self.control_scales)]
# # #             eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=control, only_mid_control=self.only_mid_control)
# # #         return eps    

# # #     @torch.no_grad()
# # #     def get_unconditional_conditioning(self, N):
# # #         return self.get_learned_conditioning([""] * N)

# # #     @torch.no_grad()
# # #     def log_images(self, batch, N=4, n_row=2, sample=False, ddim_steps=50, ddim_eta=0.0, return_keys=None,
# # #                    quantize_denoised=True, inpaint=True, plot_denoise_rows=False, plot_progressive_rows=True,
# # #                    plot_diffusion_rows=False, unconditional_guidance_scale=7.5, unconditional_guidance_label=None,
# # #                    use_ema_scope=True,
# # #                    **kwargs):
# # #         use_ddim = ddim_steps is not None

# # #         log = dict()
# # #         z, c = self.get_input(batch, self.first_stage_key, bs=N)
# # #         c_cat, c = c["c_concat"][0][:N], c["c_crossattn"][0][:N]
# # #         N = min(z.shape[0], N)
# # #         n_row = min(z.shape[0], n_row)
# # #         log["reconstruction"] = self.decode_first_stage(z)
# # #         log["control"] = c_cat * 2.0 - 1.0
# # #         log["conditioning"] = log_txt_as_img((512, 512), batch[self.cond_stage_key], size=16)

# # #         if plot_diffusion_rows:
# # #             # get diffusion row
# # #             diffusion_row = list()
# # #             z_start = z[:n_row]
# # #             for t in range(self.num_timesteps):
# # #                 if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
# # #                     t = repeat(torch.tensor([t]), '1 -> b', b=n_row)
# # #                     t = t.to(self.device).long()
# # #                     noise = torch.randn_like(z_start)
# # #                     z_noisy = self.q_sample(x_start=z_start, t=t, noise=noise)
# # #                     diffusion_row.append(self.decode_first_stage(z_noisy))

# # #             diffusion_row = torch.stack(diffusion_row)  # n_log_step, n_row, C, H, W
# # #             diffusion_grid = rearrange(diffusion_row, 'n b c h w -> b n c h w')
# # #             diffusion_grid = rearrange(diffusion_grid, 'b n c h w -> (b n) c h w')
# # #             diffusion_grid = make_grid(diffusion_grid, nrow=diffusion_row.shape[0])
# # #             log["diffusion_row"] = diffusion_grid

# # #         if sample:
# # #             # get denoise row
# # #             samples, z_denoise_row = self.sample_log(cond={"c_concat": [c_cat], "c_crossattn": [c]},
# # #                                                      batch_size=N, ddim=use_ddim,
# # #                                                      ddim_steps=ddim_steps, eta=ddim_eta)
# # #             x_samples = self.decode_first_stage(samples)
# # #             log["samples"] = x_samples
# # #             if plot_denoise_rows:
# # #                 denoise_grid = self._get_denoise_row_from_list(z_denoise_row)
# # #                 log["denoise_row"] = denoise_grid

# # #         if unconditional_guidance_scale >= -1.0:
# # #             uc_cross = self.get_unconditional_conditioning(N)
# # #             uc_cat = c_cat  # torch.zeros_like(c_cat)
# # #             uc_full = {"c_concat": [uc_cat], "c_crossattn": [uc_cross]}
# # #             for slct in self.slct:
# # #                 self.control_model.input_hint_block_new[0].cond = slct['cond']
# # #                 suffix = slct['suffix']
# # #                 print(suffix)
# # #                 samples_cfg, _ = self.sample_log(cond={"c_concat": [c_cat], "c_crossattn": [c]},
# # #                                                 batch_size=N, ddim=use_ddim,
# # #                                                 ddim_steps=ddim_steps, eta=ddim_eta,
# # #                                                 unconditional_guidance_scale=unconditional_guidance_scale,
# # #                                                 unconditional_conditioning=uc_full,training=True
# # #                                                 )
# # #                 x_samples_cfg = self.decode_first_stage(samples_cfg)
# # #                 deconv = self.control_model.input_hint_block_new[0].deconv(c_cat)
# # #                 log[f"{suffix}_samples_cfg_scale_{unconditional_guidance_scale:.2f}"] = x_samples_cfg
# # #                 log[f"{suffix}_deconv_samples_cfg_scale_{unconditional_guidance_scale:.2f}"] = deconv
# # #                 # self.control_model.input_hint_block_new[0].cond = None

# # #         return log

# # #     @torch.no_grad()
# # #     def sample_log(self, cond, batch_size, ddim, ddim_steps, training=False,**kwargs):
# # #         ddim_sampler = DDIMSampler(self)
# # #         b, c, h, w = cond["c_concat"][0].shape
# # #         shape = (self.channels, h // 8, w // 8)
# # #         samples, intermediates = ddim_sampler.sample(ddim_steps, batch_size, shape, cond, verbose=False,training=training, **kwargs)
# # #         return samples, intermediates

# # #     def configure_optimizers(self):
# # #         lr = self.learning_rate
# # #         params = list(self.control_model.parameters())
# # #         if not self.sd_locked:
# # #             params += list(self.model.diffusion_model.output_blocks.parameters())
# # #             params += list(self.model.diffusion_model.out.parameters())
# # #         opt = torch.optim.AdamW(params, lr=lr)
# # #         return opt

# # #     def low_vram_shift(self, is_diffusing):
# # #         if is_diffusing:
# # #             self.model = self.model.cuda()
# # #             self.control_model = self.control_model.cuda()
# # #             self.first_stage_model = self.first_stage_model.cpu()
# # #             self.cond_stage_model = self.cond_stage_model.cpu()
# # #         else:
# # #             self.model = self.model.cpu()
# # #             self.control_model = self.control_model.cpu()
# # #             self.first_stage_model = self.first_stage_model.cuda()
# # #             self.cond_stage_model = self.cond_stage_model.cuda()

# # #     @torch.no_grad()
# # #     def select(self):
# # #         self.slct = [{'cond':[i],'suffix':f'{i}'} for i in self.hypercond]

# import einops
# import torch
# import torch as th
# import torch.nn as nn
# import torchvision.transforms as transforms
# import sys
# import numpy as np
# import random
# from einops import rearrange, repeat,reduce
# from torchvision import transforms

# # sys.path.append('/home/chenzhiqiang/code/')
# sys.path.append('/home/uchihawdt/control-net-main-v0.5')
# from hypercolumn.vit_pytorch.train_V1_sep_new import Column_trans_rot_lgn

# from ldm.modules.diffusionmodules.util import (
#     conv_nd,
#     linear,
#     zero_module,
#     timestep_embedding,
# )

# from einops import rearrange, repeat
# from torchvision.utils import make_grid
# from ldm.modules.attention import SpatialTransformer
# from ldm.modules.diffusionmodules.openaimodel import UNetModel, TimestepEmbedSequential, ResBlock, Downsample, AttentionBlock
# from ldm.models.diffusion.ddpm import LatentDiffusion
# from ldm.util import log_txt_as_img, exists, instantiate_from_config
# from ldm.models.diffusion.ddim import DDIMSampler


# class ControlledUnetModel(UNetModel):
#     def forward(self, x, timesteps=None, context=None, control=None, only_mid_control=False, **kwargs):
#         hs = []
#         with torch.no_grad():
#             t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
#             emb = self.time_embed(t_emb)
#             h = x.type(self.dtype)
#             for module in self.input_blocks:
#                 h = module(h, emb, context)
#                 hs.append(h)
#             h = self.middle_block(h, emb, context)

#         if control is not None:
#             h += control.pop()

#         for i, module in enumerate(self.output_blocks):
#             if only_mid_control or control is None:
#                 h = torch.cat([h, hs.pop()], dim=1)
#             else:
#                 h = torch.cat([h, hs.pop() + control.pop()], dim=1)
#             h = module(h, emb, context)

#         h = h.type(x.dtype)
#         return self.out(h)

# def disabled_train(self, mode=True):
#     """Overwrite model.train with this function to make sure train/eval mode
#     does not change anymore."""
#     return self


# class HyperColumnLGN(nn.Module):
#     def __init__(self,key=0,hypercond=[0],size=512,restore_ckpt = '/home/uchihawdt/control-net-main-v0.5/hypercolumn/checkpoint/imagenet/equ_nv16_vl4_rn1_Bipolar_norm.pth'):
#         super().__init__()
#         ckpt = torch.load(restore_ckpt)
#         hc = Column_trans_rot_lgn(ckpt['arg'])
#         hc.load_state_dict(ckpt['state_dict'], strict=False)
#         self.lgn_ende = hc.lgn_ende[0].eval()
#         self.lgn_ende.train = disabled_train
#         for param in self.lgn_ende.parameters():
#             param.requires_grad = False

#         self.resize = transforms.Resize(size)
#         if size == 128:
#             self.pad = nn.ConstantPad2d((1,1,1,1),0.)
#         else:
#             self.pad = nn.Identity()
        
#         self.groups = [[0,1,4,8,9,15],[2,3],[5,12],[10],[6,7],[11,13,14],[5],[12],[2],[3]]
#         self.p = [0. for i in range(len(self.groups))]
#         self.p[0] = 1.

#         norm_mean = np.array([0.50705882, 0.48666667, 0.44078431])
#         norm_std = np.array([0.26745098, 0.25568627, 0.27607843])
#         self.norm = transforms.Normalize(norm_mean, norm_std)
#         self.cond = hypercond
#         self.slct = None
    
#     def forward(self,x, hyper_scale=1, cond=None):
#         x = self.resize(x)
#         s = x.size()
#         r = torch.zeros(1,16,1,1).to(x.device)
#         cond = cond if cond!=None else self.cond

#         if cond is None:
#             c = [i for i in range(len(self.groups))]
#             random.shuffle(c)
#             # print(self.groups[c[0]])
#             pa = random.random()
#             for i in range(len(self.groups)):
#                 if pa < self.p[i]:
#                     for j in self.groups[c[i]]:
#                         r[:,j,:,:] = 1.
#         else:
#             # for i in cond:
#             #     for j in self.groups[i]:
#             #         r[:,j,:,:] = 1
#             for j in self.groups[cond]:
#                 r[:,j,:,:] = 1

#         r = repeat(r,'n c h w -> n (c repeat) h w',repeat=4)
#         out = self.lgn_ende(self.norm(x))*r
#         out = self.pad(self.lgn_ende.deconv(out))
#         # out *= hyper_scale
#         return out
    
#     def deconv(self,x):
#         x = self.resize(x)
#         s = x.size()
#         r = torch.zeros(1,16,1,1).to(x.device)
#         if self.cond is not None:
#             for i in self.cond:
#                 for j in self.groups[i]:
#                     r[:,j,:,:] = 1

#         r = repeat(r,'n c h w -> n (c repeat) h w',repeat=4)
#         conv = self.lgn_ende(self.norm(x))*r
#         deconv = self.lgn_ende.deconv(conv)
#         deconv = (deconv - reduce(deconv,'b c h w -> b c () ()', 'min'))/(reduce(deconv,'b c h w -> b c () ()', 'max')-reduce(deconv,'b c h w -> b c () ()', 'min'))*2.-1.

#         return deconv

# class HyperColumnLGNVisual(nn.Module):
#     def __init__(self,key=0,hypercond=[0],size=512,restore_ckpt = '/home/uchihawdt/control-net-main-v0.5/hypercolumn/checkpoint/imagenet/equ_nv16_vl4_rn1_Bipolar_norm.pth'):
#         super().__init__()
#         ckpt = torch.load(restore_ckpt)
#         hc = Column_trans_rot_lgn(ckpt['arg'])
#         hc.load_state_dict(ckpt['state_dict'], strict=False)
#         self.lgn_ende = hc.lgn_ende[0].eval()
#         self.lgn_ende.train = disabled_train
#         for param in self.lgn_ende.parameters():
#             param.requires_grad = False

#         self.resize = transforms.Resize(size)
#         if size == 128:
#             self.pad = nn.ConstantPad2d((1,1,1,1),0.)
#         else:
#             self.pad = nn.Identity()
        

#         # self.groups = [[2,3],[0,1,4,8,9,15],[5,6,7,10,12],[11,13,14]]
#         self.groups = [[0,1,4,8,9,15],[2,3],[5,12],[10],[6,7],[11,13,14],[5],[12],[2],[3]]
#         # self.groups = [[2],[3],[5],[12]]
            
#         # self.p = [1.,0.5,0.25,0.125]
#         self.p = [0. for i in range(len(self.groups))]
#         self.p[0] = 1.

#         norm_mean = np.array([0.50705882, 0.48666667, 0.44078431])
#         norm_std = np.array([0.26745098, 0.25568627, 0.27607843])
#         self.norm = transforms.Normalize(norm_mean, norm_std)
#         self.cond = hypercond
#         self.slct = None

#     # def forward(self,x):
#     #     s = x.size()
#     #     r = torch.zeros(1,16,1,1).to(x.device)

#     #     if self.cond is None:
#     #         c = [i for i in range(len(self.groups))]
#     #         random.shuffle(c)
#     #         # print(self.groups[c[0]])
#     #         pa = random.random()
#     #         for i in range(4):
#     #             if pa < self.p[i]:
#     #                 for j in self.groups[c[i]]:
#     #                     r[:,j,:,:] = 1.
#     #     else:
#     #         for i in self.cond:
#     #             for j in self.groups[i]:
#     #                 r[:,j,:,:] = 1

#     #     r = repeat(r,'n c h w -> n (c repeat) h w',repeat=4)
#     #     return self.lgn_ende(self.norm(x))*r
    
#     def forward(self,x):
#         x = self.resize(x)
#         s = x.size()
#         r = torch.zeros(1,16,1,1).to(x.device)

#         if self.cond is None:
#             c = [i for i in range(len(self.groups))]
#             random.shuffle(c)
#             # print(self.groups[c[0]])
#             pa = random.random()
#             for i in range(len(self.groups)):
#                 if pa < self.p[i]:
#                     for j in self.groups[c[i]]:
#                         r[:,j,:,:] = 1.
#         else:
#             for i in self.cond:
#                 for j in self.groups[i]:
#                     r[:,j,:,:] = 1

#         # r = repeat(r,'n c h w -> n (c repeat) h w',repeat=4)
#         # out = self.lgn_ende(self.norm(x))*r
#         # # print('out_conv:',out.size())
#         # out = self.pad(self.lgn_ende.deconv(out))
#         # # out = self.lgn_ende.deconv(self.lgn_ende(self.norm(x))*r)
#         # # print('out:',out.size())
#         # return out

#         out = self.lgn_ende(self.norm(x))
#         # out = rearrange(out, 'b (c n) h w -> b c (n h) w',n=16)
#         out = rearrange(out, 'b (n c) h w -> b c (n h) w', n=16)
#         # out = self.pad(self.lgn_ende.deconv(out))
#         return out
    
#     def deconv(self,x):
#         x = self.resize(x)
#         s = x.size()
#         r = torch.zeros(1,16,1,1).to(x.device)
#         if self.cond is not None:
#             for i in self.cond:
#                 for j in self.groups[i]:
#                     r[:,j,:,:] = 1

#         r = repeat(r,'n c h w -> n (c repeat) h w',repeat=4)
#         conv = self.lgn_ende(self.norm(x))*r
#         deconv = self.lgn_ende.deconv(conv)
#         deconv = (deconv - reduce(deconv,'b c h w -> b c () ()', 'min'))/(reduce(deconv,'b c h w -> b c () ()', 'max')-reduce(deconv,'b c h w -> b c () ()', 'min'))*2.-1.

#         return deconv
    
# class HyperColumnLGNFeature(nn.Module):
#     def __init__(self,key=0,hypercond=[0],size=512,restore_ckpt = '/home/uchihawdt/control-net-main-v0.5/hypercolumn/checkpoint/imagenet/equ_nv16_vl4_rn1_Bipolar_norm.pth'):
#         super().__init__()
#         ckpt = torch.load(restore_ckpt)
#         hc = Column_trans_rot_lgn(ckpt['arg'])
#         hc.load_state_dict(ckpt['state_dict'], strict=False)
#         self.lgn_ende = hc.lgn_ende[0].eval()
#         self.lgn_ende.train = disabled_train
#         for param in self.lgn_ende.parameters():
#             param.requires_grad = False

#         self.resize = transforms.Resize(size)
#         if size == 128:
#             self.pad = nn.ConstantPad2d((1,1,1,1),0.)
#         else:
#             self.pad = nn.Identity()
        

#         # self.groups = [[2,3],[0,1,4,8,9,15],[5,6,7,10,12],[11,13,14]]
#         self.groups = [[0,1,4,8,9,15],[2,3],[5,12],[10],[6,7],[11,13,14],[5],[12],[2],[3]]
#         # self.groups = [[2],[3],[5],[12]]
            
#         # self.p = [1.,0.5,0.25,0.125]
#         self.p = [0. for i in range(len(self.groups))]
#         self.p[0] = 1.

#         norm_mean = np.array([0.50705882, 0.48666667, 0.44078431])
#         norm_std = np.array([0.26745098, 0.25568627, 0.27607843])
#         self.norm = transforms.Normalize(norm_mean, norm_std)
#         self.cond = hypercond
#         self.slct = None
    
#     def forward(self,x):
#         x = self.resize(x)
#         s = x.size()
#         # r = torch.zeros(1,16,1,1).to(x.device)

#         # if self.cond is None:
#         #     c = [i for i in range(len(self.groups))]
#         #     random.shuffle(c)
#         #     # print(self.groups[c[0]])
#         #     pa = random.random()
#         #     for i in range(len(self.groups)):
#         #         if pa < self.p[i]:
#         #             for j in self.groups[c[i]]:
#         #                 r[:,j,:,:] = 1.
#         # else:
#         #     for i in self.cond:
#         #         for j in self.groups[i]:
#         #             r[:,j,:,:] = 1

#         # r = repeat(r,'n c h w -> n (c repeat) h w',repeat=4)
#         out = self.lgn_ende(self.norm(x))
#         out = out[:,40:44,:,:]
#         # out = self.pad(self.lgn_ende.deconv(out))
#         return out

#     def deconv(self,x):
#         x = self.resize(x)
#         s = x.size()
#         r = torch.zeros(1,16,1,1).to(x.device)
#         if self.cond is not None:
#             for i in self.cond:
#                 for j in self.groups[i]:
#                     r[:,j,:,:] = 1

#         r = repeat(r,'n c h w -> n (c repeat) h w',repeat=4)
#         conv = self.lgn_ende(self.norm(x))*r
#         deconv = self.lgn_ende.deconv(conv)
#         deconv = (deconv - reduce(deconv,'b c h w -> b c () ()', 'min'))/(reduce(deconv,'b c h w -> b c () ()', 'max')-reduce(deconv,'b c h w -> b c () ()', 'min'))*2.-1.

#         return deconv


# class Canny(nn.Module):
#     def __init__(self,para=None):
#         super().__init__()
#     def forward(self,x):
#         return x
#     def deconv(self,x):
#         return x


# class ConditionEmbedding(nn.Module):
#     def __init__(self, num_conditions, embedding_dim):
#         super(ConditionEmbedding, self).__init__()
#         self.embedding = nn.Embedding(num_conditions, embedding_dim)

#     def forward(self, condition_type):
#         return self.embedding(condition_type)

# class ControlSingle(nn.Module):
#     def __init__(self,image_size, in_channels, model_channels, hint_channels, num_res_blocks, attention_resolutions, dropout=0, channel_mult=(1, 2, 4, 8), conv_resample=True, dims=2, use_checkpoint=False, use_fp16=False, num_heads=-1, num_head_channels=-1, num_heads_upsample=-1, use_scale_shift_norm=False, resblock_updown=False, use_new_attention_order=False, use_spatial_transformer=False, transformer_depth=1, context_dim=None, n_embed=None, legacy=True, disable_self_attentions=None, num_attention_blocks=None, disable_middle_self_attn=False, use_linear_in_transformer=False, hyperconfig=None, stride=[2,2,2], num_conditions=6, embedding_dim=128,hypercond=0):
#         super().__init__()
#         self.dims = dims
#         hyperconfig['params']['hypercond'] = [hypercond]
#         self.hypercolumn = instantiate_from_config(hyperconfig)
#         self.input_blocks = nn.ModuleList(
#             [
#                 TimestepEmbedSequential(
#                     conv_nd(dims, in_channels, model_channels, 3, padding=1)
#                 )
#             ]
#         )
#         self.zero_convs = nn.ModuleList([self.make_zero_conv(model_channels)])

#         if isinstance(num_res_blocks, int):
#             self.num_res_blocks = len(channel_mult) * [num_res_blocks]
#         else:
#             if len(num_res_blocks) != len(channel_mult):
#                 raise ValueError("provide num_res_blocks either as an int (globally constant) or "
#                                  "as a list/tuple (per-level) with the same length as channel_mult")
#             self.num_res_blocks = num_res_blocks
        
#         time_embed_dim = model_channels * 4
#         self.input_hint_block_new = TimestepEmbedSequential(self.hypercolumn,
#                                                             conv_nd(dims, hint_channels, 16, 3, padding=1),nn.SiLU(),
#                                                             conv_nd(dims, 16, 16, 3, padding=1),nn.SiLU(),
#                                                             conv_nd(dims, 16, 32, 3, padding=1, stride=stride[0]),nn.SiLU(),
#                                                             conv_nd(dims, 32, 32, 3, padding=1),nn.SiLU(),
#                                                             conv_nd(dims, 32, 96, 3, padding=1, stride=stride[1]),nn.SiLU(),
#                                                             conv_nd(dims, 96, 96, 3, padding=1),nn.SiLU(),
#                                                             conv_nd(dims, 96, 256, 3, padding=1, stride=stride[2]),nn.SiLU(),
#                                                             zero_module(conv_nd(dims, 256, model_channels, 3, padding=1)))
#         self._feature_size = model_channels
#         input_block_chans = [model_channels]
#         ch = model_channels
#         ds = 1
#         for level, mult in enumerate(channel_mult):
#             for nr in range(self.num_res_blocks[level]):
#                 layers = [
#                     ResBlock(
#                         ch,
#                         time_embed_dim,
#                         dropout,
#                         out_channels=mult * model_channels,
#                         dims=dims,
#                         use_checkpoint=use_checkpoint,
#                         use_scale_shift_norm=use_scale_shift_norm,
#                     )
#                 ]
#                 ch = mult * model_channels
#                 if ds in attention_resolutions:
#                     if num_head_channels == -1:
#                         dim_head = ch // num_heads
#                     else:
#                         num_heads = ch // num_head_channels
#                         dim_head = num_head_channels
#                     if legacy:
#                         # num_heads = 1
#                         dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
#                     if exists(disable_self_attentions):
#                         disabled_sa = disable_self_attentions[level]
#                     else:
#                         disabled_sa = False

#                     if not exists(num_attention_blocks) or nr < num_attention_blocks[level]:
#                         layers.append(
#                             AttentionBlock(
#                                 ch,
#                                 use_checkpoint=use_checkpoint,
#                                 num_heads=num_heads,
#                                 num_head_channels=dim_head,
#                                 use_new_attention_order=use_new_attention_order,
#                             ) if not use_spatial_transformer else SpatialTransformer(
#                                 ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
#                                 disable_self_attn=disabled_sa, use_linear=use_linear_in_transformer,
#                                 use_checkpoint=use_checkpoint
#                             )
#                         )
#                 self.input_blocks.append(TimestepEmbedSequential(*layers))
#                 self.zero_convs.append(self.make_zero_conv(ch))
#                 self._feature_size += ch
#                 input_block_chans.append(ch)
#             if level != len(channel_mult) - 1:
#                 out_ch = ch
#                 self.input_blocks.append(
#                     TimestepEmbedSequential(
#                         ResBlock(
#                             ch,
#                             time_embed_dim,
#                             dropout,
#                             out_channels=out_ch,
#                             dims=dims,
#                             use_checkpoint=use_checkpoint,
#                             use_scale_shift_norm=use_scale_shift_norm,
#                             down=True,
#                         )
#                         if resblock_updown
#                         else Downsample(
#                             ch, conv_resample, dims=dims, out_channels=out_ch
#                         )
#                     )
#                 )
#                 ch = out_ch
#                 input_block_chans.append(ch)
#                 self.zero_convs.append(self.make_zero_conv(ch))
#                 ds *= 2
#                 self._feature_size += ch

#         if num_head_channels == -1:
#             dim_head = ch // num_heads
#         else:
#             num_heads = ch // num_head_channels
#             dim_head = num_head_channels
#         if legacy:
#             # num_heads = 1
#             dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
#         self.middle_block = TimestepEmbedSequential(
#             ResBlock(
#                 ch,
#                 time_embed_dim,
#                 dropout,
#                 dims=dims,
#                 use_checkpoint=use_checkpoint,
#                 use_scale_shift_norm=use_scale_shift_norm,
#             ),
#             AttentionBlock(
#                 ch,
#                 use_checkpoint=use_checkpoint,
#                 num_heads=num_heads,
#                 num_head_channels=dim_head,
#                 use_new_attention_order=use_new_attention_order,
#             ) if not use_spatial_transformer else SpatialTransformer(  # always uses a self-attn
#                 ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
#                 disable_self_attn=disable_middle_self_attn, use_linear=use_linear_in_transformer,
#                 use_checkpoint=use_checkpoint
#             ),
#             ResBlock(
#                 ch,
#                 time_embed_dim,
#                 dropout,
#                 dims=dims,
#                 use_checkpoint=use_checkpoint,
#                 use_scale_shift_norm=use_scale_shift_norm,
#             ),
#         )
#         self.middle_block_out = self.make_zero_conv(ch)
#     def make_zero_conv(self, channels):
#         return TimestepEmbedSequential(zero_module(conv_nd(self.dims, channels, channels, 1, padding=0)))

# class ControlNet(nn.Module):
#     def __init__(
#             self,
#             image_size,
#             in_channels,
#             model_channels,
#             hint_channels,
#             num_res_blocks,
#             attention_resolutions,
#             dropout=0,
#             channel_mult=(1, 2, 4, 8),
#             conv_resample=True,
#             dims=2,
#             use_checkpoint=False,
#             use_fp16=False,
#             num_heads=-1,
#             num_head_channels=-1,
#             num_heads_upsample=-1,
#             use_scale_shift_norm=False,
#             resblock_updown=False,
#             use_new_attention_order=False,
#             use_spatial_transformer=False,  # custom transformer support
#             transformer_depth=1,  # custom transformer support
#             context_dim=None,  # custom transformer support
#             n_embed=None,  # custom support for prediction of discrete ids into codebook of first stage vq model
#             legacy=True,
#             disable_self_attentions=None,
#             num_attention_blocks=None,
#             disable_middle_self_attn=False,
#             use_linear_in_transformer=False,
#             hyperconfig=None,
#             stride=[2,2,2],
#             num_conditions=6, 
#             embedding_dim=128,
#             is_train=False,
#             epoch_num=None,
#     ):
#         super().__init__()
#         if use_spatial_transformer:
#             assert context_dim is not None, 'Fool!! You forgot to include the dimension of your cross-attention conditioning...'

#         if context_dim is not None:
#             assert use_spatial_transformer, 'Fool!! You forgot to use the spatial transformer for your cross-attention conditioning...'
#             from omegaconf.listconfig import ListConfig
#             if type(context_dim) == ListConfig:
#                 context_dim = list(context_dim)

#         if num_heads_upsample == -1:
#             num_heads_upsample = num_heads

#         if num_heads == -1:
#             assert num_head_channels != -1, 'Either num_heads or num_head_channels has to be set'

#         if num_head_channels == -1:
#             assert num_heads != -1, 'Either num_heads or num_head_channels has to be set'

#         self.dims = dims
#         self.image_size = image_size
#         self.in_channels = in_channels
#         self.model_channels = model_channels
#         if isinstance(num_res_blocks, int):
#             self.num_res_blocks = len(channel_mult) * [num_res_blocks]
#         else:
#             if len(num_res_blocks) != len(channel_mult):
#                 raise ValueError("provide num_res_blocks either as an int (globally constant) or "
#                                  "as a list/tuple (per-level) with the same length as channel_mult")
#             self.num_res_blocks = num_res_blocks
#         if disable_self_attentions is not None:
#             # should be a list of booleans, indicating whether to disable self-attention in TransformerBlocks or not
#             assert len(disable_self_attentions) == len(channel_mult)
#         if num_attention_blocks is not None:
#             assert len(num_attention_blocks) == len(self.num_res_blocks)
#             assert all(map(lambda i: self.num_res_blocks[i] >= num_attention_blocks[i], range(len(num_attention_blocks))))
#             print(f"Constructor of UNetModel received num_attention_blocks={num_attention_blocks}. "
#                   f"This option has LESS priority than attention_resolutions {attention_resolutions}, "
#                   f"i.e., in cases where num_attention_blocks[i] > 0 but 2**i not in attention_resolutions, "
#                   f"attention will still not be set.")

#         self.attention_resolutions = attention_resolutions
#         self.dropout = dropout
#         self.channel_mult = channel_mult
#         self.conv_resample = conv_resample
#         self.use_checkpoint = use_checkpoint
#         self.dtype = th.float16 if use_fp16 else th.float32
#         self.num_heads = num_heads
#         self.num_head_channels = num_head_channels
#         self.num_heads_upsample = num_heads_upsample
#         self.predict_codebook_ids = n_embed is not None
#         self.hyperconfig = hyperconfig
#         self.hypercolumn = instantiate_from_config(hyperconfig)

#         time_embed_dim = model_channels * 4
#         self.time_embed = nn.Sequential(
#             linear(model_channels, time_embed_dim),
#             nn.SiLU(),
#             linear(time_embed_dim, time_embed_dim),
#         )
#         self.group_embedding = ConditionEmbedding(num_conditions, embedding_dim)

#         self.input_blocks = nn.ModuleList(
#             [
#                 TimestepEmbedSequential(
#                     conv_nd(dims, in_channels, model_channels, 3, padding=1)
#                 )
#             ]
#         )
#         self.zero_convs = nn.ModuleList([self.make_zero_conv(model_channels)])

#         self.input_hint_block_new = TimestepEmbedSequential(
#             # HyperColumnLGN(hypercond=hypercond),
#             # Canny(),
#             self.hypercolumn,
#             conv_nd(dims, hint_channels, 16, 3, padding=1),
#             nn.SiLU(),
#             conv_nd(dims, 16, 16, 3, padding=1),
#             nn.SiLU(),
#             conv_nd(dims, 16, 32, 3, padding=1, stride=stride[0]),
#             nn.SiLU(),
#             conv_nd(dims, 32, 32, 3, padding=1),
#             nn.SiLU(),
#             conv_nd(dims, 32, 96, 3, padding=1, stride=stride[1]),
#             nn.SiLU(),
#             conv_nd(dims, 96, 96, 3, padding=1),
#             nn.SiLU(),
#             conv_nd(dims, 96, 256, 3, padding=1, stride=stride[2]),
#             nn.SiLU(),
#             zero_module(conv_nd(dims, 256, model_channels, 3, padding=1))
#         )

#         self._feature_size = model_channels
#         input_block_chans = [model_channels]
#         ch = model_channels
#         ds = 1
#         for level, mult in enumerate(channel_mult):
#             for nr in range(self.num_res_blocks[level]):
#                 layers = [
#                     ResBlock(
#                         ch,
#                         time_embed_dim,
#                         dropout,
#                         out_channels=mult * model_channels,
#                         dims=dims,
#                         use_checkpoint=use_checkpoint,
#                         use_scale_shift_norm=use_scale_shift_norm,
#                     )
#                 ]
#                 ch = mult * model_channels
#                 if ds in attention_resolutions:
#                     if num_head_channels == -1:
#                         dim_head = ch // num_heads
#                     else:
#                         num_heads = ch // num_head_channels
#                         dim_head = num_head_channels
#                     if legacy:
#                         # num_heads = 1
#                         dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
#                     if exists(disable_self_attentions):
#                         disabled_sa = disable_self_attentions[level]
#                     else:
#                         disabled_sa = False

#                     if not exists(num_attention_blocks) or nr < num_attention_blocks[level]:
#                         layers.append(
#                             AttentionBlock(
#                                 ch,
#                                 use_checkpoint=use_checkpoint,
#                                 num_heads=num_heads,
#                                 num_head_channels=dim_head,
#                                 use_new_attention_order=use_new_attention_order,
#                             ) if not use_spatial_transformer else SpatialTransformer(
#                                 ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
#                                 disable_self_attn=disabled_sa, use_linear=use_linear_in_transformer,
#                                 use_checkpoint=use_checkpoint
#                             )
#                         )
#                 self.input_blocks.append(TimestepEmbedSequential(*layers))
#                 self.zero_convs.append(self.make_zero_conv(ch))
#                 self._feature_size += ch
#                 input_block_chans.append(ch)
#             if level != len(channel_mult) - 1:
#                 out_ch = ch
#                 self.input_blocks.append(
#                     TimestepEmbedSequential(
#                         ResBlock(
#                             ch,
#                             time_embed_dim,
#                             dropout,
#                             out_channels=out_ch,
#                             dims=dims,
#                             use_checkpoint=use_checkpoint,
#                             use_scale_shift_norm=use_scale_shift_norm,
#                             down=True,
#                         )
#                         if resblock_updown
#                         else Downsample(
#                             ch, conv_resample, dims=dims, out_channels=out_ch
#                         )
#                     )
#                 )
#                 ch = out_ch
#                 input_block_chans.append(ch)
#                 self.zero_convs.append(self.make_zero_conv(ch))
#                 ds *= 2
#                 self._feature_size += ch

#         if num_head_channels == -1:
#             dim_head = ch // num_heads
#         else:
#             num_heads = ch // num_head_channels
#             dim_head = num_head_channels
#         if legacy:
#             # num_heads = 1
#             dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
#         self.middle_block = TimestepEmbedSequential(
#             ResBlock(
#                 ch,
#                 time_embed_dim,
#                 dropout,
#                 dims=dims,
#                 use_checkpoint=use_checkpoint,
#                 use_scale_shift_norm=use_scale_shift_norm,
#             ),
#             AttentionBlock(
#                 ch,
#                 use_checkpoint=use_checkpoint,
#                 num_heads=num_heads,
#                 num_head_channels=dim_head,
#                 use_new_attention_order=use_new_attention_order,
#             ) if not use_spatial_transformer else SpatialTransformer(  # always uses a self-attn
#                 ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
#                 disable_self_attn=disable_middle_self_attn, use_linear=use_linear_in_transformer,
#                 use_checkpoint=use_checkpoint
#             ),
#             ResBlock(
#                 ch,
#                 time_embed_dim,
#                 dropout,
#                 dims=dims,
#                 use_checkpoint=use_checkpoint,
#                 use_scale_shift_norm=use_scale_shift_norm,
#             ),
#         )
#         self.middle_block_out = self.make_zero_conv(ch)
#         self._feature_size += ch
#         # if is_train==True:
#         #     cur_hypercond=0
#         #     self.new_controlNet_1 = ControlSingle(image_size=image_size, in_channels=in_channels, model_channels=model_channels, hint_channels=hint_channels, num_res_blocks=num_res_blocks, attention_resolutions=attention_resolutions,
#         #                                         dropout=dropout, channel_mult=channel_mult, conv_resample=conv_resample, dims=dims, use_checkpoint=use_checkpoint, use_fp16=use_fp16, num_heads=num_heads, num_head_channels=num_head_channels,
#         #                                         num_heads_upsample=num_heads_upsample, use_scale_shift_norm=use_scale_shift_norm,resblock_updown=resblock_updown,use_new_attention_order=use_new_attention_order, use_spatial_transformer=use_spatial_transformer,
#         #                                         transformer_depth=transformer_depth,context_dim=context_dim,n_embed=n_embed,legacy=legacy,disable_self_attentions=None,num_attention_blocks=num_attention_blocks,disable_middle_self_attn=disable_middle_self_attn,
#         #                                         use_linear_in_transformer=use_linear_in_transformer,hyperconfig=hyperconfig,stride=stride,num_conditions=num_conditions,embedding_dim=embedding_dim,hypercond=cur_hypercond)        
#         #     self.load_model_parts(self.new_controlNet_1, f'./multi_ckpt/10epoch_0.pt') 
#         #     self.new_controlNets = nn.ModuleList([self.new_controlNet_1])

#     def make_zero_conv(self, channels):
#         return TimestepEmbedSequential(zero_module(conv_nd(self.dims, channels, channels, 1, padding=0)))

#     def save_model_parts(self, model, filepath):
#         state_dict = {
#             'input_hint_block_new': model.input_hint_block_new.state_dict(),
#             'input_blocks': model.input_blocks.state_dict(),
#             'zero_convs': model.zero_convs.state_dict(),
#             'middle_block': model.middle_block.state_dict(),
#             'middle_block_out': model.middle_block_out.state_dict()
#         }
#         torch.save(state_dict, filepath)

#     def load_model_parts(self, model, filepath):
#         state_dict = torch.load(filepath)
#         model.input_hint_block_new.load_state_dict(state_dict['input_hint_block_new'])
#         model.input_blocks.load_state_dict(state_dict['input_blocks'])
#         model.zero_convs.load_state_dict(state_dict['zero_convs'])
#         model.middle_block.load_state_dict(state_dict['middle_block'])
#         model.middle_block_out.load_state_dict(state_dict['middle_block_out'])

#     def forward(self, x, hint, timesteps, context ,hyper_scales=None,PCA=None, control_scale=None,training=False,multi=False,**kwargs):
#         t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
#         emb = self.time_embed(t_emb)
#         if training==False:
#             if multi==True:
#                 self.hypercolumn.cond = [0, 1]
#                 hs = [[] for _ in range(len(self.input_blocks) + 1)]
#                 outs = []
#                 for i in self.hypercolumn.cond:
#                     h = x.type(self.dtype)
#                     if i == 0:
#                         guided_hint = self.input_hint_block_new(hint, emb, context,hyper_scale=hyper_scales[i], cond=i)
#                         for j, (module, zero_conv) in enumerate(zip(self.input_blocks, self.zero_convs)):
#                             if guided_hint is not None:
#                                 h = module(h, emb, context)
#                                 h += guided_hint
#                                 guided_hint = None
#                             else:
#                                 h = module(h, emb, context)
#                             hs[j].append(zero_conv(h, emb, context))
#                         h = self.middle_block(h, emb, context)    
#                         hs[-1].append(self.middle_block_out(h, emb, context))
#                     else:
#                         model = self.new_controlNets[i-1]
#                         guided_hint = model.input_hint_block_new(hint, emb, context,hyper_scale=hyper_scales[i], cond=i)
#                         for j, (module, zero_conv) in enumerate(zip(model.input_blocks, model.zero_convs)):
#                             if guided_hint is not None:
#                                 h = module(h, emb, context)
#                                 h += guided_hint
#                                 guided_hint = None
#                             else:
#                                 h = module(h, emb, context)
#                             hs[j].append(zero_conv(h, emb, context))
#                         h = model.middle_block(h, emb, context)
#                         hs[-1].append(self.middle_block_out(h, emb, context))
#                 for i in range(len(hs)):
#                     outs.append(control_scale[0]*hs[i][0]+ control_scale[1]*hs[i][1])
#             else:
#                 guided_hint = self.input_hint_block_new(hint, emb, context, hyper_scale=hyper_scales, cond=self.hypercolumn.cond[0])
#                 outs = []
#                 h = x.type(self.dtype)
#                 for j, (module, zero_conv) in enumerate(zip(self.input_blocks, self.zero_convs)):
#                     if guided_hint is not None:
#                         h = module(h, emb, context)
#                         h += guided_hint
#                         guided_hint = None
#                     else:
#                         h = module(h, emb, context)
#                     outs.append(zero_conv(h, emb, context))
#                 h = self.middle_block(h, emb, context)
#                 outs.append(self.middle_block_out(h, emb, context))
#         else:
#             guided_hint = self.input_hint_block_new(hint, emb, context, hyper_scale=1, cond=self.hypercolumn.cond[0])
#             outs = []
#             h = x.type(self.dtype)
#             for j, (module, zero_conv) in enumerate(zip(self.input_blocks, self.zero_convs)):
#                 if guided_hint is not None:
#                     h = module(h, emb, context)
#                     h += guided_hint
#                     guided_hint = None
#                 else:
#                     h = module(h, emb, context)
#                 outs.append(zero_conv(h, emb, context))
#             h = self.middle_block(h, emb, context)
#             outs.append(self.middle_block_out(h, emb, context))
#         return outs   

# class ControlLDM(LatentDiffusion):

#     def __init__(self, control_stage_config, control_key, only_mid_control,control_step=0, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.control_model = instantiate_from_config(control_stage_config)
#         self.control_key = control_key
#         self.only_mid_control = only_mid_control
#         self.control_scales = [1.] * 13
#         self.control_step = control_step
#         if isinstance(self.control_model.hypercolumn,HyperColumnLGN):
#             self.hypercond = control_stage_config['params']['hyperconfig']['params']['hypercond']
#         else:
#             self.hypercond = [0]
#         self.select()
    
#     def training_step(self, batch, batch_idx):
#         return super().training_step(batch, batch_idx)
    
#     @torch.no_grad()
#     def get_input(self, batch, k, bs=None, *args, **kwargs):
#         x, c = super().get_input(batch, self.first_stage_key, *args, **kwargs)
#         control = batch[self.control_key]
#         # print('c:',c)
        
#         if bs is not None:
#             control = control[:bs]
#         control = control.to(self.device)
#         control = einops.rearrange(control, 'b h w c -> b c h w')
#         control = control.to(memory_format=torch.contiguous_format).float()
#         return x, dict(c_crossattn=[c], c_concat=[control])

#     def apply_model(self, x_noisy, t, cond, hyper_scales=None, PCA=None, control_scale=None,training=False,multi=False,*args, **kwargs):
#         assert isinstance(cond, dict)
#         diffusion_model = self.model.diffusion_model
#         # print('t:',t)

#         cond_txt = torch.cat(cond['c_crossattn'], 1)

#         if cond['c_concat'] is None:
#             eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=None, only_mid_control=self.only_mid_control)
#         else:
#             control = self.control_model(x=x_noisy, hint=torch.cat(cond['c_concat'], 1), timesteps=t, context=cond_txt, group_num=0, hyper_scales=hyper_scales,control_scale=control_scale,training=training,multi=multi)
#             if t[0]<self.control_step:
#                 control = [c * 0. for c, scale in zip(control, self.control_scales)]
#             else:
#                 control = [c * scale for c, scale in zip(control, self.control_scales)]
#             eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=control, only_mid_control=self.only_mid_control)
#             # controls = self.control_model(x=x_noisy, hint=torch.cat(cond['c_concat'], 1), timesteps=t, context=cond_txt, group_num=0)
#             # epss = []
#             # for i in range(len(controls[0])):
#             #     control = [item[i] for item in controls]
#             #     if t[0]<self.control_step:
#             #         control = [c * 0. for c, scale in zip(control, self.control_scales)]
#             #     else:
#             #         control = [c * scale for c, scale in zip(control, self.control_scales)]
#             #     eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=control, only_mid_control=self.only_mid_control)
#             #     epss.append(eps)
#             # eps = epss[0] #0.7*epss[0] + 0.1*epss[1] + 0.1*epss[2] + 0.1*epss[3]  #torch.mean(torch.stack(epss), dim=0)  #0.7*epss[0] + 0.1*epss[1] + 0.1*epss[2] + 0.1*epss[3]

#         return eps
    
#     def apply_model_train(self, x_noisy, t, cond, *args, **kwargs):
#         assert isinstance(cond, dict)
#         diffusion_model = self.model.diffusion_model
#         # print('t:',t)

#         cond_txt = torch.cat(cond['c_crossattn'], 1)

#         if cond['c_concat'] is None:
#             eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=None, only_mid_control=self.only_mid_control)
#         else:
#             control = self.control_model(x=x_noisy, hint=torch.cat(cond['c_concat'], 1), timesteps=t, context=cond_txt,training=True)
#             if t[0]<self.control_step:
#                 control = [c * 0. for c, scale in zip(control, self.control_scales)]
#             else:
#                 control = [c * scale for c, scale in zip(control, self.control_scales)]
#             eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=control, only_mid_control=self.only_mid_control)
#         return eps    

#     @torch.no_grad()
#     def get_unconditional_conditioning(self, N):
#         return self.get_learned_conditioning([""] * N)

#     @torch.no_grad()
#     def log_images(self, batch, N=4, n_row=2, sample=False, ddim_steps=50, ddim_eta=0.0, return_keys=None,
#                    quantize_denoised=True, inpaint=True, plot_denoise_rows=False, plot_progressive_rows=True,
#                    plot_diffusion_rows=False, unconditional_guidance_scale=7.5, unconditional_guidance_label=None,
#                    use_ema_scope=True,
#                    **kwargs):
#         use_ddim = ddim_steps is not None

#         log = dict()
#         z, c = self.get_input(batch, self.first_stage_key, bs=N)
#         c_cat, c = c["c_concat"][0][:N], c["c_crossattn"][0][:N]
#         N = min(z.shape[0], N)
#         n_row = min(z.shape[0], n_row)
#         log["reconstruction"] = self.decode_first_stage(z)
#         log["control"] = c_cat * 2.0 - 1.0
#         log["conditioning"] = log_txt_as_img((512, 512), batch[self.cond_stage_key], size=16)

#         if plot_diffusion_rows:
#             # get diffusion row
#             diffusion_row = list()
#             z_start = z[:n_row]
#             for t in range(self.num_timesteps):
#                 if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
#                     t = repeat(torch.tensor([t]), '1 -> b', b=n_row)
#                     t = t.to(self.device).long()
#                     noise = torch.randn_like(z_start)
#                     z_noisy = self.q_sample(x_start=z_start, t=t, noise=noise)
#                     diffusion_row.append(self.decode_first_stage(z_noisy))

#             diffusion_row = torch.stack(diffusion_row)  # n_log_step, n_row, C, H, W
#             diffusion_grid = rearrange(diffusion_row, 'n b c h w -> b n c h w')
#             diffusion_grid = rearrange(diffusion_grid, 'b n c h w -> (b n) c h w')
#             diffusion_grid = make_grid(diffusion_grid, nrow=diffusion_row.shape[0])
#             log["diffusion_row"] = diffusion_grid

#         if sample:
#             # get denoise row
#             samples, z_denoise_row = self.sample_log(cond={"c_concat": [c_cat], "c_crossattn": [c]},
#                                                      batch_size=N, ddim=use_ddim,
#                                                      ddim_steps=ddim_steps, eta=ddim_eta)
#             x_samples = self.decode_first_stage(samples)
#             log["samples"] = x_samples
#             if plot_denoise_rows:
#                 denoise_grid = self._get_denoise_row_from_list(z_denoise_row)
#                 log["denoise_row"] = denoise_grid

#         if unconditional_guidance_scale >= -1.0:
#             uc_cross = self.get_unconditional_conditioning(N)
#             uc_cat = c_cat  # torch.zeros_like(c_cat)
#             uc_full = {"c_concat": [uc_cat], "c_crossattn": [uc_cross]}
#             for slct in self.slct:
#                 self.control_model.input_hint_block_new[0].cond = slct['cond']
#                 suffix = slct['suffix']
#                 print(suffix)
#                 samples_cfg, _ = self.sample_log(cond={"c_concat": [c_cat], "c_crossattn": [c]},
#                                                 batch_size=N, ddim=use_ddim,
#                                                 ddim_steps=ddim_steps, eta=ddim_eta,
#                                                 unconditional_guidance_scale=unconditional_guidance_scale,
#                                                 unconditional_conditioning=uc_full,training=True
#                                                 )
#                 x_samples_cfg = self.decode_first_stage(samples_cfg)
#                 deconv = self.control_model.input_hint_block_new[0].deconv(c_cat)
#                 log[f"{suffix}_samples_cfg_scale_{unconditional_guidance_scale:.2f}"] = x_samples_cfg
#                 log[f"{suffix}_deconv_samples_cfg_scale_{unconditional_guidance_scale:.2f}"] = deconv
#                 # self.control_model.input_hint_block_new[0].cond = None

#         return log

#     @torch.no_grad()
#     def sample_log(self, cond, batch_size, ddim, ddim_steps, training=False,**kwargs):
#         ddim_sampler = DDIMSampler(self)
#         b, c, h, w = cond["c_concat"][0].shape
#         shape = (self.channels, h // 8, w // 8)
#         samples, intermediates = ddim_sampler.sample(ddim_steps, batch_size, shape, cond, verbose=False,training=training, **kwargs)
#         return samples, intermediates

#     def configure_optimizers(self):
#         lr = self.learning_rate
#         params = list(self.control_model.parameters())
#         if not self.sd_locked:
#             params += list(self.model.diffusion_model.output_blocks.parameters())
#             params += list(self.model.diffusion_model.out.parameters())
#         opt = torch.optim.AdamW(params, lr=lr)
#         return opt

#     def low_vram_shift(self, is_diffusing):
#         if is_diffusing:
#             self.model = self.model.cuda()
#             self.control_model = self.control_model.cuda()
#             self.first_stage_model = self.first_stage_model.cpu()
#             self.cond_stage_model = self.cond_stage_model.cpu()
#         else:
#             self.model = self.model.cpu()
#             self.control_model = self.control_model.cpu()
#             self.first_stage_model = self.first_stage_model.cuda()
#             self.cond_stage_model = self.cond_stage_model.cuda()

#     @torch.no_grad()
#     def select(self):
#         # self.slct = [{'cond':[0],'suffix':'0'},{'cond':[1],'suffix':'1'},{'cond':[2],'suffix':'2'},{'cond':[3],'suffix':'3'},{'cond':[0,1],'suffix':'01'},
#         #              {'cond':[0,2],'suffix':'02'},{'cond':[0,3],'suffix':'03'},{'cond':[1,2],'suffix':'12'},{'cond':[1,3],'suffix':'13'},{'cond':[2,3],'suffix':'23'},
#         #              {'cond':[0,1,2],'suffix':'012'},{'cond':[0,1,3],'suffix':'013'},{'cond':[1,2,3],'suffix':'123'},
#         #              {'cond':[0,1,2,3],'suffix':'0123'}]
#         self.slct = [{'cond':[i],'suffix':f'{i}'} for i in self.hypercond]

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

# sys.path.append('/home/chenzhiqiang/code/')
sys.path.append('/mnt/dantongwu/control-net-main-v0.5')
from hypercolumn.vit_pytorch.train_V1_sep_new import Column_trans_rot_lgn

from ldm.modules.diffusionmodules.util import (
    conv_nd,
    linear,
    zero_module,
    timestep_embedding,
)

from einops import rearrange, repeat
from torchvision.utils import make_grid
from ldm.modules.attention import SpatialTransformer
from ldm.modules.attention import FeedForward
from ldm.modules.diffusionmodules.openaimodel import UNetModel, TimestepEmbedSequential, ResBlock, Downsample, AttentionBlock
from ldm.models.diffusion.ddpm import LatentDiffusion
from ldm.util import log_txt_as_img, exists, instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler


class ControlledUnetModel(UNetModel):
    def forward(self, x, timesteps=None, context=None, control=None, only_mid_control=False, **kwargs):
        hs = []
        with torch.no_grad():
            t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
            emb = self.time_embed(t_emb)
            h = x.type(self.dtype)
            for module in self.input_blocks:
                h = module(h, emb, context)
                hs.append(h)
            h = self.middle_block(h, emb, context)

        if control is not None:
            h += control.pop()

        for i, module in enumerate(self.output_blocks):
            if only_mid_control or control is None:
                h = torch.cat([h, hs.pop()], dim=1)
            else:
                h = torch.cat([h, hs.pop() + control.pop()], dim=1)
            h = module(h, emb, context)

        h = h.type(x.dtype)
        return self.out(h)

def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


class HyperColumnLGN(nn.Module):
    def __init__(self,key=0,hypercond=[0],size=512,restore_ckpt = '/home/uchihawdt/control-net-main-v0.5/hypercolumn/checkpoint/imagenet/equ_nv16_vl4_rn1_Bipolar_norm.pth',groups = [[0,1,4,8,9,15],[2,3],[5,12],[10],[6,7],[11,13,14],[5],[12],[2],[3]]):
        super().__init__()
        ckpt = torch.load(restore_ckpt,weights_only=False)
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
        
        self.groups = groups
        self.p = [0. for i in range(len(self.groups))]
        self.p[0] = 1.

        norm_mean = np.array([0.50705882, 0.48666667, 0.44078431])
        norm_std = np.array([0.26745098, 0.25568627, 0.27607843])
        self.norm = transforms.Normalize(norm_mean, norm_std)
        self.cond = hypercond
        self.slct = None
    
    def forward(self,x, hyper_scale=1, cond=None):
        x = self.resize(x)
        s = x.size()
        r = torch.zeros(1,self.lgn_ende.n_vector,1,1).to(x.device)
        cond = cond if cond!=None else self.cond

        if cond is None:
            c = [i for i in range(len(self.groups))]
            random.shuffle(c)
            # print(self.groups[c[0]])
            pa = random.random()
            for i in range(len(self.groups)):
                if pa < self.p[i]:
                    for j in self.groups[c[i]]:
                        r[:,j,:,:] = 1.
        else:
            # for i in cond:
            #     for j in self.groups[i]:
            #         r[:,j,:,:] = 1
            for j in self.groups[cond]:
                r[:,j,:,:] = 1

        r = repeat(r,'n c h w -> n (c repeat) h w',repeat=self.lgn_ende.vector_length)
        out = self.lgn_ende(self.norm(x))*r
        out = self.pad(self.lgn_ende.deconv(out))
        # out *= hyper_scale
        return out
    
    def deconv(self,x):
        x = self.resize(x)
        s = x.size()
        r = torch.zeros(1,self.lgn_ende.n_vector,1,1).to(x.device)
        if self.cond is not None:
            for i in self.cond:
                for j in self.groups[i]:
                    r[:,j,:,:] = 1

        r = repeat(r,'n c h w -> n (c repeat) h w',repeat=self.lgn_ende.vector_length)
        conv = self.lgn_ende(self.norm(x))*r
        deconv = self.lgn_ende.deconv(conv)
        deconv = (deconv - reduce(deconv,'b c h w -> b c () ()', 'min'))/(reduce(deconv,'b c h w -> b c () ()', 'max')-reduce(deconv,'b c h w -> b c () ()', 'min'))*2.-1.

        return deconv

class HyperColumnLGNVisual(nn.Module):
    def __init__(self,key=0,hypercond=[0],size=512,restore_ckpt = '/home/uchihawdt/control-net-main-v0.5/hypercolumn/checkpoint/imagenet/equ_nv16_vl4_rn1_Bipolar_norm.pth'):
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
        

        # self.groups = [[2,3],[0,1,4,8,9,15],[5,6,7,10,12],[11,13,14]]
        self.groups = [[0,1,4,8,9,15],[2,3],[5,12],[10],[6,7],[11,13,14],[5],[12],[2],[3]]
        # self.groups = [[2],[3],[5],[12]]
            
        # self.p = [1.,0.5,0.25,0.125]
        self.p = [0. for i in range(len(self.groups))]
        self.p[0] = 1.

        norm_mean = np.array([0.50705882, 0.48666667, 0.44078431])
        norm_std = np.array([0.26745098, 0.25568627, 0.27607843])
        self.norm = transforms.Normalize(norm_mean, norm_std)
        self.cond = hypercond
        self.slct = None

    # def forward(self,x):
    #     s = x.size()
    #     r = torch.zeros(1,16,1,1).to(x.device)

    #     if self.cond is None:
    #         c = [i for i in range(len(self.groups))]
    #         random.shuffle(c)
    #         # print(self.groups[c[0]])
    #         pa = random.random()
    #         for i in range(4):
    #             if pa < self.p[i]:
    #                 for j in self.groups[c[i]]:
    #                     r[:,j,:,:] = 1.
    #     else:
    #         for i in self.cond:
    #             for j in self.groups[i]:
    #                 r[:,j,:,:] = 1

    #     r = repeat(r,'n c h w -> n (c repeat) h w',repeat=4)
    #     return self.lgn_ende(self.norm(x))*r
    
    def forward(self,x):
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

        # r = repeat(r,'n c h w -> n (c repeat) h w',repeat=4)
        # out = self.lgn_ende(self.norm(x))*r
        # # print('out_conv:',out.size())
        # out = self.pad(self.lgn_ende.deconv(out))
        # # out = self.lgn_ende.deconv(self.lgn_ende(self.norm(x))*r)
        # # print('out:',out.size())
        # return out

        out = self.lgn_ende(self.norm(x))
        # out = rearrange(out, 'b (c n) h w -> b c (n h) w',n=16)
        out = rearrange(out, 'b (n c) h w -> b c (n h) w', n=16)
        # out = self.pad(self.lgn_ende.deconv(out))
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
    
class HyperColumnLGNFeature(nn.Module):
    def __init__(self,key=0,hypercond=[0],size=512,restore_ckpt = '/home/uchihawdt/control-net-main-v0.5/hypercolumn/checkpoint/imagenet/equ_nv16_vl4_rn1_Bipolar_norm.pth'):
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
        

        # self.groups = [[2,3],[0,1,4,8,9,15],[5,6,7,10,12],[11,13,14]]
        self.groups = [[0,1,4,8,9,15],[2,3],[5,12],[10],[6,7],[11,13,14],[5],[12],[2],[3]]
        # self.groups = [[2],[3],[5],[12]]
            
        # self.p = [1.,0.5,0.25,0.125]
        self.p = [0. for i in range(len(self.groups))]
        self.p[0] = 1.

        norm_mean = np.array([0.50705882, 0.48666667, 0.44078431])
        norm_std = np.array([0.26745098, 0.25568627, 0.27607843])
        self.norm = transforms.Normalize(norm_mean, norm_std)
        self.cond = hypercond
        self.slct = None
    
    def forward(self,x):
        x = self.resize(x)
        s = x.size()
        # r = torch.zeros(1,16,1,1).to(x.device)

        # if self.cond is None:
        #     c = [i for i in range(len(self.groups))]
        #     random.shuffle(c)
        #     # print(self.groups[c[0]])
        #     pa = random.random()
        #     for i in range(len(self.groups)):
        #         if pa < self.p[i]:
        #             for j in self.groups[c[i]]:
        #                 r[:,j,:,:] = 1.
        # else:
        #     for i in self.cond:
        #         for j in self.groups[i]:
        #             r[:,j,:,:] = 1

        # r = repeat(r,'n c h w -> n (c repeat) h w',repeat=4)
        out = self.lgn_ende(self.norm(x))
        out = out[:,40:44,:,:]
        # out = self.pad(self.lgn_ende.deconv(out))
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


class SemanticAdapter(nn.Module):
    def __init__(self, in_dim, channel_mult=[2, 4]):
        super().__init__()
        dim_out1, mult1 = in_dim*channel_mult[0], channel_mult[0]*2
        dim_out2, mult2 = in_dim*channel_mult[1], channel_mult[1]*2//channel_mult[0]
        self.in_dim = in_dim
        self.channel_mult = channel_mult
        
        self.ff1 = FeedForward(in_dim, dim_out=dim_out1, mult=mult1, glu=True, dropout=0.1)
        self.ff2 = FeedForward(dim_out1, dim_out=dim_out2, mult=mult2, glu=True, dropout=0.3)
        self.norm1 = nn.LayerNorm(in_dim)
        self.norm2 = nn.LayerNorm(dim_out1)

    def forward(self, x):
        x = self.ff1(self.norm1(x))
        x = self.ff2(self.norm2(x))
        x = rearrange(x, 'b (n d) -> b n d', n=self.channel_mult[-1], d=self.in_dim).contiguous()
        return x


class Canny(nn.Module):
    def __init__(self,para=None):
        super().__init__()
    def forward(self,x):
        return x
    def deconv(self,x):
        return x


class ConditionEmbedding(nn.Module):
    def __init__(self, num_conditions, embedding_dim):
        super(ConditionEmbedding, self).__init__()
        self.embedding = nn.Embedding(num_conditions, embedding_dim)

    def forward(self, condition_type):
        return self.embedding(condition_type)

class ControlSingle(nn.Module):
    def __init__(self,image_size, in_channels, model_channels, hint_channels, num_res_blocks, attention_resolutions, dropout=0, channel_mult=(1, 2, 4, 8), conv_resample=True, dims=2, use_checkpoint=False, use_fp16=False, num_heads=-1, num_head_channels=-1, num_heads_upsample=-1, use_scale_shift_norm=False, resblock_updown=False, use_new_attention_order=False, use_spatial_transformer=False, transformer_depth=1, context_dim=None, n_embed=None, legacy=True, disable_self_attentions=None, num_attention_blocks=None, disable_middle_self_attn=False, use_linear_in_transformer=False, hyperconfig=None, stride=[2,2,2], num_conditions=6, embedding_dim=128,hypercond=0,size=None,restore_ckpt=None,groups=None):
        super().__init__()
        self.dims = dims
        hyperconfig['params']['hypercond'] = [hypercond]
        if size and restore_ckpt and groups:
            hyperconfig['params']['size']=size #256
            hyperconfig['params']['restore_ckpt']=restore_ckpt  #'/home/chenzhiqiang/code/ControlNet-main/hypercolumn/checkpoint/imagenet/equ_nv128_vl1_rn8_Vanilla_ks17_norm.pth'
            hyperconfig['params']['groups']=groups  #: [[0],[1],[2],[3],[4],[5],[6],[7],[8],[9],[10],[11],[12],[13],[14],[15]]
        self.hypercolumn = instantiate_from_config(hyperconfig)
        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(dims, in_channels, model_channels, 3, padding=1)
                )
            ]
        )
        self.zero_convs = nn.ModuleList([self.make_zero_conv(model_channels)])

        if isinstance(num_res_blocks, int):
            self.num_res_blocks = len(channel_mult) * [num_res_blocks]
        else:
            if len(num_res_blocks) != len(channel_mult):
                raise ValueError("provide num_res_blocks either as an int (globally constant) or "
                                 "as a list/tuple (per-level) with the same length as channel_mult")
            self.num_res_blocks = num_res_blocks
        
        time_embed_dim = model_channels * 4
        self.input_hint_block_new = TimestepEmbedSequential(self.hypercolumn,
                                                            conv_nd(dims, hint_channels, 16, 3, padding=1),nn.SiLU(),
                                                            conv_nd(dims, 16, 16, 3, padding=1),nn.SiLU(),
                                                            conv_nd(dims, 16, 32, 3, padding=1, stride=stride[0]),nn.SiLU(),
                                                            conv_nd(dims, 32, 32, 3, padding=1),nn.SiLU(),
                                                            conv_nd(dims, 32, 96, 3, padding=1, stride=stride[1]),nn.SiLU(),
                                                            conv_nd(dims, 96, 96, 3, padding=1),nn.SiLU(),
                                                            conv_nd(dims, 96, 256, 3, padding=1, stride=stride[2]),nn.SiLU(),
                                                            zero_module(conv_nd(dims, 256, model_channels, 3, padding=1)))
        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for nr in range(self.num_res_blocks[level]):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        # num_heads = 1
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                    if exists(disable_self_attentions):
                        disabled_sa = disable_self_attentions[level]
                    else:
                        disabled_sa = False

                    if not exists(num_attention_blocks) or nr < num_attention_blocks[level]:
                        layers.append(
                            AttentionBlock(
                                ch,
                                use_checkpoint=use_checkpoint,
                                num_heads=num_heads,
                                num_head_channels=dim_head,
                                use_new_attention_order=use_new_attention_order,
                            ) if not use_spatial_transformer else SpatialTransformer(
                                ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                                disable_self_attn=disabled_sa, use_linear=use_linear_in_transformer,
                                use_checkpoint=use_checkpoint
                            )
                        )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self.zero_convs.append(self.make_zero_conv(ch))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                self.zero_convs.append(self.make_zero_conv(ch))
                ds *= 2
                self._feature_size += ch

        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels
        if legacy:
            # num_heads = 1
            dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=dim_head,
                use_new_attention_order=use_new_attention_order,
            ) if not use_spatial_transformer else SpatialTransformer(  # always uses a self-attn
                ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                disable_self_attn=disable_middle_self_attn, use_linear=use_linear_in_transformer,
                use_checkpoint=use_checkpoint
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self.middle_block_out = self.make_zero_conv(ch)
    def make_zero_conv(self, channels):
        return TimestepEmbedSequential(zero_module(conv_nd(self.dims, channels, channels, 1, padding=0)))

class ControlNet(nn.Module):
    def __init__(
            self,
            image_size,
            in_channels,
            model_channels,
            hint_channels,
            num_res_blocks,
            attention_resolutions,
            dropout=0,
            channel_mult=(1, 2, 4, 8),
            conv_resample=True,
            dims=2,
            use_checkpoint=False,
            use_fp16=False,
            num_heads=-1,
            num_head_channels=-1,
            num_heads_upsample=-1,
            use_scale_shift_norm=False,
            resblock_updown=False,
            use_new_attention_order=False,
            use_spatial_transformer=False,  # custom transformer support
            transformer_depth=1,  # custom transformer support
            context_dim=None,  # custom transformer support
            n_embed=None,  # custom support for prediction of discrete ids into codebook of first stage vq model
            legacy=True,
            disable_self_attentions=None,
            num_attention_blocks=None,
            disable_middle_self_attn=False,
            use_linear_in_transformer=False,
            hyperconfig=None,
            stride=[2,2,2],
            num_conditions=6, 
            embedding_dim=128,
            is_train=False,
            epoch_num=None,
    ):
        super().__init__()
        if use_spatial_transformer:
            assert context_dim is not None, 'Fool!! You forgot to include the dimension of your cross-attention conditioning...'

        if context_dim is not None:
            assert use_spatial_transformer, 'Fool!! You forgot to use the spatial transformer for your cross-attention conditioning...'
            from omegaconf.listconfig import ListConfig
            if type(context_dim) == ListConfig:
                context_dim = list(context_dim)

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        if num_heads == -1:
            assert num_head_channels != -1, 'Either num_heads or num_head_channels has to be set'

        if num_head_channels == -1:
            assert num_heads != -1, 'Either num_heads or num_head_channels has to be set'

        self.dims = dims
        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        if isinstance(num_res_blocks, int):
            self.num_res_blocks = len(channel_mult) * [num_res_blocks]
        else:
            if len(num_res_blocks) != len(channel_mult):
                raise ValueError("provide num_res_blocks either as an int (globally constant) or "
                                 "as a list/tuple (per-level) with the same length as channel_mult")
            self.num_res_blocks = num_res_blocks
        if disable_self_attentions is not None:
            # should be a list of booleans, indicating whether to disable self-attention in TransformerBlocks or not
            assert len(disable_self_attentions) == len(channel_mult)
        if num_attention_blocks is not None:
            assert len(num_attention_blocks) == len(self.num_res_blocks)
            assert all(map(lambda i: self.num_res_blocks[i] >= num_attention_blocks[i], range(len(num_attention_blocks))))
            print(f"Constructor of UNetModel received num_attention_blocks={num_attention_blocks}. "
                  f"This option has LESS priority than attention_resolutions {attention_resolutions}, "
                  f"i.e., in cases where num_attention_blocks[i] > 0 but 2**i not in attention_resolutions, "
                  f"attention will still not be set.")

        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.predict_codebook_ids = n_embed is not None
        self.hyperconfig = hyperconfig
        self.hypercolumn = instantiate_from_config(hyperconfig)

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )
        self.group_embedding = ConditionEmbedding(num_conditions, embedding_dim)

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(dims, in_channels, model_channels, 3, padding=1)
                )
            ]
        )
        self.zero_convs = nn.ModuleList([self.make_zero_conv(model_channels)])

        self.input_hint_block_new = TimestepEmbedSequential(
            # HyperColumnLGN(hypercond=hypercond),
            # Canny(),
            self.hypercolumn,
            conv_nd(dims, hint_channels, 16, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 16, 16, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 16, 32, 3, padding=1, stride=stride[0]),
            nn.SiLU(),
            conv_nd(dims, 32, 32, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 32, 96, 3, padding=1, stride=stride[1]),
            nn.SiLU(),
            conv_nd(dims, 96, 96, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 96, 256, 3, padding=1, stride=stride[2]),
            nn.SiLU(),
            zero_module(conv_nd(dims, 256, model_channels, 3, padding=1))
        )

        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for nr in range(self.num_res_blocks[level]):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        # num_heads = 1
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                    if exists(disable_self_attentions):
                        disabled_sa = disable_self_attentions[level]
                    else:
                        disabled_sa = False

                    if not exists(num_attention_blocks) or nr < num_attention_blocks[level]:
                        layers.append(
                            AttentionBlock(
                                ch,
                                use_checkpoint=use_checkpoint,
                                num_heads=num_heads,
                                num_head_channels=dim_head,
                                use_new_attention_order=use_new_attention_order,
                            ) if not use_spatial_transformer else SpatialTransformer(
                                ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                                disable_self_attn=disabled_sa, use_linear=use_linear_in_transformer,
                                use_checkpoint=use_checkpoint
                            )
                        )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self.zero_convs.append(self.make_zero_conv(ch))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                self.zero_convs.append(self.make_zero_conv(ch))
                ds *= 2
                self._feature_size += ch

        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels
        if legacy:
            # num_heads = 1
            dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=dim_head,
                use_new_attention_order=use_new_attention_order,
            ) if not use_spatial_transformer else SpatialTransformer(  # always uses a self-attn
                ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                disable_self_attn=disable_middle_self_attn, use_linear=use_linear_in_transformer,
                use_checkpoint=use_checkpoint
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self.middle_block_out = self.make_zero_conv(ch)
        self._feature_size += ch
        # control_net_args = {'image_size': image_size, 'in_channels': in_channels, 'model_channels': model_channels, 'hint_channels': hint_channels, 'num_res_blocks': num_res_blocks, 'attention_resolutions': attention_resolutions,
        #                     'dropout': dropout, 'channel_mult': channel_mult, 'conv_resample': conv_resample, 'dims': dims, 'use_checkpoint': use_checkpoint, 'use_fp16': use_fp16, 'num_heads': num_heads, 'num_head_channels': num_head_channels,
        #                     'num_heads_upsample': num_heads_upsample, 'use_scale_shift_norm': use_scale_shift_norm,'resblock_updown': resblock_updown,'use_new_attention_order': use_new_attention_order, 'use_spatial_transformer': use_spatial_transformer,
        #                     'transformer_depth': transformer_depth,'context_dim': context_dim,'n_embed': n_embed,'legacy': legacy,'disable_self_attentions': None,'num_attention_blocks': num_attention_blocks,'disable_middle_self_attn': disable_middle_self_attn,
        #                     'use_linear_in_transformer': use_linear_in_transformer,'hyperconfig': hyperconfig,'stride': stride,'num_conditions': num_conditions,'embedding_dim': embedding_dim,}
        # self.new_controlNet = ControlSingle(self.image_size, self.in_channels, self.model_channels, hint_channels, self.num_res_blocks, self.attention_resolutions, hyperconfig=hyperconfig, dims = self.dims, channel_mult=self.channel_mult,
        #                                     num_heads = self.num_heads, num_heads_upsample=self.num_heads_upsample, stride=stride, legacy=legacy)
        if is_train==False:  #and epoch_num:
            cur_hypercond=1
            self.new_controlNet_1 = ControlSingle(image_size=image_size, in_channels=in_channels, model_channels=model_channels, hint_channels=hint_channels, num_res_blocks=num_res_blocks, attention_resolutions=attention_resolutions,
                                                dropout=dropout, channel_mult=channel_mult, conv_resample=conv_resample, dims=dims, use_checkpoint=use_checkpoint, use_fp16=use_fp16, num_heads=num_heads, num_head_channels=num_head_channels,
                                                num_heads_upsample=num_heads_upsample, use_scale_shift_norm=use_scale_shift_norm,resblock_updown=resblock_updown,use_new_attention_order=use_new_attention_order, use_spatial_transformer=use_spatial_transformer,
                                                transformer_depth=transformer_depth,context_dim=context_dim,n_embed=n_embed,legacy=legacy,disable_self_attentions=None,num_attention_blocks=num_attention_blocks,disable_middle_self_attn=disable_middle_self_attn,
                                                use_linear_in_transformer=use_linear_in_transformer,hyperconfig=hyperconfig,stride=stride,num_conditions=num_conditions,embedding_dim=embedding_dim,hypercond=cur_hypercond)        
            # self.load_model_parts(self.new_controlNet_1, f'./multi_ckpt/1_10epoch_512_tuned.pt') #f'./multi_ckpt/1_10epoch_512_tuned.pt')      #10epoch_{cur_hypercond}_mini.pt   10epoch_{cur_hypercond}_512_10.pt')  #{epoch_num}   10epoch_{cur_hypercond}.pt')
            self.load_model_parts(self.new_controlNet_1, f'./multi_ckpt/1_10epoch_512.pt')  
            self.new_controlNets = nn.ModuleList([self.new_controlNet_1])

    def make_zero_conv(self, channels):
        return TimestepEmbedSequential(zero_module(conv_nd(self.dims, channels, channels, 1, padding=0)))

    def save_model_parts(self, model, filepath):
        state_dict = {
            'input_hint_block_new': model.input_hint_block_new.state_dict(),
            'input_blocks': model.input_blocks.state_dict(),
            'zero_convs': model.zero_convs.state_dict(),
            'middle_block': model.middle_block.state_dict(),
            'middle_block_out': model.middle_block_out.state_dict()
        }
        torch.save(state_dict, filepath)

    def load_model_parts(self, model, filepath):
        state_dict = torch.load(filepath)
        model.input_hint_block_new.load_state_dict(state_dict['input_hint_block_new'])
        model.input_blocks.load_state_dict(state_dict['input_blocks'])
        model.zero_convs.load_state_dict(state_dict['zero_convs'])
        model.middle_block.load_state_dict(state_dict['middle_block'])
        model.middle_block_out.load_state_dict(state_dict['middle_block_out'])

    def forward(self, x, hint, timesteps, context ,hyper_scales=None,PCA=None, control_scale=None,training=False,multi=False,**kwargs):
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)
        if training==False:
            if multi==True:
                self.hypercolumn.cond = [0, 1]
                hs = [[] for _ in range(len(self.input_blocks) + 1)]
                outs = []
                # self.save_model_parts(self, 'multi_ckpt/5epoch_4_256.pt')
                # for i in self.hypercolumn.cond:
                for i in range(2):
                    h = x.type(self.dtype)
                    if i==0: #self.hypercolumn.cond.index(i)==0:
                        guided_hint = self.input_hint_block_new(hint, emb, context,hyper_scale=1, cond=self.hypercolumn.cond[i]) #self.hypercolumn.cond.index(i)
                        for j, (module, zero_conv) in enumerate(zip(self.input_blocks, self.zero_convs)):
                            if guided_hint is not None:
                                h = module(h, emb, context)
                                h += guided_hint
                                guided_hint = None
                            else:
                                h = module(h, emb, context)
                            hs[j].append(zero_conv(h, emb, context))
                        h = self.middle_block(h, emb, context)    
                        hs[-1].append(self.middle_block_out(h, emb, context))
                    else:
                        model = self.new_controlNets[i-1] #self.hypercolumn.cond.index(i)-1] #[i-1]
                        guided_hint = model.input_hint_block_new(hint, emb, context,hyper_scale=1, cond=self.hypercolumn.cond[i]) #hyper_scales[self.hypercolumn.cond.index(i)], cond=i)
                        for j, (module, zero_conv) in enumerate(zip(model.input_blocks, model.zero_convs)):
                            if guided_hint is not None:
                                h = module(h, emb, context)
                                h += guided_hint
                                guided_hint = None
                            else:
                                h = module(h, emb, context)
                            hs[j].append(zero_conv(h, emb, context))
                        h = model.middle_block(h, emb, context)
                        hs[-1].append(self.middle_block_out(h, emb, context))
                for i in range(len(hs)):
                    outs.append(control_scale[0]*hs[i][0]+ control_scale[1]*hs[i][1]) #+ control_scale[2]*hs[i][2])
            else:
                
                guided_hint = self.input_hint_block_new(hint, emb, context, hyper_scale=hyper_scales, cond=self.hypercolumn.cond[0])
                outs = []
                h = x.type(self.dtype)
                for j, (module, zero_conv) in enumerate(zip(self.input_blocks, self.zero_convs)):
                    if guided_hint is not None:
                        h = module(h, emb, context)
                        h += guided_hint
                        guided_hint = None
                    else:
                        h = module(h, emb, context)
                    outs.append(zero_conv(h, emb, context))
                h = self.middle_block(h, emb, context)
                outs.append(self.middle_block_out(h, emb, context))
        else:
            guided_hint = self.input_hint_block_new(hint, emb, context, hyper_scale=1, cond=self.hypercolumn.cond[0])
            outs = []
            h = x.type(self.dtype)
            for j, (module, zero_conv) in enumerate(zip(self.input_blocks, self.zero_convs)):
                if guided_hint is not None:
                    h = module(h, emb, context)
                    h += guided_hint
                    guided_hint = None
                else:
                    h = module(h, emb, context)
                outs.append(zero_conv(h, emb, context))
            h = self.middle_block(h, emb, context)
            outs.append(self.middle_block_out(h, emb, context))
        return outs   

class ControlLDM(LatentDiffusion):

    def __init__(self, control_stage_config, control_key, only_mid_control,control_step=0, use_semantic=False,semantic_strength=1,semantic_control_config=None, model_path=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.control_model = instantiate_from_config(control_stage_config)
        self.control_key = control_key
        self.only_mid_control = only_mid_control
        self.control_scales = [1.] * 13      #scale0
        # self.control_scales = [1] * 3 + [2.]*3 + [4.]*3 + [8.]*4      #scale1
        # self.control_scales = [0.5] * 3 + [1.]*3 + [2.]*3 + [4.]*4      #scale2
        # self.control_scales = [1. * (0.825 ** float(12 - i)) for i in range(13)]      #scale4
        self.control_step = control_step
        self.use_semantic = use_semantic
        self.semantic_strength = semantic_strength
        self.semantic_control_config = semantic_control_config
        if semantic_control_config:
            self.model_path = model_path
            self.semantic_adapter = instantiate_from_config(self.semantic_control_config)
            semantic_adapter_state_dict = torch.load(self.model_path)
            self.semantic_adapter.load_state_dict(semantic_adapter_state_dict)
        
        # self.control_scales = [1.5] * 4 + [1.]*3 + [0.75]*3 + [0.5]*3
        # self.control_scales = [4.,4.,4.,4.,2.,2.,2.,1.,1.,1.,0.5,0.5,0.5]      #scale3
        if isinstance(self.control_model.hypercolumn,HyperColumnLGN):
            self.hypercond = control_stage_config['params']['hyperconfig']['params']['hypercond']
        else:
            self.hypercond = [0]
        self.select()
    
    def training_step(self, batch, batch_idx):
        return super().training_step(batch, batch_idx)
    
    
    # def validation_step(self, batch, batch_idx):
    #     return super().training_step(batch, batch_idx)

    @torch.no_grad()
    def get_input(self, batch, k, bs=None, *args, **kwargs):
        x, c = super().get_input(batch, self.first_stage_key, *args, **kwargs)
        control = batch[self.control_key]
        # print('c:',c)
        
        if bs is not None:
            control = control[:bs]
        control = control.to(self.device)
        control = einops.rearrange(control, 'b h w c -> b c h w')
        control = control.to(memory_format=torch.contiguous_format).float()
        return x, dict(c_crossattn=[c], c_concat=[control])

    def apply_model(self, x_noisy, t, cond, hyper_scales=None, PCA=None, control_scale=None,training=False,multi=False,*args, **kwargs):
        assert isinstance(cond, dict)
        diffusion_model = self.model.diffusion_model
        # print('t:',t)

        cond_txt = torch.cat(cond['c_crossattn'], 1)

        if cond['c_concat'] is None:
            eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=None, only_mid_control=self.only_mid_control)
        else:
            if self.use_semantic:
                semantic_control = self.semantic_adapter(cond['semantic_control'][0])
                cond_txt = torch.cat([cond_txt, self.semantic_strength*semantic_control], dim=1)
            control = self.control_model(x=x_noisy, hint=torch.cat(cond['c_concat'], 1), timesteps=t, context=cond_txt, group_num=0, hyper_scales=hyper_scales,control_scale=control_scale,training=training,multi=multi)
            if t[0]<self.control_step:
                control = [c * 0. for c, scale in zip(control, self.control_scales)]
            else:
                control = [c * scale for c, scale in zip(control, self.control_scales)]
            eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=control, only_mid_control=self.only_mid_control)            
            # control = self.control_model(x=x_noisy, hint=torch.cat(cond['c_concat'], 1), timesteps=t, context=cond_txt, group_num=0, hyper_scales=hyper_scales,control_scale=control_scale,training=training,multi=multi)
            # if t[0]<self.control_step:
            #     control = [c * 0. for c, scale in zip(control, self.control_scales)]
            # else:
            #     control = [c * scale for c, scale in zip(control, self.control_scales)]
            # eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=control, only_mid_control=self.only_mid_control)

            # controls = self.control_model(x=x_noisy, hint=torch.cat(cond['c_concat'], 1), timesteps=t, context=cond_txt, group_num=0)
            # epss = []
            # for i in range(len(controls[0])):
            #     control = [item[i] for item in controls]
            #     if t[0]<self.control_step:
            #         control = [c * 0. for c, scale in zip(control, self.control_scales)]
            #     else:
            #         control = [c * scale for c, scale in zip(control, self.control_scales)]
            #     eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=control, only_mid_control=self.only_mid_control)
            #     epss.append(eps)
            # eps = epss[0] #0.7*epss[0] + 0.1*epss[1] + 0.1*epss[2] + 0.1*epss[3]  #torch.mean(torch.stack(epss), dim=0)  #0.7*epss[0] + 0.1*epss[1] + 0.1*epss[2] + 0.1*epss[3]

        return eps
    
    def apply_model_train(self, x_noisy, t, cond, *args, **kwargs):
        assert isinstance(cond, dict)
        diffusion_model = self.model.diffusion_model
        # print('t:',t)

        cond_txt = torch.cat(cond['c_crossattn'], 1)

        if cond['c_concat'] is None:
            eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=None, only_mid_control=self.only_mid_control)
        else:
            control = self.control_model(x=x_noisy, hint=torch.cat(cond['c_concat'], 1), timesteps=t, context=cond_txt,training=True)
            if t[0]<self.control_step:
                control = [c * 0. for c, scale in zip(control, self.control_scales)]
            else:
                control = [c * scale for c, scale in zip(control, self.control_scales)]
            eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=control, only_mid_control=self.only_mid_control)
        return eps    

    @torch.no_grad()
    def get_unconditional_conditioning(self, N):
        return self.get_learned_conditioning([""] * N)

    @torch.no_grad()
    def log_images(self, batch, N=4, n_row=2, sample=False, ddim_steps=50, ddim_eta=0.0, return_keys=None,
                   quantize_denoised=True, inpaint=True, plot_denoise_rows=False, plot_progressive_rows=True,
                   plot_diffusion_rows=False, unconditional_guidance_scale=7.5, unconditional_guidance_label=None,
                   use_ema_scope=True,
                   **kwargs):
        use_ddim = ddim_steps is not None

        log = dict()
        z, c = self.get_input(batch, self.first_stage_key, bs=N)
        c_cat, c = c["c_concat"][0][:N], c["c_crossattn"][0][:N]
        N = min(z.shape[0], N)
        n_row = min(z.shape[0], n_row)
        log["reconstruction"] = self.decode_first_stage(z)
        log["control"] = c_cat * 2.0 - 1.0
        log["conditioning"] = log_txt_as_img((512, 512), batch[self.cond_stage_key], size=16)

        if plot_diffusion_rows:
            # get diffusion row
            diffusion_row = list()
            z_start = z[:n_row]
            for t in range(self.num_timesteps):
                if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
                    t = repeat(torch.tensor([t]), '1 -> b', b=n_row)
                    t = t.to(self.device).long()
                    noise = torch.randn_like(z_start)
                    z_noisy = self.q_sample(x_start=z_start, t=t, noise=noise)
                    diffusion_row.append(self.decode_first_stage(z_noisy))

            diffusion_row = torch.stack(diffusion_row)  # n_log_step, n_row, C, H, W
            diffusion_grid = rearrange(diffusion_row, 'n b c h w -> b n c h w')
            diffusion_grid = rearrange(diffusion_grid, 'b n c h w -> (b n) c h w')
            diffusion_grid = make_grid(diffusion_grid, nrow=diffusion_row.shape[0])
            log["diffusion_row"] = diffusion_grid

        if sample:
            # get denoise row
            samples, z_denoise_row = self.sample_log(cond={"c_concat": [c_cat], "c_crossattn": [c]},
                                                     batch_size=N, ddim=use_ddim,
                                                     ddim_steps=ddim_steps, eta=ddim_eta)
            x_samples = self.decode_first_stage(samples)
            log["samples"] = x_samples
            if plot_denoise_rows:
                denoise_grid = self._get_denoise_row_from_list(z_denoise_row)
                log["denoise_row"] = denoise_grid

        if unconditional_guidance_scale >= -1.0:
            uc_cross = self.get_unconditional_conditioning(N)
            uc_cat = c_cat  # torch.zeros_like(c_cat)
            uc_full = {"c_concat": [uc_cat], "c_crossattn": [uc_cross]}
            for slct in self.slct:
                self.control_model.input_hint_block_new[0].cond = slct['cond']
                suffix = slct['suffix']
                print(suffix)
                samples_cfg, _ = self.sample_log(cond={"c_concat": [c_cat], "c_crossattn": [c]},
                                                batch_size=N, ddim=use_ddim,
                                                ddim_steps=ddim_steps, eta=ddim_eta,
                                                unconditional_guidance_scale=unconditional_guidance_scale,
                                                unconditional_conditioning=uc_full,training=True
                                                )
                x_samples_cfg = self.decode_first_stage(samples_cfg)
                deconv = self.control_model.input_hint_block_new[0].deconv(c_cat)
                log[f"{suffix}_samples_cfg_scale_{unconditional_guidance_scale:.2f}"] = x_samples_cfg
                log[f"{suffix}_deconv_samples_cfg_scale_{unconditional_guidance_scale:.2f}"] = deconv
                # self.control_model.input_hint_block_new[0].cond = None

        return log

    @torch.no_grad()
    def sample_log(self, cond, batch_size, ddim, ddim_steps, training=False,**kwargs):
        ddim_sampler = DDIMSampler(self)
        b, c, h, w = cond["c_concat"][0].shape
        shape = (self.channels, h // 8, w // 8)
        samples, intermediates = ddim_sampler.sample(ddim_steps, batch_size, shape, cond, verbose=False,training=training, **kwargs)
        return samples, intermediates

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.control_model.parameters())
        if not self.sd_locked:
            params += list(self.model.diffusion_model.output_blocks.parameters())
            params += list(self.model.diffusion_model.out.parameters())
        opt = torch.optim.AdamW(params, lr=lr)
        return opt

    def low_vram_shift(self, is_diffusing):
        if is_diffusing:
            self.model = self.model.cuda()
            self.control_model = self.control_model.cuda()
            self.first_stage_model = self.first_stage_model.cpu()
            self.cond_stage_model = self.cond_stage_model.cpu()
        else:
            self.model = self.model.cpu()
            self.control_model = self.control_model.cpu()
            self.first_stage_model = self.first_stage_model.cuda()
            self.cond_stage_model = self.cond_stage_model.cuda()

    @torch.no_grad()
    def select(self):
        self.slct = [{'cond':[i],'suffix':f'{i}'} for i in self.hypercond]