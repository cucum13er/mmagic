#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 20:56:33 2023

@author: rui
"""
import torch
from mmedit.models.backbones.sr_backbones.swinir.swinir_net import SwinIRNet_HA
from mmedit.models.backbones.sr_backbones.swinir.swinir_rstb import *

net1 = SwinIRNet_HA(
                  img_size=64,
                  in_chans=3,
                  embed_dim=180,
                  depths=[6, 6, 6, 6, 6, 6],
                  num_heads=[6, 6, 6, 6, 6, 6],
                  window_size=8,
                  mlp_ratio=2.,
                  use_checkpoint=False,
                  upscale=4,
                  img_range=1.,
                  upsampler='nearest+conv',
                  resi_connection='1conv',
                  fuse_rstb = True,
                  fuse_basic = True,
    
                )
feat = torch.rand([4,512,7,7])
in_img = torch.rand([4,3,64,64])

out = net1(in_img,feat)

# net2 = BasicLayer_fuse(                 
#                  dim=180,
#                  input_resolution=(64,64),
#                  depth=6,
#                  num_heads=6,
#                  window_size=8,
#                  mlp_ratio=4.,
#                  qkv_bias=True,
#                  qk_scale=None,
#                  drop=0.,
#                  attn_drop=0.,
#                  drop_path=0.,
#                  norm_layer=nn.LayerNorm,
#                  downsample=None,
#                  use_checkpoint=False
#                  )

# in_img = torch.rand([4,4096,180])
# feat = torch.rand([4,512,7,7])

# out2 = net2(in_img,(64,64), feat)
