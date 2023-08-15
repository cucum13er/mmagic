#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  9 09:45:06 2023

@author: rui
"""

# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# path1 = '/home/rui/Rui_registration/datasets/testdata1/0028.tif'
# regis = [1,0,100, 0,1,100, 0,0,1]
# regist= np.reshape(np.array(regis), (3,3)).astype(np.float32)

# im1 = cv2.imread(path1,cv2.IMREAD_GRAYSCALE)
# H, W = im1.shape
# warped = cv2.warpPerspective(im1, regist, (W,H))
# # plt.figure()
# # plt.imshow(im1, cmap='gray')
# # plt.figure()
# # plt.imshow(warped, cmap='gray')


# import kornia.geometry as kg
# import torch
# im3d = torch.from_numpy(im1).repeat(3,1,1).unsqueeze(0).type(torch.FloatTensor)
# regist = torch.from_numpy(regist).unsqueeze(0)
# warped1 = kg.transform.warp_perspective(im3d, regist, (H,W))

# trans_im = warped1.squeeze(0).numpy().transpose((1,2,0)).astype(np.uint8)
# plt.figure()
# plt.imshow(trans_im)


# from reformer_net import Reformer
# import torch

# regis = torch.stack((torch.eye(3),torch.eye(3)), 0)
# regis[0,0,2] = 100
# regis[1,0,2] = -100
# print('The prior registration matrices are: ', regis)

# inp_img = {'imgLR': torch.rand(2,3,112,112), 'imgRef': torch.rand(2,3,224,224)}
# print('The input image LRs shapes are:', inp_img['imgLR'].shape)
# print('The input reference image HRs shapes are:', inp_img['imgRef'].shape)

# net = Reformer(prior_regis=regis)

# y = net(inp_img)

# import torch
# from reformer_utils import PatchEmbed, PatchUnEmbed


# tmp = torch.ones(36)
# tmp1 = torch.stack((tmp,2*tmp,3*tmp,4*tmp),0)
# tmp2 = 10*tmp1
# inp_img = torch.stack((tmp1,tmp2),0)
# net = PatchUnEmbed(8, 4, embed_dim=36)
# y = net(inp_img)
# # y = inp_img.permute(0,2,1).view(2,3,16,4).reshape(2,3,4,4,2,2).permute(0,1,4,3,5,2).reshape(2,3,8,8)


# net = PatchEmbed(8, 4, embed_dim=48)
# x = net(y)


# import kornia.geometry as kg
# import kornia.feature as kf
# import cv2
# import torch
# path1 = '/home/rui/Rui_registration/datasets/testdata1/0028.tif'
# path2 = '/home/rui/Rui_registration/datasets/testdata1/0029.tif'
# im1 = cv2.imread(path1,cv2.IMREAD_GRAYSCALE)
# im2 = cv2.imread(path2,cv2.IMREAD_GRAYSCALE)

# im1_3d = torch.from_numpy(im1).repeat(3,1,1).unsqueeze(0).type(torch.FloatTensor)
# im2_3d = torch.from_numpy(im2).repeat(3,1,1).unsqueeze(0).type(torch.FloatTensor)
# sift = kf.SIFTDescriptor(41, rootsift=True)
# detector = kf.ScaleSpaceDetector(num_features=400,
#                       # resp_module=resp,
#                       # scale_space_response=True,#We need that, because DoG operates on scale-space
#                       # nms_module=nms,
#                       # scale_pyr_module=scale_pyr,
#                       # ori_module=kornia.feature.LAFOrienter(19),
#                       # mr_size=6.0,
#                       # minima_are_also_good=True
#                       )

# afs, resps = detector(im1_3d.float())

# from scipy.io import loadmat 
# import numpy as np
# import torch
# M = loadmat('/media/rui/Samsung4TB/Datasets/testdata2_choose/regis.mat')
# regis = M['M']
# regis = torch.tensor(np.stack(regis[0,:]))

# import torch
# import matplotlib.pyplot as plt
# import numpy as np
# lrs = torch.load('/home/rui/Rui_SR/mmagic_Rui/lrs.pt')
# refs = torch.load('/home/rui/Rui_SR/mmagic_Rui/refs.pt')

# for i, (lr, ref) in enumerate(zip(lrs, refs)):
#     lr = lr.permute((1,2,0)).numpy()
#     ref = ref.permute((1,2,0)).numpy()
#     ref_norm = (ref - np.min(ref))/np.ptp(ref)
#     plt.imshow(lr)
#     # plt.show()
#     plt.savefig(f'/home/rui/Rui_SR/mmagic_Rui/lr{i}.png')
#     plt.imshow(ref_norm)
#     # plt.show()
#     plt.savefig(f'/home/rui/Rui_SR/mmagic_Rui/ref{i}.png')
#     # breakpoint()


## for testing loss function: unfold and convolution
import torch
import torch.nn.functional as F 

num = torch.arange(256).float()
num1 = torch.arange(24*24).float()
LR = torch.reshape(num, (1,1,16,16))
LR = torch.cat((LR,LR,LR), 0)
Ref = torch.reshape(num1, (1,1,24,24))
Ref = torch.cat((Ref,Ref,Ref), 0)
unfold_LR = torch.nn.Unfold(kernel_size=(4,4), dilation=1, stride=4)
unfold_Ref = torch.nn.Unfold(kernel_size=(12,12), dilation=1, stride=4)
out_LR = unfold_LR(LR)
B, C, HW = out_LR.shape
out_LR = out_LR.permute(0,2,1).reshape(B,-1, 4, 4)
out_Ref = unfold_Ref(Ref).permute(0,2,1).reshape(B,-1, 12, 12)

# out_LR = F.normalize(out_LR, dim=(2,3))
# out_Ref = F.normalize(out_Ref, dim=(2,3))
print(out_LR.shape)
print(out_Ref.shape)
res = []
for lr, ref in zip(out_LR, out_Ref):
    print(lr.shape, ref.shape)
    ref_norm_factor = torch.sqrt(torch.sum(torch.square(ref), dim=(1,2))).unsqueeze(1).unsqueeze(1)
    lr_norm_factor = torch.sqrt(torch.sum(torch.square(lr), dim=(1,2))).unsqueeze(1).unsqueeze(1)
    
    im_sim =  F.conv2d((ref/ref_norm_factor).unsqueeze(0), (lr/lr_norm_factor).unsqueeze(0).permute(1,0,2,3), groups=16)
    im_sim = torch.max_pool2d(im_sim, kernel_size=im_sim.shape[2:])
    res.append(im_sim )
    # print(res.shape)
print(res[0].shape)
res = 1- torch.cat(res).mean()
# outputs = F.conv2d(out_Ref, out_LR)
print(res.shape)








