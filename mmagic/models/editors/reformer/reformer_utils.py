#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 10 12:25:11 2023

@author: rui
"""

import numbers
import torch
import torch.nn as nn
from einops import rearrange
from mmengine.model import BaseModule
import collections.abc
from itertools import repeat

def to_3d(x):
    """Reshape input tensor."""
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    """Reshape input tensor."""
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(BaseModule):
    """Layer normalization without bias.

    Args:
        normalized_shape (tuple): The shape of inputs.
    """

    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape, )
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (B, C, H, W).

        Returns:
            Tensor: Forward results.
        """
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(BaseModule):
    """Layer normalization with bias. The bias can be learned.

    Args:
        normalized_shape (tuple): The shape of inputs.
    """

    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape, )
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (B, C, H, W).

        Returns:
            Tensor: Forward results.
        """
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(BaseModule):
    """Layer normalization module.

    Note: This is different from the layernorm2d in pytorch.
        The layer norm here can select Layer Normalization type.
    Args:
        dim (int): Channel number of inputs.
        LayerNorm_type (str): Layer Normalization type.
    """

    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (B, C, H, W).

        Returns:
            Tensor: Forward results.
        """
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)




# class Downsample(BaseModule):
#     """Downsample modules.

#     Args:
#         n_feat(int): Channel number of features.
#     """

#     def __init__(self, n_feat):
#         super(Downsample, self).__init__()

#         self.body = nn.Sequential(
#             nn.Conv2d(
#                 n_feat,
#                 n_feat // 2,
#                 kernel_size=3,
#                 stride=1,
#                 padding=1,
#                 bias=False), nn.PixelUnshuffle(2))

#     def forward(self, x):
#         """Forward function.

#         Args:
#             x (Tensor): Input tensor with shape (B, C, H, W).

#         Returns:
#             Tensor: Forward results.
#         """
#         return self.body(x)


class Upsample2D(BaseModule):
    """Upsample modules.

    Args:
        n_feat(int): Channel number of features.
    """

    def __init__(self, n_feat, scale):
        super(Upsample2D, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(
                n_feat,            
                n_feat * scale**2,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False), nn.PixelShuffle(scale))

    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (B, C, H, W).

        Returns:
            Tensor: Forward results.
        """
        return self.body(x)

class PatchEmbed(BaseModule):
    """
    Implement non-overlapping patch embedding
    """
    def __init__(self, image_size, patch_size, embed_dim=768, in_channels=3, dropout=0.):
        super(PatchEmbed, self).__init__()
        self.patch_embedding = nn.Conv2d(in_channels, 
                                         out_channels=embed_dim, 
                                         kernel_size=patch_size,
                                         stride=patch_size,
                                         bias=False)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        # x is a batch of images with shape [N, C, H, W]
        x = self.patch_embedding(x)
        x = x.flatten(2)
        x = x.permute([0,2,1])
        self.dropout(x)
        
        return x

# to be revised and to be determined if needed
class PatchUnEmbed(BaseModule):
    """
    Implement non-overlapping patch unembedding (reconstrcut the image from patches)
    """
    def __init__(self, image_size, patch_size, embed_dim=768, im_channels=3):
        super(PatchUnEmbed, self).__init__()
        image_size = to_2tuple(image_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [
            image_size[0] // patch_size[0], image_size[1] // patch_size[1]
        ]
        if not patch_size[0] * patch_size[1] * im_channels == embed_dim:
            self.proj = nn.Linear(embed_dim, patch_size[0] * patch_size[1] * im_channels)
        else:
            self.proj = nn.Identity()
        self.img_size = image_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]
 
        self.im_chans = im_channels
        self.embed_dim = patch_size[0] * patch_size[1] * im_channels


    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (B, L, C).

        Returns:
            Tensor: Forward results.
        """
        # breakpoint()
        x_size = self.patches_resolution
        x = self.proj(x)
        B, HW, C = x.shape
        # breakpoint()
        x = x.transpose(1, 2).view(B, self.im_chans, self.embed_dim//self.im_chans, HW)  # B im_channels PH*PW HW(meaning how many patches)
        x = x.reshape(B, self.im_chans, self.patch_size[0], self.patch_size[1], 
                      x_size[0], x_size[1]).permute(0,1,4,2,5,3).reshape(
                          B, self.im_chans, self.patch_size[0]*x_size[0], self.patch_size[1]*x_size[1])
        return x


def _ntuple(n):
    """A `to_tuple` function generator. It returns a function, this function
    will repeat the input to a tuple of length ``n`` if the input is not an
    Iterable object, otherwise, return the input directly.

    Args:
        n (int): The number of the target length.
    """

    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return x
        return tuple(repeat(x, n))

    return parse


to_2tuple = _ntuple(2)
    
    
    