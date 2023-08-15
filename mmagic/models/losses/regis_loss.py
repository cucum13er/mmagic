# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmagic.registry import MODELS
from .loss_wrapper import masked_loss

# _reduction_modes = ['none', 'mean', 'sum']

@MODELS.register_module()
class Registration_loss(nn.Module):
    """Registration loss for Reformer.

    Args:
        loss_weight (float): Loss weight for registration loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
        sample_wise (bool): Whether calculate the loss sample-wise. This
            argument only takes effect when `reduction` is 'mean' and `weight`
            (argument of `forward()`) is not None. It will first reduce loss
            with 'mean' per-sample, and then it means over all the samples.
            Default: False.
    """

    def __init__(self,
                 loss_weight: float = 1.0,
                 padding_size: int = 16,
                 patch_size: int = 16,
                 # reduction: str = 'mean',
                 ) -> None:
        super().__init__()
        # if reduction not in ['none', 'mean', 'sum']:
        #     raise ValueError(f'Unsupported reduction mode: {reduction}. '
        #                      f'Supported ones are: {_reduction_modes}')
        
        self.loss_weight = loss_weight
        self.padding_size = padding_size # padding_size is also the search range
        self.patch_size = patch_size
        self.unfoldLR = torch.nn.Unfold(kernel_size=self.patch_size, stride=self.patch_size)
        ref_patch_size = self.patch_size+2*self.padding_size
        self.unfoldRef = torch.nn.Unfold(kernel_size=ref_patch_size, stride=self.patch_size)
        # self.normlization = F.normalize()
        
    def forward(self,
                LR: torch.Tensor,
                Ref: torch.Tensor,
                **kwargs) -> torch.Tensor:
        """Forward Function.

        Args:
            LR (Tensor): of shape (B, C, H, W). LR feature map with padding.
            Ref (Tensor): of shape (B, C, H, W). Reference feature map.
            weight (Tensor, optional): of shape (B, C, H, W). Element-wise
                weights. Default: None.
        """
        # breakpoint()
        # crop LR to no-padding
        B, C, H, W = LR.shape
        LR_crop = LR[:,:,self.padding_size:H-self.padding_size, self.padding_size:W-self.padding_size]
        # crop LR to patch size
        # unfoldLR = torch.nn.Unfold(kernel_size=self.patch_size, stride=self.patch_size)
        LR_unfold = self.unfoldLR(LR_crop).permute(0,2,1).reshape(B,-1,self.patch_size,self.patch_size)
                        
        # crop Ref to padding_size + patch_size + padding_size
        ref_patch_size = self.patch_size+2*self.padding_size
        # unfoldRef = torch.nn.Unfold(kernel_size=ref_patch_size, stride=self.patch_size)
        Ref_unfold = self.unfoldRef(Ref).permute(0,2,1).reshape(B,-1,ref_patch_size,ref_patch_size)
                     
        outputs = []
        for ref, lr in zip(Ref_unfold, LR_unfold):
            # get the similarity between each patch pair in LR and ref-HR, 
            assert ref.shape[0] == lr.shape[0], 'check your patch unfolding!'
            ref_norm_factor = torch.sqrt(torch.sum(torch.square(ref), dim=(1,2))).unsqueeze(1).unsqueeze(1)+1e-12
            lr_norm_factor = torch.sqrt(torch.sum(torch.square(lr), dim=(1,2))).unsqueeze(1).unsqueeze(1)+1e-12
            im_sim =  F.conv2d((ref/ref_norm_factor).unsqueeze(0), (lr/lr_norm_factor).unsqueeze(0).permute(1,0,2,3), groups=lr.shape[0])
            im_sim = torch.max_pool2d(im_sim, kernel_size=im_sim.shape[2:])
            outputs.append(im_sim)
        # desired similatiry is 1
        outputs = 1 - torch.stack(outputs).mean()
        return self.loss_weight * outputs

