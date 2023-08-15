# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule
from mmagic.registry import MODELS
# from swinir_modules import PatchEmbed, PatchUnEmbed
from .reformer_utils import Upsample2D, PatchEmbed, PatchUnEmbed
from .reformer_module import TransformerBlock, TransformerBlock_cross, Warpping
from mmagic.structures import DataSample
from typing import Dict, List, Optional, Union
@MODELS.register_module()
class Reformer(BaseModule):
    """Reformer
    Ref repo:

    Args:
        inp_channels (int): Number of input image channels. Default: 3.
        out_channels (int): Number of output image channels: 3.
        dim (int): Number of feature dimension. Default: 64.
        num_blocks (List(int)): Depth of each Transformer layer.
            Default: [8, 8].
        heads (List(int)): Number of attention heads in different layers.
            Default: [8, 8].
        dim: embedding dimensions of the Transformer blocks, 
            Default: 768 (16*16*3 image patch)
        image_size: Default: 224 (input image default is 224*224)
        patch_size: Default: 16 (cropping patch size default is 16*16)
        ffn_expansion_factor (float): Ratio of feed forward network expansion.
            Default: 4.
        prior_regis: the prior registration result between the ref-HR and upsampled-LR (no scaling)
        bias (bool): The bias of convolution. Default: False
        dual_keys (List): Keys of both images in inputs.
            Default: ['imgLR', 'imgRef'].
    """

    def __init__(self,
                 inp_channels=3,
                 out_channels=3,
                 num_blocks=[8, 8],
                 heads=[8, 8],
                 dim = 768, # to be determined
                 image_size = 224,
                 patch_size = 16,
                 ffn_expansion_factor=4,                 
                 bias=False,
                 scale = 2,
                 # keys=['lr', 'ref', 'hr', 'prior_regis'],
                 ):

        super(Reformer, self).__init__()
        
        self.upLR = Upsample2D(inp_channels, scale=scale)
        self.patch_embed = PatchEmbed(inp_channels, patch_size)
        self.image_size = image_size
        self.patch_size = patch_size
        self.scale = scale
        # self-attention for both LR and Ref
        self.encoder_both = nn.Sequential(*[
            TransformerBlock( 
                dim = dim,                
                num_heads=heads[0],
                mlp_ratio=ffn_expansion_factor,
                qkv_bias=bias,
                ) for i in range(num_blocks[0])
        ])
        # cross attention for LR and Ref
        self.decoder = nn.Sequential(*[
            TransformerBlock_cross(
                dim = dim,
                num_heads=heads[1],
                mlp_ratio=ffn_expansion_factor,
                qkv_bias=bias,
                ) for i in range(num_blocks[1])
        ])
        
        # self.keys = keys

        self.restore = nn.Sequential( 
                            PatchUnEmbed(image_size, patch_size),
                                    )
        self.warp = Warpping(
                             image_size=self.image_size, 
                             patch_size=self.patch_size, 
                             embed_dim=dim, 
                             in_channels=3
                             )
        self.output = PatchUnEmbed(self.image_size, self.patch_size, embed_dim=dim, im_channels=inp_channels)
        
    def forward(self, inp_img):
        """Forward function.

        Args:
            inp_img (Tensor): Input tensor with shape (B, C, H, W).
        Returns:
            Tensor: Forward results.
        """
        # breakpoint()
        LR = inp_img['lr']
        Ref = inp_img['ref']
        prior_regis = inp_img['regis']
        LR = self.upLR(LR)
        # inp_img = torch.cat((LRup,Ref), dim=1)
        
        _, _, h, w = Ref.shape
        # breakpoint()
        assert h == LR.shape[2] and w == LR.shape[3]
        
        if h % 8 == 0:
            padding_h = 0
        else:
            padding_h = 8 - h % 8
        if w % 8 == 0:
            padding_w = 0
        else:
            padding_w = 8 - w % 8

        LR = F.pad(LR, (0, padding_w, 0, padding_h), 'reflect')
        Ref = F.pad(Ref,(0, padding_w, 0, padding_h), 'reflect')
        # encode the LR
        inp_enc_level1_LR = self.patch_embed(LR)
        # print('The embedded LR shape is: ', inp_enc_level1_LR.shape)
        # breakpoint()
        out_enc_level1_LR = self.encoder_both(inp_enc_level1_LR)
        LRfeats = self.output(out_enc_level1_LR)
        # print('The encoded LR shape is: ', out_enc_level1_LR.shape)
        # breakpoint()
        # encode the Ref HR
        inp_enc_level1_Ref = self.patch_embed(Ref)
        out_enc_level1_Ref = self.encoder_both(inp_enc_level1_Ref)
        # print('The encoded Ref-HR shape is: ', out_enc_level1_Ref.shape)
        # warp the reference image and embed the features
        # breakpoint()
        shift_enc_Ref = self.warp(out_enc_level1_Ref, prior_regis) 
        Reffeats = shift_enc_Ref
        shift_enc_Ref = self.patch_embed(shift_enc_Ref)
        cat_inp = torch.cat((out_enc_level1_LR, shift_enc_Ref), 1)
        cross_out = self.decoder(cat_inp)   
        # print('The cross attention output shape is: ', cross_out.shape)
        # cross_out = cross_out + out_enc_level1_LR
        _, N, _ = cross_out.shape
        cross_out = cross_out[:,:N//2,:]
        out = self.output(cross_out) 
        # print('The final output shape is: ', out.shape)
        return out, LRfeats, Reffeats

    # def convert_to_datasample(self, predictions: DataSample,
    #                           data_samples: DataSample,
    #                           # inputs: Optional[torch.Tensor]
    #                           ) -> List[DataSample]:
    #     """Add predictions and destructed inputs (if passed) to data samples.

    #     Args:
    #         predictions (DataSample): The predictions of the model.
    #         data_samples (DataSample): The data samples loaded from
    #             dataloader.
    #         inputs (Optional[torch.Tensor]): The input of model. Defaults to
    #             None.

    #     Returns:
    #         List[DataSample]: Modified data samples.
    #     """

    #     # if inputs is not None:
    #     #     destructed_input = self.data_preprocessor.destruct(
    #     #         inputs, data_samples, ['lr','hr',])
    #     #     data_samples.set_tensor_data({'input': destructed_input})
    #     # split to list of data samples
    #     data_samples = data_samples.split()
    #     predictions = predictions.split()

    #     for data_sample, pred in zip(data_samples, predictions):
    #         data_sample.output = pred

    #     return data_samples



