# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Union
import torch
from mmengine.registry import HOOKS
from mmengine.hooks import Hook
# from mmagic.models.editors.reformer.reformer_module import Warpping
DATA_BATCH = Optional[Union[dict, tuple, list]]

from mmcv.transforms import BaseTransform
import numpy as np
import kornia.geometry as kg
import kornia 
import torch.nn.functional as F
import cv2
import operator
class Reformer_RandomCrop(BaseTransform):
    """Paired random crop.

    It crops a pair of LR, Ref, and HR with corresponding locations.
    It also supports accepting lr list and hr list.
    Required keys are "scale", "lq_key", and "gt_key", "ref_key",
    added or modified keys are "lq_key" and "gt_key", "ref_key".

    Args:
        hr_patch_size (int): cropped hr patch size.
        padding_size: How many extra pixels the reference hr will give (each side). Default: 16
        lq_key (str): Key of LQ img. Default: 'lr'.
        gt_key (str): Key of GT img. Default: 'hr'.
        ref_key (str): Key of ref-hr img. Default: 'ref'.
    """

    def __init__(self, 
                 hr_patch_size, 
                 padding_size=16,
                 lq_key='lr', 
                 gt_key='gt_img', 
                 ref_key='ref', 
                 scale=None,
                 # feature_size = 41,
                 ):

        self.hr_patch_size = hr_patch_size
        self.padding_size = padding_size    
        self.lq_key = lq_key
        self.gt_key = gt_key
        self.ref_key = ref_key
        self.scale = scale
        # self.patch_size = feature_size
    def regis(self, lrs, refs, regis):

        # upsample = torch.nn.Upsample(scale_factor=(self.scale, self.scale) )
        refs = torch.stack(refs,0).type(torch.FloatTensor)
        regis = [reg.inverse() for reg in regis]
        regis = torch.stack(regis,0).type(torch.FloatTensor)
        B,_,H,W = refs.shape
        # breakpoint()
        refs = kg.transform.warp_perspective(refs, regis, (H,W))

        # for lr, ref, reg in zip(lrs, refs, regis):            
            # lr_up = upsample(lr.unsqueeze(0))
            # warped = kg.transform.warp_perspective(ref, reg, )
        # breakpoint()
        return refs
    def transform(self, results):
        """Transform function.

        Args:
            results (dict): A dict containing the necessary information and
                data for augmentation.

        Returns:
            dict: A dict containing the processed data and information.
        """
        # breakpoint()
        data_samples = results['data_samples']
        results = results['inputs']
    	# modified by Rui        
        if self.scale == None:
            if 'scale' in results.keys():
        	    scale = results['scale']	    
            else:
        	    scale = 1
        else:
            scale = self.scale
        
        
        lq_patch_size = self.hr_patch_size // scale

        lq_is_list = isinstance(results[self.lq_key], list)
        if not lq_is_list:
            results[self.lq_key] = [results[self.lq_key]]
        gt_is_list = isinstance(results[self.gt_key], list)
        if not gt_is_list:
            results[self.gt_key] = [results[self.gt_key]]

        _, h_lq, w_lq = results[self.lq_key][0].shape
        _, h_gt, w_gt = results[self.gt_key][0].shape

        if h_gt != h_lq * scale or w_gt != w_lq * scale:
            raise ValueError(
                f'Scale mismatches. GT ({h_gt}, {w_gt}) is not {scale}x '
                f'multiplication of LQ ({h_lq}, {w_lq}).')
        if h_lq < lq_patch_size or w_lq < lq_patch_size:
            raise ValueError(
                f'LQ ({h_lq}, {w_lq}) is smaller than patch size '
                f'({lq_patch_size}, {lq_patch_size}). Please check '
                f'{results[f"{self.lq_key}_path"]} and '
                f'{results[f"{self.gt_key}_path"]}.')

        # find the correspondences and homography matrix between lr 
        # and ref-hr
        # input: lrs, ref-hrs
        # output: lrs, transformed ref-hrs in lr*scale's view
        # (e.g., if lr is 96*96 and upsample sacle is 2, ref-hr is 192*192)
        # breakpoint()
        if 'regis' in results.keys():
            T_ref = self.regis(results[self.lq_key], results[self.ref_key], results['regis'])
        else:
            # to be done using registration!
            raise ValueError('need to register the lq and ref online! impelment later!')
        # randomly choose top and left coordinates for img patch
        for i, (lr, hr, ref)  in enumerate(zip(results[self.lq_key], results[self.gt_key], T_ref) ):
            # finding a random position at lq image    
            top = np.random.randint(low=self.padding_size//scale + 1, high=h_lq - lq_patch_size - self.padding_size//scale + 1)
            left = np.random.randint(low=self.padding_size//scale + 1, high=w_lq - lq_patch_size - self.padding_size//scale + 1)
            lr_pad_size = self.padding_size//scale
            results[self.lq_key][i] = F.pad(lr[..., top:top + lq_patch_size, left:left + lq_patch_size], (lr_pad_size,lr_pad_size,lr_pad_size,lr_pad_size)) 
            top_hr, left_hr = int(top * scale), int(left * scale)
            results[self.gt_key][i] = F.pad(hr[..., top_hr:top_hr + self.hr_patch_size, left_hr:left_hr + self.hr_patch_size], (self.padding_size,self.padding_size,self.padding_size,self.padding_size))
            # make the ref-hr bigger (same as the padded hr) 
            top_ref, left_ref = top_hr-self.padding_size, left_hr-self.padding_size
            ref_size = self.hr_patch_size + self.padding_size + self.padding_size
            # print('ref_size: ', ref_size)
            # print('bounding box: ', top_ref, top_ref+ref_size, left_ref, left_ref+ref_size)
            # breakpoint()
            results[self.ref_key][i]= ref[..., top_ref:top_ref + ref_size, left_ref:left_ref + ref_size]
            # breakpoint()
            data_samples[i].__setattr__(self.lq_key, results[self.lq_key][i])
            data_samples[i].__setattr__(self.gt_key, results[self.gt_key][i])
            data_samples[i].__setattr__(self.ref_key, results[self.ref_key][i])
            
        
        if not lq_is_list:
            results[self.lq_key] = results[self.lq_key][0]
        if not gt_is_list:
            results[self.gt_key] = results[self.gt_key][0]
        
        results = dict(inputs=results, data_samples=data_samples)
        
        # breakpoint()
        return results

    def __repr__(self):

        repr_str = self.__class__.__name__
        repr_str += (f'(hr_patch_size={self.hr_patch_size}, '
                     f'lq_key={self.lq_key}, '
                     f'gt_key={self.gt_key})')

        return repr_str
@HOOKS.register_module()
class Reformer_PreRegis(Hook):
    """
    Reformer to get the prior registration results from input LRs and Ref-HR
    
    Input: data batch: {'inputs':{'lr': [tensor, tensor, ...],
                                  'ref': [tensor, tensor, ...],
                                  'hr': [tensor, tensor, ...],
                                  }
                        'data_samples':[
                        <DataSample(

                                    META INFORMATION
                                    sample_idx: 40
                                    lr_path: '/media/rui/Samsung4TB/Datasets/testdata2/LR_bicubic/0059.tif'
                                    hr_path: '/media/rui/Samsung4TB/Datasets/testdata2/HR/0059.tif'
                                    ref_path: '/media/rui/Samsung4TB/Datasets/testdata2/Ref/0021.tif'
                                
                                    DATA FIELDS
                                    ref_img: tensor([[[141, 146, 152,  ...,  90,  84,  98],
                                                 [159, 123, 144,  ..., 106,  72,  86],
                                                 [159, 129, 159,  ...,  85,  90,  99],
                                                 ...,
                                                 [183, 184, 196,  ...,  75,  71,  73],
                                                 [177, 189, 183,  ...,  57,  73,  85],
                                                 [197, 193, 187,  ...,  52,  63,  78]],
                                        
                                                [[141, 146, 152,  ...,  90,  84,  98],
                                                 [159, 123, 144,  ..., 106,  72,  86],
                                                 [159, 129, 159,  ...,  85,  90,  99],
                                                 ...,
                                                 [183, 184, 196,  ...,  75,  71,  73],
                                                 [177, 189, 183,  ...,  57,  73,  85],
                                                 [197, 193, 187,  ...,  52,  63,  78]],
                                        
                                                [[141, 146, 152,  ...,  90,  84,  98],
                                                 [159, 123, 144,  ..., 106,  72,  86],
                                                 [159, 129, 159,  ...,  85,  90,  99],
                                                 ...,
                                                 [183, 184, 196,  ...,  75,  71,  73],
                                                 [177, 189, 183,  ...,  57,  73,  85],
                                                 [197, 193, 187,  ...,  52,  63,  78]]], device='cuda:0',
                                               dtype=torch.uint8)
                                    ) at 0x7fcf248c2d40>, 
                        <DataSample()>, 
                        ...
                           ]                         
                    }
    
    """

    def __init__(self,                  
                 hr_patch_size=192, 
                 padding_size=16,
                 lq_key='lr', 
                 gt_key='gt_img', 
                 ref_key='ref',          
                 ):
        super().__init__()
        self.hr_patch_size = hr_patch_size
        self.padding_size = padding_size
        self.lq_key = lq_key
        self.gt_key = gt_key
        self.ref_key = ref_key 
        
    def before_train_iter(self,
                          runner,
                          batch_idx: int,
                          data_batch: DATA_BATCH = None) -> None:
        
        crop = Reformer_RandomCrop(
                self.hr_patch_size,
                self.padding_size,
                self.lq_key,
                self.gt_key,
                self.ref_key,
                runner.cfg['scale'],        
            )
        
        # breakpoint()   
        data_batch = crop(data_batch)
        # data_batch['inputs'] = crop(data_batch['inputs'])
        # breakpoint()
        # batch_size = len(data_batch['inputs']['lr'])
        # data_batch['inputs']['prior_regis'] = torch.eye(3).unsqueeze(0).repeat(batch_size,1,1)
        
        return None

    def before_val_iter(self,
                        runner,
                        batch_idx: int,
                        data_batch: DATA_BATCH = None) -> None:
        
        crop = Reformer_RandomCrop(
                self.hr_patch_size,
                self.padding_size,
                self.lq_key,
                self.gt_key,
                self.ref_key,
                runner.cfg['scale'],        
            )
        
        # breakpoint()    
        # data_batch['inputs'] = crop(data_batch['inputs'])
        data_batch = crop(data_batch)


    def before_test_iter(self,
                        runner,
                        batch_idx: int,
                        data_batch: DATA_BATCH = None) -> None:
        
        crop = Reformer_RandomCrop(
                self.hr_patch_size,
                self.padding_size,
                self.lq_key,
                self.gt_key,
                self.ref_key,
                runner.cfg['scale'],        
            )
        
        # breakpoint()    
        # data_batch['inputs'] = crop(data_batch['inputs'])
        data_batch = crop(data_batch)





















 
