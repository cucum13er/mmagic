# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import os
from typing import Optional

from mmengine.dataset import BaseDataset
from mmengine.fileio import get_file_backend

from mmagic.registry import DATASETS
# import kornia.geometry as kg
from scipy.io import loadmat 
import torch
import numpy as np

IMG_EXTENSIONS = ('.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm',
                  '.PPM', '.bmp', '.BMP', '.tif', '.TIF', '.tiff', '.TIFF')


@DATASETS.register_module()
class RefImageDataset(BaseDataset):
    """General paired reference HR, LR, and Ground-truth HR dataset, 
       including the perspective transformation matrices

    It assumes that the training directory is '/path/to/data/train'.
    During test time, the directory is '/path/to/data/test'. '/path/to/data'
    can be initialized by args 'dataroot'. 
    
    Each sample contains a pair of
    images concatenated in the w dimension (A|B). to be determined

    Args:
        dataroot (str | :obj:`Path`): Path to the folder root of paired images.
        pipeline (List[dict | callable]): A sequence of data transformations.
            {
                only available for flip, transpose
                }
        test_mode (bool): Store `True` when building test dataset.
            Default: `False`.
        test_dir (str): Subfolder of dataroot which contain test images.
            Default: 'test'.
    """

    def __init__(self,
                 data_root,
                 pipeline,
                 scale = 2,
                 io_backend: Optional[str] = None,
                 test_mode=False,
                 test_dir='test'):
        # phase = test_dir if test_mode else 'train'
        # self.data_root = osp.join(str(data_root), phase)
        self.scale = scale
        self.data_root = data_root
        if io_backend is None:
            self.file_backend = get_file_backend(uri=data_root)
        else:
            self.file_backend = get_file_backend(
                backend_args={'backend': io_backend})
        # breakpoint()
        super().__init__(
            data_root=self.data_root, pipeline=pipeline, test_mode=test_mode)
        # self.data_infos = self.load_annotations()
        # breakpoint()
    def load_data_list(self):
        """Load paired image paths.

        Returns:
            list[dict]: List that contains paired image paths.
        """
        data_infos = []
        LRs, Ref, HRs = self.scan_folder(self.data_root)
        if 'regis.mat' in os.listdir(self.data_root):
            M = loadmat('/media/rui/Samsung4TB/Datasets/testdata2_choose/regis.mat')
            regis = M['M']
            regis = torch.tensor(np.stack(regis[0,:]))
        else:
            regis = None
        # breakpoint()
        # pair_paths = sorted(self.scan_folder(self.data_root))
        if regis is not None:
            for lr, hr, reg in zip(LRs,HRs,regis):
                data_infos.append(dict(lr_path=lr, ref_path=Ref[0], gt_img_path=hr, regis=reg))
        else: 
            for lr, hr in zip(LRs, HRs):
                data_infos.append(dict(lr_path=lr, ref_path=Ref[0], gt_img_path=hr))
            
        # for pair_path in pair_paths:
        #     data_infos.append(dict(pair_path=pair_path))
        # breakpoint()    
        return data_infos


    def scan_folder(self, path):
        """Obtain image path list (including sub-folders) from a given folder.

        Args:
            path (str | :obj:`Path`): Folder path.

        Returns:
            list[str]: Image list obtained from the given folder.
        """
        files = os.listdir(path)
        folders = [osp.join(path, f) for f in files if osp.isdir(osp.join(path, f))]
            
        assert len(folders) >= 2, 'Dataset should contain 2 or 3 folders: HR(optional), ref-HR, and LR'
        lr_f = [f for f in folders if 'lr' in f.lower()][0]
        ref_f = [f for f in folders if 'ref' in f.lower()][0]
        hr_f = [f for f in folders if 'hr' in f.lower()][0]
        
        LRs = self.file_backend.list_dir_or_file(lr_f, list_dir=False, suffix=IMG_EXTENSIONS, recursive=True)
        LRimgs = sorted([osp.join(lr_f, n ) for n in LRs] )
        
        Ref_HR = self.file_backend.list_dir_or_file(ref_f, list_dir=False, suffix=IMG_EXTENSIONS, recursive=True)
        Refimg = sorted([osp.join(ref_f, n ) for n in Ref_HR] )
        if hr_f:
            HRs = self.file_backend.list_dir_or_file(hr_f, list_dir=False, suffix=IMG_EXTENSIONS, recursive=True)
            HRimgs = sorted([osp.join(hr_f, n ) for n in HRs] )
        # imgs_list = self.file_backend.list_dir_or_file(
        #     path, list_dir=False, suffix=IMG_EXTENSIONS, recursive=True)
        # images = [self.file_backend.join_path(path, img) for img in imgs_list]
        assert LRimgs and Refimg, f'{path} has no valid image file.'
        return LRimgs, Refimg, HRimgs