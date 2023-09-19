# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmedit.models.registry import MODELS
from mmselfsup.utils import (batch_shuffle_ddp, batch_unshuffle_ddp,
                             concat_all_gather)
from mmedit.models.builder import build_backbone, build_component, build_loss
from .base import BaseModel


@MODELS.register_module()
class MoCo_label(BaseModel):
    """MoCo.

    Implementation of `Momentum Contrast for Unsupervised Visual
    Representation Learning <https://arxiv.org/abs/1911.05722>`_.
    Part of the code is borrowed from:
    `<https://github.com/facebookresearch/moco/blob/master/moco/builder.py>`_.

    Args:
        backbone (dict): Config dict for module of backbone.
        neck (dict): Config dict for module of deep features to compact
            feature vectors. Defaults to None.
        head (dict): Config dict for module of loss functions.
            Defaults to None.
        queue_len (int): Number of negative keys maintained in the queue.
            Defaults to 65536.
        feat_dim (int): Dimension of compact feature vectors. Defaults to 128.
        momentum (float): Momentum coefficient for the momentum-updated
            encoder. Defaults to 0.999.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 head=None,
                 queue_len=65536,
                 feat_dim=128,
                 momentum=0.999,
                 init_cfg=None,
                 **kwargs):
        super(MoCo_label, self).__init__(init_cfg)
        assert neck is not None
        self.encoder_q = nn.Sequential(
            build_backbone(backbone), build_component(neck))
        self.encoder_k = nn.Sequential(
            build_backbone(backbone), build_component(neck))

        for param_q, param_k in zip(self.encoder_q.parameters(),
                                    self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        self.backbone = self.encoder_q[0]
        self.neck = self.encoder_q[1]
        assert head is not None
        self.head = build_loss(head)

        self.queue_len = queue_len
        self.momentum = momentum

        # create the queue
        self.register_buffer('queue', torch.randn(feat_dim, queue_len))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer('queue_ptr', torch.zeros(1, dtype=torch.long))
        self.register_buffer('queue_label', torch.ones(1, queue_len, dtype=torch.long)*99999999)
    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """Momentum update of the key encoder."""
        for param_q, param_k in zip(self.encoder_q.parameters(),
                                    self.encoder_k.parameters()):
            param_k.data = param_k.data * self.momentum + \
                           param_q.data * (1. - self.momentum)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, labels):
        """Update queue."""
        # gather keys before updating queue
        keys = concat_all_gather(keys)
        labels = concat_all_gather(labels)
        # breakpoint()
        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.queue_len % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.transpose(0, 1)
        self.queue_label[:,ptr:ptr + batch_size] = labels
        ptr = (ptr + batch_size) % self.queue_len  # move pointer

        self.queue_ptr[0] = ptr
        # breakpoint()
    def extract_feat(self, img):
        """Function to extract features from backbone.

        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.

        Returns:
            tuple[Tensor]: backbone outputs.
        """
        x = self.backbone(img)
        # x = self.encoder_k(img)
        return x

    def forward_train(self, img, label, **kwargs):###add label
        """Forward computation during training.

        Args:
            img (list[Tensor]): A list of input images with shape
                (N, C, H, W). Typically these should be mean centered
                and std scaled.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        # assert isinstance(img, list)
        feat = self.backbone(img)
        N, _, _, _ = img.shape
        im_q = img[:int(N/2), :, :, :]
        im_k = img[int(N/2):, :, :, :]        
        label = label[:int(N/2)]
        # im_q = img[0]
        # im_k = img[1]
        # print(im_q,'11111111\n')
        # print(im_k,'22222222\n')
        # breakpoint()
        # exit()
        # compute query features
        q = self.encoder_q(im_q)[0]  # queries: NxC
        q = nn.functional.normalize(q, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            # update the key encoder
            self._momentum_update_key_encoder()

            # shuffle for making use of BN
            im_k, idx_unshuffle = batch_shuffle_ddp(im_k)

            k = self.encoder_k(im_k)[0]  # keys: NxC
            k = nn.functional.normalize(k, dim=1)

            # undo shuffle
            k = batch_unshuffle_ddp(k, idx_unshuffle)
        # print(label)            
        # breakpoint()
        # q = 256*128, k = 256*128, 256: batchsize 128:feature length
        self._dequeue_and_enqueue(k, label)
        s = torch.matmul(q, self.queue.clone().detach())  # similarity matrix between query and queue
        
        
        # # compute logits
        # # Einstein sum is more intuitive
        # # positive logits: Nx1
        # l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # # negative logits: NxK
        # l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
        
        # breakpoint()
        losses = self.head(s, label, queue_label=self.queue_label)

        # update the queue
        
        
        return losses, feat
