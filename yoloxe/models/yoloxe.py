#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.

from loguru import logger

import torch.nn as nn

class YOLOXE(nn.Module):
    """
    YOLOXE model module. The module list is defined by create_yolo_modules function.
    The network returns loss values from three YOLO layers during training
    and detection results during test.
    """

    def __init__(self, preproc=None, backbone=None, neck=None, head=None, aux_list=[]):
        super().__init__()

        # preproc
        self.preproc = nn.Identity() if preproc is None else preproc
        # backbone
        self.backbone = nn.Identity() if backbone is None else backbone
        # neck
        self.neck = nn.Identity() if neck is None else neck
        self.head = nn.Identity() if head is None else head
        self.aux_list = nn.ModuleList()
        for aux in aux_list:
            self.aux_list.append(aux)

    def forward(self, x, targets=None, kwargs={}):
        inputs = {"input":x}
        inputs = self.preproc(inputs)
        inputs = self.backbone(inputs)
        inputs = self.neck(inputs)

        if self.training:
            assert targets is not None

            outputs = self.head(inputs, kwargs)
            losses = self.head.get_losses(outputs, targets)

            for aux in self.aux_list:
                aux_outputs = aux(inputs, kwargs)
                aux_losses = self.head.get_losses(aux_outputs, targets)
                if "total_loss" in losses and "total_loss" in aux_losses:
                    losses["total_loss"] += aux_losses["total_loss"]*0.1
            
            outputs = losses

        else:
            outputs = self.head(inputs)

        return outputs
