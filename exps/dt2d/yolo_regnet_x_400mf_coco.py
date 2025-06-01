#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import os
import torch
import torch.nn as nn

from yoloxe.exp import Exp as BaseExp

from loguru import logger

from yoloxe.utils.checkpoint import get_missing_parameters_message, get_unexpected_parameters_message

from yoloxe.utils.anchor import COCO2Anchors

_CKPT_FULL_PATH = "weights/yolo_regnet_x_400mf_coco.pth"

class Exp(BaseExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        self.num_classes = 80

        self.data_dir = "COCO"

        self.train_ann = "instances_train2017.json"
        self.train_name = "train2017/"
        self.val_ann = "instances_val2017.json"
        self.val_name = "val2017/"

        self.input_size = (512, 512)  # (height, width)
        self.test_size = (512, 512)

        self.basic_lr_per_img = 0.001 / 64.0

        self.act = "relu"
        self.max_epoch = 120

        self.model_name = "regnet_x_400mf"

        self.warmup_epochs = 10
        self.no_aug_epochs = 10
        self.data_num_workers = 8
        self.eval_epoch_interval = 5


    def get_model(self):

        if "model" not in self.__dict__:
            from yoloxe.models import YOLOXE, \
            BaseNorm, Regnet, YOLONeckFPN, RegnetNeckPAN, \
            YOLOHead, \
            C2aLayer, C2kLayer

            scales = None

            if scales is None:
                from yoloxe.data.dataloading import get_yoloxe_datadir
                data_dir = os.path.join(get_yoloxe_datadir(), self.data_dir)
                data_dir = os.path.join(data_dir, "annotations")
                anno_dir = os.path.join(data_dir, self.val_ann)
                anchor_dir = os.path.join(data_dir, "anchor.txt")
                scales = COCO2Anchors(anno_dir, self.test_size, 3)
                print("scales",scales)
                with open(anchor_dir, "w") as f:
                    f.writelines(str(scales))
                    f.close()

            pp_repeats = 0 if min(self.test_size[0], self.test_size[1])//32 <= 4 \
                else (min(self.test_size[0], self.test_size[1])//32 - 4)//6 + 1

            preproc = BaseNorm(trainable=True)
            backbone = Regnet(
                self.model_name,
                act=self.act,
                pp_repeats=pp_repeats, 
                transformer=True,
                drop_rate=0.1,
            )
            self.channels = backbone.output_channels[-3:]
            neck = nn.Sequential(*[
                YOLONeckFPN(
                    in_channels=self.channels,
                    act=self.act,
                    layer_type=C2aLayer,
                    simple_reshape=True,
                    n=1
                ),
                RegnetNeckPAN(
                    self.model_name,
                    in_channels=self.channels,
                    act=self.act, 
                    layer_type=C2kLayer,
                    n=2
                )
            ])
            head = YOLOHead(
                self.num_classes,
                in_channels=self.channels,
                scales=scales, 
                act=self.act, 
                drop_rate=0.1,
                softmax=True,
            )

            aux_head_list = [
                YOLOHead(
                    self.num_classes,
                    in_features=("fpn3", "fpn4", "fpn5"),
                    in_channels=self.channels,
                    scales=scales, 
                    act=self.act,
                    aux_head=True,
                ),
            ]

            self.model = YOLOXE(
                preproc=preproc,
                backbone=backbone,  
                neck=neck, 
                head=head,
                aux_head_list=aux_head_list,
            )

        ckpt = torch.load(_CKPT_FULL_PATH, map_location="cpu", weights_only=False)
        if "model" in ckpt:
            ckpt = ckpt["model"]

        for k in list(ckpt.keys()):
            if "pred" in k \
                or "loss" in k or "Loss" in k:
                del ckpt[k]

        incompatible = self.model.load_state_dict(ckpt, strict=False)
        logger.info("missing_keys:")
        if incompatible.missing_keys:
            logger.info(
                get_missing_parameters_message(incompatible.missing_keys)
            )

        logger.info("unexpected_keys:")
        if incompatible.unexpected_keys:
            logger.info(
                get_unexpected_parameters_message(incompatible.unexpected_keys)
            )

        return self.model
