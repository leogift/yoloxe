#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) Hanqtech Inc. All rights reserved.
import torch
from torch import nn

from yoloxe.models.network_blocks import get_activation, C2PPLayer, SelfTransformer
from yoloxe.utils import initialize_weights

import ssl
context = ssl._create_unverified_context()
ssl._create_default_https_context = ssl._create_unverified_context
import torchvision

class Resnet(nn.Module):
    def __init__(
        self,
        model="resnet18",
        model_reduce=1, # 模型深度缩减
        act="silu",
        pp_repeats=0,
        transformer=False,
        heads=16,
        drop_rate=0.,
    ):
        super().__init__()

        # 加载预训练模型
        self.model = torchvision.models.__dict__[model](weights='IMAGENET1K_V1')

        # 丢弃分类头
        del self.model.avgpool
        del self.model.fc
        # 组合stem
        self.model.stem = nn.Sequential(
            self.model.conv1,
            self.model.bn1,
            self.model.relu,
            self.model.maxpool,
        )
        # 模型深度缩减
        self.output_channels = []
        new_lens = []
        self.model.trunk_output = [self.model.layer1, self.model.layer2, self.model.layer3, self.model.layer4]
        for idx in range(self.model.trunk_output.__len__()):
            if model_reduce > 1:
                old_len = self.model.trunk_output[idx].__len__()
                del_len = old_len*(model_reduce-1)//model_reduce
                if del_len > 0:
                    del self.model.trunk_output[idx][-del_len:]
            new_lens.append(self.model.trunk_output[idx].__len__())
            self.output_channels.append(self.model.trunk_output[idx][-1].conv2.out_channels)
        print(f"{model} {model_reduce}: LENS={new_lens}, CHANNELS={self.output_channels}")

        # 训练参数
        for p in self.model.parameters():
            p.requires_grad = True  # for training

        self.drop   = nn.Dropout(drop_rate) if drop_rate > 0. else nn.Identity()

        # last_layer
        self.last_layer = nn.Sequential(
            nn.Identity() if pp_repeats==0 else C2PPLayer(
                self.output_channels[-1],
                self.output_channels[-1],
                n=pp_repeats,
                act=act,
                drop_rate=drop_rate,
            ),
            nn.Identity() if transformer==False else SelfTransformer(
                self.output_channels[-1],
                heads=heads,
                act=act,
                drop_rate=drop_rate,
            ),
        )
        initialize_weights(self.last_layer)


    def forward(self, inputs):
        x = inputs["input"]

        outputs = inputs

        x = self.model.stem(x)

        for idx in range(self.model.trunk_output.__len__()):
            x = self.model.trunk_output[idx](x)
            key = f"backbone{2+idx}"
            outputs[key] = x

        if self.training:
            x = self.drop(x)
        x = self.last_layer(x)
        outputs["backbone5"] = x

        return outputs
