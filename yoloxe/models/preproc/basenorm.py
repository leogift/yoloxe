#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) Hanqtech Inc. All rights reserved.
import torch
from torch import nn

class BaseNorm(nn.Module):
    def __init__(
        self,
        mean = [114,114,114],
        std = [58,58,58],
    ):
        super().__init__()

        assert len(mean) == len(std)

        self.inversed_std = torch.true_divide(1, torch.tensor(std, dtype=torch.float32))
        self.mean = torch.tensor(mean, dtype=torch.float32)

    def forward(self, inputs):
        x = inputs["input"]

        outputs = inputs
        
        x = (x - self.mean.view(1, -1, 1, 1).to(x.device))*self.inversed_std.view(1, -1, 1, 1).to(x.device)
        
        outputs["input"] = x

        return outputs
