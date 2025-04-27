#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Hanqtech Inc. All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F

from yoloxe.utils import meshgrid


from yoloxe.models.head.yolo_kpts_head import YOLOKPTSHead

class YOLOKPTSHeadSplit(YOLOKPTSHead):
    def forward(self, inputs, kwargs={}):
        bbox_outputs = []
        score_outputs = []
        kpts_outputss = []
        kpts_conf_outputs = []

        xin = [inputs[f] for f in self.in_features]

        for k, (cls_pred, reg_pred, obj_pred, reg_conv, kpts_pred, kpts_conf_pred, kpts_conv, stride_this_level, scale_this_level, x) in enumerate(
            zip(self.cls_preds, self.reg_preds, self.obj_preds, self.reg_convs, self.kpts_preds, self.kpts_conf_preds, self.kpts_convs, self.strides, self.scales, xin)
        ):
            cls_stem_x = self.cls_stems[k](x)
            reg_stem_x = self.reg_stems[k](x)
            if not self.aux_head:
                reg_stem_x = reg_conv(reg_stem_x)

            reg_xywh = torch.sigmoid(reg_pred(reg_stem_x))
            obj_conf = obj_pred(reg_stem_x)

            if self.training:
                cls_stem_x = self.dropout(cls_stem_x)
            cls_conf = cls_pred(cls_stem_x)

            kpts_stem_x = self.kpts_stems[k](x)
            if not self.aux_head:
                kpts_stem_x = kpts_conv(kpts_stem_x)
            kpts_xy = torch.sigmoid(kpts_pred(kpts_stem_x))
            kpts_conf = kpts_conf_pred(kpts_stem_x)

            if not self.aux_head:
                bbox_outputs.append(reg_xywh)

                if self.softmax:
                    score = torch.sigmoid(obj_conf) * torch.softmax(cls_conf, dim=1)
                else:
                    score = torch.sigmoid(obj_conf) * torch.sigmoid(cls_conf)
                score_outputs.append(score)

                kpts_outputss.append(kpts_xy)
                kpts_conf_outputs.append(torch.sigmoid(kpts_conf))

        if not self.aux_head:
            self.hw = [x.shape[-2:] for x in bbox_outputs]

            bbox_outputs = torch.cat(
                [x.flatten(start_dim=2) for x in bbox_outputs], dim=2
            )
            score_outputs = torch.cat(
                [x.flatten(start_dim=2) for x in score_outputs], dim=2
            )
            kpts_outputss = torch.cat(
                [x.flatten(start_dim=2) for x in kpts_outputss], dim=2
            )
            kpts_conf_outputs = torch.cat(
                [x.flatten(start_dim=2) for x in kpts_conf_outputs], dim=2
            )

            bbox_outputs = bbox_outputs.permute(0, 2, 1)
            score_outputs = score_outputs.permute(0, 2, 1)
            kpts_outputss = kpts_outputss.permute(0, 2, 1)
            kpts_conf_outputs = kpts_conf_outputs.permute(0, 2, 1)

            outputs = [bbox_outputs, score_outputs, kpts_outputss, kpts_conf_outputs]
            if self.decode_in_inference:
                outputs = self.decode_outputs(
                    *outputs, 
                    dtype=xin[0].type())

            return outputs


    def decode_outputs(self, 
                    *outputs, 
                    dtype):
        grids = []
        strides = []
        scales = []
        for (hsize, wsize), stride, scale in zip(self.hw, self.strides, self.scales):
            yv, xv = meshgrid([torch.arange(hsize), torch.arange(wsize)])
            grid = torch.stack((xv, yv), 2).view(1, -1, 2)
            grids.append(grid)
            shape = grid.shape[:2]
            strides.append(torch.full((*shape, 1), stride))
            scales.append(torch.full((*shape, 1), scale))
        
        grids = torch.cat(grids, dim=1).type(dtype)
        strides = torch.cat(strides, dim=1).type(dtype)
        scales = torch.cat(scales, dim=1).type(dtype)

        bbox_outputs, score_outputs, kpts_outputss, kpts_conf_outputs = outputs

        bbox_outputs[..., :2] = ((bbox_outputs[..., :2]-0.5)*4 + (grids+0.5)) * strides
        bbox_outputs[..., 2:4] = (bbox_outputs[..., 2:4]*4) * scales

        for ikpt in range(self.num_kpts):
            start_idx = 2*ikpt
            end_idx = start_idx + 2
            kpts_outputss[..., start_idx : end_idx] = ((kpts_outputss[..., start_idx : end_idx]-0.5)*4)*scales + (grids+0.5) * strides

        return [bbox_outputs, score_outputs, kpts_outputss, kpts_conf_outputs]
