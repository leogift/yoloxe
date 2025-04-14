#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Hanqtech Inc. All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F

import loguru as logger

from yoloxe.utils import initialize_weights

from yoloxe.models.network_blocks import BaseConv
from yoloxe.models.losses import UncertaintyLoss, OKSLoss

from yoloxe.models.head.yolo_head import YOLOHead

class YOLOKPTSHead(YOLOHead):
    def __init__(
        self,
        num_classes=1,
        in_features=("pan3", "pan4", "pan5"),
        in_channels=[256, 512, 1024],
        strides=[8, 16, 32],
        scales=[8, 16, 32],
        act="silu",
        drop_rate=0.,
        softmax=False,
        aux_head=False,
        num_kpts=0,
        kpts_weight=None
    ):
        super().__init__(
            num_classes=num_classes,
            in_features=in_features,
            in_channels=in_channels,
            strides=strides,
            scales=scales,
            act=act,
            drop_rate=drop_rate,
            softmax=softmax,
            aux_head=aux_head
        )

        assert num_kpts > 0, "num_kpts should be greater than 0."
        self.num_kpts = num_kpts
        self.kpts_weight = kpts_weight

        self.kpts_stems = nn.ModuleList()
        self.kpts_convs = nn.ModuleList()
        self.kpts_preds = nn.ModuleList()
        self.kpts_conf_preds = nn.ModuleList()

        for i in range(len(in_channels)):
            self.kpts_stems.append(
                BaseConv(
                    int(in_channels[i]),
                    int(in_channels[0] / 2),
                    ksize=3,
                    stride=1,
                    act=act,
                )
            )
            self.kpts_convs.append(
                BaseConv(
                    in_channels=int(in_channels[0] / 2),
                    out_channels=int(in_channels[0] / 2),
                    ksize=3,
                    stride=1,
                    act=act,
                )
            )
            self.kpts_preds.append(
                nn.Conv2d(
                    in_channels=int(in_channels[0] / 2),
                    out_channels=self.num_kpts * 2,
                    kernel_size=1,
                    bias=True)
            )
            self.kpts_conf_preds.append(
                nn.Conv2d(
                    in_channels=int(in_channels[0] / 2),
                    out_channels=self.num_kpts ,
                    kernel_size=1,
                    bias=True)
            )

        initialize_weights(self)
        
        # loss
        self.kpts_oks_loss = OKSLoss(self.num_kpts, kpts_weight=self.kpts_weight, reduction="none")

        if not self.aux_head:
            # uncertainty loss
            self.kpts_uncertainty = UncertaintyLoss()
            self.kpts_conf_uncertainty = UncertaintyLoss()


    def forward(self, inputs, kwargs={}):
        outputs = []
        x_shifts = []
        y_shifts = []
        expanded_strides = []

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

            if self.training:
                output = torch.cat([reg_xywh, obj_conf, cls_conf, kpts_xy, kpts_conf], 1)
                output, _, grid, _ = \
                    self.get_output_and_grid(output, stride_this_level, scale_this_level, x.type())
                x_shifts.append(grid[:, :, 0])
                y_shifts.append(grid[:, :, 1])
                expanded_strides.append(
                    torch.zeros(1, grid.shape[1])
                    .fill_(stride_this_level)
                    .type_as(x)
                )

            elif not self.aux_head:
                # (4+1+CLASSES)*H*W
                if self.softmax:
                    output = torch.cat(
                        [reg_xywh, torch.sigmoid(obj_conf), torch.softmax(cls_conf, dim=1), kpts_xy, torch.sigmoid(kpts_conf)], 1
                    )
                else:
                    output = torch.cat(
                        [reg_xywh, torch.sigmoid(obj_conf), torch.sigmoid(cls_conf), kpts_xy, torch.sigmoid(kpts_conf)], 1
                    )

            outputs.append(output)

        if self.training:
            return (
                x_shifts,
                y_shifts,
                expanded_strides,
                torch.cat(outputs, 1),
                xin[0].dtype
            )
        
        elif not self.aux_head:
            self.hw = [x.shape[-2:] for x in outputs]
            outputs = torch.cat(
                [x.flatten(start_dim=2) for x in outputs], dim=2
            )
            outputs = outputs.permute(0, 2, 1)

            if self.decode_in_inference:
                outputs, _, _, _ = self.decode_outputs(outputs, dtype=xin[0].type())

            return outputs


    def get_output_and_grid(self, output, stride, scale, dtype):
        output, stride, grid, scale = \
            super().get_output_and_grid(output, stride, scale, dtype)

        for ikpt in range(self.num_kpts):
            start_idx = 5 + self.num_classes + 2*ikpt
            end_idx = start_idx + 2
            output[..., start_idx : end_idx] = ((output[..., start_idx : end_idx]-0.5)*4)*scale + (grid+0.5) * stride # [-2, 2]*scale + grid_center*stride

        return output, stride, grid, scale


    def decode_outputs(self, outputs, dtype):
        outputs, strides, grids, scales = \
            super().decode_outputs(outputs, dtype)

        for ikpt in range(self.num_kpts):
            start_idx = 5 + self.num_classes + 2*ikpt
            end_idx = start_idx + 2
            outputs[..., start_idx : end_idx] = ((outputs[..., start_idx : end_idx]-0.5)*4)*scales + (grids+0.5) * strides

        return outputs, strides, grids, scales


    # loss calculation
    def get_losses(
        self,
        params_list,
        targets
    ):
        (
            x_shifts, \
            y_shifts, \
            expanded_strides, \
            outputs, \
            dtype
        ) = params_list
        labels = targets

        bbox_preds = outputs[:, :, :4]  # [batch, n_anchors_all, 4]
        obj_preds = outputs[:, :, 4:5]  # [batch, n_anchors_all, 1]
        cls_preds = outputs[:, :, 5:5+self.num_classes]  # [batch, n_anchors_all, n_cls]
        kpts_preds = outputs[:, :, 5+self.num_classes:5+self.num_classes+self.num_kpts*2]  # [batch, n_anchors_all, n_kpts*2]
        kpts_conf_preds = outputs[:, :, 5+self.num_classes+self.num_kpts*2:]  # [batch, n_anchors_all, 1]

        # calculate targets
        nlabel = (labels.sum(dim=2) > 0).sum(dim=1)  # number of objects

        total_num_anchors = outputs.shape[1]
        x_shifts = torch.cat(x_shifts, 1)  # [1, n_anchors_all]
        y_shifts = torch.cat(y_shifts, 1)  # [1, n_anchors_all]
        expanded_strides = torch.cat(expanded_strides, 1)

        cls_targets = []
        bbox_targets = []
        obj_targets = []
        kpts_targets = []
        kpts_conf_targets = []
        fg_masks = []

        num_fg = 0.0
        num_gts = 0.0

        for batch_idx in range(outputs.shape[0]):
            num_gt = int(nlabel[batch_idx])
            num_gts += num_gt
            if num_gt == 0:
                cls_target = outputs.new_zeros((0, self.num_classes))
                bbox_target = outputs.new_zeros((0, 4))
                obj_target = outputs.new_zeros((total_num_anchors, 1))
                kpts_target = outputs.new_zeros((0, 2*self.num_kpts))
                kpts_conf_target = outputs.new_zeros((0, self.num_kpts))
                fg_mask = outputs.new_zeros(total_num_anchors).bool()
            else:
                gt_classes = labels[batch_idx, :num_gt, 0]
                gt_bboxes_per_image = labels[batch_idx, :num_gt, 1:5]
                gt_kpts_per_image = labels[batch_idx, :num_gt, 5:5+self.num_kpts*2]
                gt_kpts_conf_per_image = labels[batch_idx, :num_gt, 5+self.num_kpts*2:]
                bboxes_preds_per_image = bbox_preds[batch_idx]

                try:
                    (
                        gt_matched_classes,
                        fg_mask,
                        pred_ious_this_matching,
                        matched_gt_inds,
                        num_fg_img,
                    ) = self.get_assignments(  # noqa
                        batch_idx,
                        num_gt,
                        gt_bboxes_per_image,
                        gt_classes,
                        bboxes_preds_per_image,
                        expanded_strides,
                        x_shifts,
                        y_shifts,
                        cls_preds,
                        obj_preds,
                    )

                except RuntimeError as e:
                    # TODO: the string might change, consider a better way
                    if "CUDA out of memory. " not in str(e):
                        raise  # RuntimeError might not caused by CUDA OOM

                    logger.error(
                        "OOM RuntimeError is raised due to the huge memory cost during label assignment. \
                           CPU mode is applied in this batch. If you want to avoid this issue, \
                           try to reduce the batch size or image size."
                    )
                    torch.cuda.empty_cache()
                    (
                        gt_matched_classes,
                        fg_mask,
                        pred_ious_this_matching,
                        matched_gt_inds,
                        num_fg_img,
                    ) = self.get_assignments(  # noqa
                        batch_idx,
                        num_gt,
                        gt_bboxes_per_image,
                        gt_classes,
                        bboxes_preds_per_image,
                        expanded_strides,
                        x_shifts,
                        y_shifts,
                        cls_preds,
                        obj_preds,
                        "cpu",
                    )

                torch.cuda.empty_cache()
                num_fg += num_fg_img

                cls_target = F.one_hot(
                    gt_matched_classes.to(torch.int64), self.num_classes
                ) * (pred_ious_this_matching**0.5).unsqueeze(-1)
                obj_target = fg_mask.unsqueeze(-1)
                bbox_target = gt_bboxes_per_image[matched_gt_inds]
                kpts_target = gt_kpts_per_image[matched_gt_inds]
                kpts_conf_target = gt_kpts_conf_per_image[matched_gt_inds]

            cls_targets.append(cls_target)
            bbox_targets.append(bbox_target)
            obj_targets.append(obj_target.to(dtype))
            kpts_targets.append(kpts_target)
            kpts_conf_targets.append(kpts_conf_target)
            fg_masks.append(fg_mask)

        cls_targets = torch.cat(cls_targets, 0)
        bbox_targets = torch.cat(bbox_targets, 0)
        obj_targets = torch.cat(obj_targets, 0)
        kpts_targets = torch.cat(kpts_targets, 0)
        kpts_conf_targets = torch.cat(kpts_conf_targets, 0)
        fg_masks = torch.cat(fg_masks, 0)

        num_fg = max(num_fg, 1)
        iou_loss = (
            self.iou_loss(bbox_preds.view(-1, 4)[fg_masks], bbox_targets)
        ).sum() / num_fg
        obj_loss = (
            self.obj_loss(obj_preds.view(-1, 1), obj_targets)
        ).sum() / num_fg
        cls_loss = (
            self.cls_loss(
                cls_preds.view(-1, self.num_classes)[fg_masks], cls_targets
            )
        ).sum() / num_fg

        kpts_loss, kpts_conf_loss = self.kpts_oks_loss(
            kpts_preds.view(-1, 2*self.num_kpts)[fg_masks], kpts_conf_preds.view(-1, self.num_kpts)[fg_masks],
            kpts_targets, kpts_conf_targets, bbox_targets
        )
        kpts_loss = kpts_loss.sum() / num_fg
        kpts_conf_loss = kpts_conf_loss.sum() / num_fg

        if not self.aux_head:
            # uncertainty loss
            iou_loss = self.bbox_uncertainty(iou_loss)
            obj_loss = self.obj_uncertainty(obj_loss)
            cls_loss = self.cls_uncertainty(cls_loss)
            kpts_loss = self.kpts_uncertainty(kpts_loss)
            kpts_conf_loss = self.kpts_conf_uncertainty(kpts_conf_loss)

        loss = 5*iou_loss + obj_loss + cls_loss \
            + 5*kpts_loss + kpts_conf_loss

        return {
            "total_loss": loss,
            "iou_loss": 5*iou_loss,
            "obj_loss": obj_loss,
            "cls_loss": cls_loss,
            "kpts_loss": 5*kpts_loss,
            "kpts_conf_loss": kpts_conf_loss,
        }
