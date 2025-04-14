#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Hanqtech Inc. All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F

import loguru as logger

from yoloxe.utils import meshgrid,initialize_weights, bboxes_iou

from yoloxe.models.network_blocks import BaseConv
from yoloxe.models.losses import IOULoss,FocalLoss,UncertaintyLoss

class YOLOHead(nn.Module):
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
        aux_head=False
    ):
        """
        Args:
            act (str): activation type of conv. Defalut value: "silu".
        """
        super().__init__()
        self.in_features = in_features

        assert num_classes > 0, "num_classes should be greater than 0."
        self.num_classes = num_classes
        self.decode_in_inference = True  # for deploy, set to False

        self.reg_stems = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.obj_preds = nn.ModuleList()

        self.cls_stems = nn.ModuleList()
        self.cls_preds = nn.ModuleList()

        self.dropout = nn.Dropout(drop_rate) if drop_rate > 0. else nn.Identity()

        for i in range(len(in_channels)):
            self.reg_stems.append(
                BaseConv(
                    int(in_channels[i]),
                    int(in_channels[0] / 2),
                    ksize=3,
                    stride=1,
                    act=act,
                )
            )
            self.reg_convs.append(
                BaseConv(
                    in_channels=int(in_channels[0] / 2),
                    out_channels=int(in_channels[0] / 2),
                    ksize=3,
                    stride=1,
                    act=act,
                )
            )
            self.reg_preds.append(
                nn.Conv2d(
                    in_channels=int(in_channels[0] / 2),
                    out_channels=4,
                    kernel_size=1,
                    bias=True)
            )
            self.obj_preds.append(
                nn.Conv2d(
                    in_channels=int(in_channels[0] / 2),
                    out_channels=1,
                    kernel_size=1,
                    bias=True)
            )

            self.cls_stems.append(
                BaseConv(
                    int(in_channels[i]),
                    int(in_channels[0]),
                    ksize=3,
                    stride=1,
                    act=act,
                )
            )
            self.cls_preds.append(
                nn.Conv2d(
                    in_channels=int(in_channels[0]),
                    out_channels=self.num_classes,
                    kernel_size=1,
                    bias=True)
            )

        self.strides = strides
        self.scales = scales
        
        self.softmax = softmax
        self.aux_head = aux_head

        initialize_weights(self)
        
        # loss
        self.cls_loss = nn.BCEWithLogitsLoss(reduction="none")
        self.obj_loss = FocalLoss(reduction="none")
        self.iou_loss = IOULoss(reduction="none")

        if not self.aux_head:
            # uncertainty loss
            self.cls_uncertainty = UncertaintyLoss()
            self.obj_uncertainty = UncertaintyLoss()
            self.bbox_uncertainty = UncertaintyLoss()


    def forward(self, inputs, kwargs={}):
        outputs = []
        x_shifts = []
        y_shifts = []
        expanded_strides = []

        xin = [inputs[f] for f in self.in_features]

        for k, (cls_pred, reg_pred, obj_pred, reg_conv, stride_this_level, scale_this_level, x) in enumerate(
            zip(self.cls_preds, self.reg_preds, self.obj_preds, self.reg_convs, self.strides, self.scales, xin)
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

            if self.training:
                output = torch.cat([reg_xywh, obj_conf, cls_conf], 1)
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
                        [reg_xywh, torch.sigmoid(obj_conf), torch.softmax(cls_conf, dim=1)], 1
                    )
                else:
                    output = torch.cat(
                        [reg_xywh, torch.sigmoid(obj_conf), torch.sigmoid(cls_conf)], 1
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
        batch_size = output.shape[0]
        hsize, wsize = output.shape[-2:]
        yv, xv = meshgrid([torch.arange(hsize), torch.arange(wsize)])
        grid = torch.stack((xv, yv), 2).view(1, hsize, wsize, 2).type(dtype)

        output = output.view(batch_size, 1, -1, hsize, wsize)
        output = output.permute(0, 1, 3, 4, 2).reshape(
            batch_size, hsize * wsize, -1
        )
        grid = grid.view(1, -1, 2)

        output = torch.cat([
            ((output[..., :2]-0.5)*4 + (grid+0.5)) * stride, # ([-2,2] + grid_center) * stride
            (output[..., 2:4] * 4) * scale, # [0,4] * scale
            output[..., 4:],
        ], dim=-1)

        return output, stride, grid, scale


    def decode_outputs(self, outputs, dtype):
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

        outputs = torch.cat([
            ((outputs[..., :2]-0.5)*4 + (grids+0.5)) * strides,
            (outputs[..., 2:4] * 4) * scales,
            outputs[..., 4:]
        ], dim=-1)
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

        # calculate targets
        nlabel = (labels.sum(dim=2) > 0).sum(dim=1)  # number of objects

        total_num_anchors = outputs.shape[1]
        x_shifts = torch.cat(x_shifts, 1)  # [1, n_anchors_all]
        y_shifts = torch.cat(y_shifts, 1)  # [1, n_anchors_all]
        expanded_strides = torch.cat(expanded_strides, 1)

        cls_targets = []
        bbox_targets = []
        obj_targets = []
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
                fg_mask = outputs.new_zeros(total_num_anchors).bool()
            else:
                gt_classes = labels[batch_idx, :num_gt, 0]
                gt_bboxes_per_image = labels[batch_idx, :num_gt, 1:5]
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

            cls_targets.append(cls_target)
            bbox_targets.append(bbox_target)
            obj_targets.append(obj_target.to(dtype))
            fg_masks.append(fg_mask)

        cls_targets = torch.cat(cls_targets, 0)
        bbox_targets = torch.cat(bbox_targets, 0)
        obj_targets = torch.cat(obj_targets, 0)
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

        if not self.aux_head:
            # uncertainty loss
            iou_loss = self.bbox_uncertainty(iou_loss)
            obj_loss = self.obj_uncertainty(obj_loss)
            cls_loss = self.cls_uncertainty(cls_loss)

        loss = 5 * iou_loss + obj_loss + cls_loss

        return {
            "total_loss": loss,
            "iou_loss": 5 * iou_loss,
            "obj_loss": obj_loss,
            "cls_loss": cls_loss,
        }


    @torch.no_grad()
    def get_assignments(
        self,
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
        mode="gpu",
    ):

        if mode == "cpu":
            print("------------CPU Mode for This Batch-------------")
            gt_bboxes_per_image = gt_bboxes_per_image.cpu().float()
            bboxes_preds_per_image = bboxes_preds_per_image.cpu().float()
            gt_classes = gt_classes.cpu().float()
            expanded_strides = expanded_strides.cpu().float()
            x_shifts = x_shifts.cpu()
            y_shifts = y_shifts.cpu()

        fg_mask, geometry_relation = self.get_geometry_constraint(
            gt_bboxes_per_image,
            expanded_strides,
            x_shifts,
            y_shifts,
        )
        
        # NOTE: Fix `selected index k out of range`
        # number of positive anchors
        if fg_mask.sum().item() == 0:
            gt_matched_classes = torch.zeros(0, device=fg_mask.device).long()
            pred_ious_this_matching = torch.rand(0, device=fg_mask.device)
            matched_gt_inds = gt_matched_classes
            num_fg = 0

            if mode == "cpu":
                gt_matched_classes = gt_matched_classes.cuda()
                fg_mask = fg_mask.cuda()
                pred_ious_this_matching = pred_ious_this_matching.cuda()
                matched_gt_inds = matched_gt_inds.cuda()
                num_fg = num_fg.cuda()

            return (
                gt_matched_classes,
                fg_mask,
                pred_ious_this_matching,
                matched_gt_inds,
                num_fg,
            )

        bboxes_preds_per_image = bboxes_preds_per_image[fg_mask]
        cls_preds_ = cls_preds[batch_idx][fg_mask]
        obj_preds_ = obj_preds[batch_idx][fg_mask]
        num_in_boxes_anchor = bboxes_preds_per_image.shape[0]

        if mode == "cpu":
            gt_bboxes_per_image = gt_bboxes_per_image.cpu()
            bboxes_preds_per_image = bboxes_preds_per_image.cpu()
        pair_wise_ious = bboxes_iou(gt_bboxes_per_image, bboxes_preds_per_image, False)
        pair_wise_ious_loss = ((1 - pair_wise_ious).clamp(min=0))**2 # 即使无交集，也要保证iou_loss为正

        gt_cls_per_image = (
            F.one_hot(gt_classes.to(torch.int64), self.num_classes)
            .float()
        )
        if mode == "cpu":
            cls_preds_, obj_preds_ = cls_preds_.cpu(), obj_preds_.cpu()

        with torch.cuda.amp.autocast(enabled=False):
            if self.softmax:
                cls_preds_ = (
                    torch.sigmoid(obj_preds_.float()) * torch.softmax(cls_preds_.float(), dim=1)
                ).sqrt()
            else:
                cls_preds_ = (
                    torch.sigmoid(obj_preds_.float()) * torch.sigmoid(cls_preds_.float())
                ).sqrt()
            pair_wise_cls_loss = F.binary_cross_entropy(
                cls_preds_.unsqueeze(0).repeat(num_gt, 1, 1),
                gt_cls_per_image.unsqueeze(1).repeat(1, num_in_boxes_anchor, 1),
                reduction="none"
            ).sum(-1)
        del cls_preds_

        cost = (
            pair_wise_cls_loss # 0~num_classes
            + 100. * pair_wise_ious_loss # 0~100
            + float(1e4) * (~geometry_relation) # 0 or 1e4
        )
        
        (
            num_fg,
            gt_matched_classes,
            pred_ious_this_matching,
            matched_gt_inds,
        ) = self.simota_matching(cost, pair_wise_ious, gt_classes, num_gt, fg_mask)
        del pair_wise_cls_loss, cost, pair_wise_ious, pair_wise_ious_loss

        if mode == "cpu":
            gt_matched_classes = gt_matched_classes.cuda()
            fg_mask = fg_mask.cuda()
            pred_ious_this_matching = pred_ious_this_matching.cuda()
            matched_gt_inds = matched_gt_inds.cuda()

        return (
            gt_matched_classes,
            fg_mask,
            pred_ious_this_matching,
            matched_gt_inds,
            num_fg,
        )

    def get_geometry_constraint(
        self,
        gt_bboxes_per_image,
        expanded_strides,
        x_shifts,
        y_shifts,
    ):
        expanded_strides_per_image = expanded_strides[0]
        x_centers_per_image = ((x_shifts[0] + 0.5) * expanded_strides_per_image).unsqueeze(0)
        y_centers_per_image = ((y_shifts[0] + 0.5) * expanded_strides_per_image).unsqueeze(0)

        # in fixed center
        center_radius = 2.5
        center_dist = expanded_strides_per_image.unsqueeze(0) * center_radius
        gt_bboxes_per_image_l = (gt_bboxes_per_image[:, 0:1]) - center_dist
        gt_bboxes_per_image_r = (gt_bboxes_per_image[:, 0:1]) + center_dist
        gt_bboxes_per_image_t = (gt_bboxes_per_image[:, 1:2]) - center_dist
        gt_bboxes_per_image_b = (gt_bboxes_per_image[:, 1:2]) + center_dist

        c_l = x_centers_per_image - gt_bboxes_per_image_l
        c_r = gt_bboxes_per_image_r - x_centers_per_image
        c_t = y_centers_per_image - gt_bboxes_per_image_t
        c_b = gt_bboxes_per_image_b - y_centers_per_image
        center_deltas = torch.stack([c_l, c_t, c_r, c_b], 2)
        is_in_centers = center_deltas.min(dim=-1).values > 0.0
        anchor_filter = is_in_centers.sum(dim=0) > 0
        geometry_relation = is_in_centers[:, anchor_filter]

        return anchor_filter, geometry_relation

    def simota_matching(self, cost, pair_wise_ious, gt_classes, num_gt, fg_mask):
        # Dynamic K
        # ---------------------------------------------------------------
        matching_matrix = torch.zeros_like(cost)
        n_candidate_k = min(10, pair_wise_ious.size(1))
        topk_ious, _ = torch.topk(pair_wise_ious, n_candidate_k, dim=1)
        dynamic_ks = torch.clamp(topk_ious.sum(1).int(), 1)
        for gt_idx in range(num_gt):
            _, pos_idx = torch.topk(
                cost[gt_idx], k=dynamic_ks[gt_idx].item(), largest=False
            )
            matching_matrix[gt_idx][pos_idx] = 1
        del topk_ious, dynamic_ks, pos_idx

        anchor_matching_gt = matching_matrix.sum(0)
        # deal with the case that one anchor matches multiple ground-truths
        if anchor_matching_gt.max() > 1:
            multiple_match_mask = anchor_matching_gt > 1
            _, cost_argmin = torch.min(cost[:, multiple_match_mask], dim=0)
            matching_matrix[:, multiple_match_mask] *= 0
            matching_matrix[cost_argmin, multiple_match_mask] = 1
        fg_mask_inboxes = anchor_matching_gt > 0
        num_fg = fg_mask_inboxes.sum().item()

        fg_mask[fg_mask.clone()] = fg_mask_inboxes

        matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0)
        gt_matched_classes = gt_classes[matched_gt_inds]

        pred_ious_this_matching = (matching_matrix * pair_wise_ious).sum(0)[
            fg_mask_inboxes
        ]
        return num_fg, gt_matched_classes, pred_ious_this_matching, matched_gt_inds