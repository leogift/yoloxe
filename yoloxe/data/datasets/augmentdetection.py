#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import random

import cv2
import numpy as np

from yoloxe.utils import adjust_points_anns

from yoloxe.data.data_augment import random_affine
from yoloxe.data.datasets.datasets_wrapper import Dataset


def get_mosaic_coordinate(mosaic_index, xc, yc, w, h, input_h, input_w):
    # TODO update doc
    # index0 to top left part of image
    if mosaic_index == 0:
        x1, y1, x2, y2 = max(xc - w, 0), max(yc - h, 0), xc, yc
        small_coord = w - (x2 - x1), h - (y2 - y1), w, h
    # index1 to top right part of image
    elif mosaic_index == 1:
        x1, y1, x2, y2 = xc, max(yc - h, 0), min(xc + w, input_w * 2), yc
        small_coord = 0, h - (y2 - y1), min(w, x2 - x1), h
    # index2 to bottom left part of image
    elif mosaic_index == 2:
        x1, y1, x2, y2 = max(xc - w, 0), yc, xc, min(input_h * 2, yc + h)
        small_coord = w - (x2 - x1), 0, w, min(y2 - y1, h)
    # index2 to bottom right part of image
    elif mosaic_index == 3:
        x1, y1, x2, y2 = xc, yc, min(xc + w, input_w * 2), min(input_h * 2, yc + h)  # noqa
        small_coord = 0, 0, min(w, x2 - x1), min(y2 - y1, h)
    return (x1, y1, x2, y2), small_coord


class AugmentDetection(Dataset):
    """Detection dataset wrapper that performs augment for normal dataset."""

    def __init__(
        self, dataset, img_size, enable_aug=True, preproc=None,
        degrees=0.0, 
        translate=0.0,
        shear=0.0,
        scale=(1, 1),
        mosaic_prob=0.0,
        mixup_prob=0.0,
        num_kpts=0,
        *args
    ):
        """
        Args:
            dataset(Dataset) : Pytorch dataset object.
            img_size (tuple):
            enable_aug (bool): enable augments or not.
            preproc (func):
            degrees (float):
            translate (float):
            shear (float):
            scale (tuple):
            mosaic_prob (float):
            mixup_prob (float):
            *args(tuple) : Additional arguments for mixup random sampler.
        """
        super().__init__(img_size, enable_aug=enable_aug)
        self._dataset = dataset
        self.preproc = preproc
        self.enable_aug = enable_aug

        self.degrees = degrees
        self.translate = translate
        self.shear = shear
        self.scale = scale

        self.mosaic_prob = mosaic_prob
        self.mixup_prob = mixup_prob

        self.num_kpts = num_kpts

    def __len__(self):
        return len(self._dataset)

    @Dataset.wrap_getitem
    def __getitem__(self, idx):
        input_dim = self._dataset.input_dim
        input_h, input_w = input_dim[0], input_dim[1]

        # no aug
        if not self.enable_aug:
            img, labels, img_info, img_id = self._dataset.pull_item(idx)

            img, labels = random_affine(
                img,
                targets=labels,
                target_size=(input_w, input_h),
                degrees=0,
                translate=0,
                scales=(1,1),
                shear=0,
                num_kpts=self.num_kpts
            )
            
        # mosaic
        elif random.random() < self.mosaic_prob:
            mosaic_labels = []

            # yc, xc = s, s  # mosaic center x, y
            yc = int(random.uniform(0.5 * input_h, 1.5 * input_h))
            xc = int(random.uniform(0.5 * input_w, 1.5 * input_w))

            # 3 additional image indices
            indices = [idx] + [random.randint(0, len(self._dataset) - 1) for _ in range(3)]

            for i_mosaic, index in enumerate(indices):
                img, _labels, _, img_id = self._dataset.pull_item(index)
                h0, w0 = img.shape[:2]  # orig hw
                scale = min(1. * input_h / h0, 1. * input_w / w0)

                inter_modes = [cv2.INTER_LINEAR, cv2.INTER_NEAREST]
                inter_idx = random.randint(0,len(inter_modes)-1)

                img = cv2.resize(
                    img, (int(w0 * scale), int(h0 * scale)), interpolation=inter_modes[inter_idx]
                )
                
                # generate output mosaic image
                (h, w, c) = img.shape[:3]
                if i_mosaic == 0:
                    mosaic_img = np.full((input_h * 2, input_w * 2, c), 114, dtype=np.uint8)

                # suffix l means large image, while s means small image in mosaic aug.
                (l_x1, l_y1, l_x2, l_y2), (s_x1, s_y1, s_x2, s_y2) = get_mosaic_coordinate(
                    i_mosaic, xc, yc, w, h, input_h, input_w
                )
                mosaic_img[l_y1:l_y2, l_x1:l_x2] = img[s_y1:s_y2, s_x1:s_x2]

                padw, padh = l_x1 - s_x1, l_y1 - s_y1

                labels = _labels.copy()
                # Normalized xywh to pixel xyxy format
                if _labels.size > 0:
                    labels[:, 0] = scale * _labels[:, 0] + padw
                    labels[:, 1] = scale * _labels[:, 1] + padh
                    labels[:, 2] = scale * _labels[:, 2] + padw
                    labels[:, 3] = scale * _labels[:, 3] + padh
                    if self.num_kpts > 0:
                        for ikpt in range(self.num_kpts):
                            labels[:, 5+ikpt*2] = scale * _labels[:, 5+ikpt*2] + padw
                            labels[:, 5+ikpt*2+1] = scale * _labels[:, 5+ikpt*2+1] + padh
                mosaic_labels.append(labels)

            if len(mosaic_labels):
                mosaic_labels = np.concatenate(mosaic_labels, 0)

            img, labels = mosaic_img, mosaic_labels

        # other augs
        else:
            img, labels, img_info, img_id = self._dataset.pull_item(idx)

            # -----------------------------------------------------------------
            # CopyPaste: https://arxiv.org/abs/2012.07177
            # -----------------------------------------------------------------
            if (
                not len(labels) == 0
                and random.random() < self.mixup_prob
            ):
                img, labels = self.mixup(img, labels, self.input_dim)
    
            img, labels = random_affine(
                img,
                targets=labels,
                target_size=(input_w, input_h),
                degrees=self.degrees,
                translate=self.translate,
                scales=self.scale,
                shear=self.shear,
                num_kpts=self.num_kpts
            )

        # common
        if self.preproc:
            img, labels = self.preproc(img, labels, self.input_dim)
        
        img_info = (img.shape[1], img.shape[0])

        return img, labels, img_info, img_id


    def mixup(self, origin_img, origin_labels, input_dim):
        cp_labels = []
        while len(cp_labels) == 0:
            cp_index = random.randint(0, self.__len__() - 1)
            cp_labels = self._dataset.load_anno(cp_index)
        img, cp_labels, _, _ = self._dataset.pull_item(cp_index)

        if len(img.shape) == 3:
            cp_img = np.ones((input_dim[0], input_dim[1], 3), dtype=np.uint8) * 114
        else:
            cp_img = np.ones(input_dim, dtype=np.uint8) * 114
        
        cp_scale_ratio = min(input_dim[0] / img.shape[0], input_dim[1] / img.shape[1])

        inter_modes = [cv2.INTER_LINEAR, cv2.INTER_NEAREST]
        inter_idx = random.randint(0,len(inter_modes)-1)

        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * cp_scale_ratio), int(img.shape[0] * cp_scale_ratio)),
            interpolation=inter_modes[inter_idx],
        )
        
        cp_img[
            : int(img.shape[0] * cp_scale_ratio), : int(img.shape[1] * cp_scale_ratio)
        ] = resized_img

        origin_h, origin_w = cp_img.shape[:2]
        target_h, target_w = origin_img.shape[:2]
        padded_img = np.zeros(
            (max(origin_h, target_h), max(origin_w, target_w), 3), dtype=np.uint8
        )
        padded_img[:origin_h, :origin_w] = cp_img

        x_offset, y_offset = 0, 0
        if padded_img.shape[0] > target_h:
            y_offset = random.randint(0, padded_img.shape[0] - target_h - 1)
        if padded_img.shape[1] > target_w:
            x_offset = random.randint(0, padded_img.shape[1] - target_w - 1)
        padded_cropped_img = padded_img[
            y_offset: y_offset + target_h, x_offset: x_offset + target_w
        ]

        cp_bboxes_origin_np = adjust_points_anns(
            cp_labels[:, :4].copy(), cp_scale_ratio, 0, 0
        )
        cp_bboxes_transformed_np = cp_bboxes_origin_np.copy()
        cp_bboxes_transformed_np[:, 0::2] = cp_bboxes_transformed_np[:, 0::2] - x_offset
        cp_bboxes_transformed_np[:, 1::2] = cp_bboxes_transformed_np[:, 1::2] - y_offset
        box_labels = cp_bboxes_transformed_np

        cls_labels = cp_labels[:, 4:5].copy()

        if self.num_kpts > 0:
            cp_kpts_origin_np = adjust_points_anns(
                cp_labels[:, 5:5+self.num_kpts*2].copy(), cp_scale_ratio, 0, 0
            )
            cp_kpts_transformed_np = cp_kpts_origin_np.copy()
            cp_kpts_transformed_np[:, 0::2] = cp_kpts_transformed_np[:, 0::2] - x_offset
            cp_kpts_transformed_np[:, 1::2] = cp_kpts_transformed_np[:, 1::2] - y_offset
            kpts_labels = cp_kpts_transformed_np
            kpts_conf_labels = cp_labels[:, 5+self.num_kpts*2:].copy()

            labels = np.hstack((box_labels, cls_labels, kpts_labels, kpts_conf_labels))

        else:
            labels = np.hstack((box_labels, cls_labels))

        origin_labels = np.vstack((origin_labels, labels))
        origin_img = origin_img.astype(np.float32)
        origin_img = 0.5 * origin_img + 0.5 * padded_cropped_img.astype(np.float32)
        origin_img = origin_img.astype(np.uint8)

        return origin_img, origin_labels
