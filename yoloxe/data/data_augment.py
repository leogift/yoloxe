#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.
"""
Data augmentation functionality. Passed as callable transformations to
Dataset classes.

The data augmentation procedures were interpreted from @weiliu89's SSD paper
http://arxiv.org/abs/1512.02325
"""

import math
import random

import cv2
import numpy as np

from yoloxe.utils import xyxy2cxcywh

# Hrange 42, Srange 212, Vrange 209
def augment_hsv(img, hgain=42/2, sgain=212/2, vgain=209/2):
    hsv_augs = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain]  # random gains
    hsv_augs = hsv_augs.astype(np.int16)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int16)

    img_hsv[..., 0] = (img_hsv[..., 0] + hsv_augs[0]) % 180
    img_hsv[..., 1] = np.clip(img_hsv[..., 1] + hsv_augs[1], 0, 255)
    img_hsv[..., 2] = np.clip(img_hsv[..., 2] + hsv_augs[2], 0, 255)

    return cv2.cvtColor(img_hsv.astype(img.dtype), cv2.COLOR_HSV2BGR)


# 运动模糊
def augment_blur(img, kernel=15, angle=180):
    kernel = abs(kernel)
    angle = abs(angle)

    # be sure the kernel size is odd
    kernel = round(np.random.randint(3, kernel))//2*2+1
    angle = np.random.uniform(-angle, angle)

    M = cv2.getRotationMatrix2D((kernel / 2, kernel / 2), angle, 1)
    motion_blur_kernel = np.diag(np.ones(kernel))
    motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, M, (kernel, kernel))
 
    motion_blur_kernel = motion_blur_kernel / kernel
    blurred = cv2.filter2D(img, -1, motion_blur_kernel)
 
    # convert to uint8
    cv2.normalize(blurred, blurred, 0, 255, cv2.NORM_MINMAX)
    blurred = np.array(blurred, dtype=np.uint8)

    # gaussian blur
    blurred = cv2.GaussianBlur(blurred, ksize=(kernel, kernel), sigmaX=0, sigmaY=0)

    return blurred


# 随机擦除
def augment_erase(img, ratio=0.2):
    ratio = abs(ratio)
    
    H,W = img.shape[:2]

    w = np.random.randint(3, round(W*ratio))
    h = np.random.randint(3, round(H*ratio))
    x = np.random.randint(0, W - w)
    y = np.random.randint(0, H - h)

    img[y:y+h, x:x+w] = 114
    return img


def get_aug_params(value, center=0):
    if isinstance(value, float) or isinstance(value, int):
        return random.uniform(center - value, center + value)
    elif len(value) == 2:
        return random.uniform(value[0], value[1])
    else:
        raise ValueError(
            "Affine params should be either a sequence containing two values\
             or single float values. Got {}".format(value)
        )


def get_affine_matrix(
    target_size,
    degrees=10,
    translate=0.1,
    scales=0.1,
    shear=10,
):
    twidth, theight = target_size

    # Rotation and Scale
    angle = get_aug_params(degrees)
    scale = get_aug_params(scales, center=1.0)

    if scale <= 0.0:
        raise ValueError("Argument scale should be positive")

    R = cv2.getRotationMatrix2D(angle=angle, center=(twidth/2, theight/2), scale=scale)

    M = np.ones([2, 3])
    # Shear
    shear_x = math.tan(get_aug_params(shear) * math.pi / 180)
    shear_y = math.tan(get_aug_params(shear) * math.pi / 180)

    M[0] = R[0] + shear_y * R[1]
    M[1] = R[1] + shear_x * R[0]

    # Translation
    translation_x = get_aug_params(translate) * twidth  # x translation (pixels)
    translation_y = get_aug_params(translate) * theight  # y translation (pixels)

    M[0, 2] += translation_x
    M[1, 2] += translation_y

    return M, scale


def apply_affine_to_bboxes(targets, target_size, M, scale):
    num_gts = len(targets)

    # warp corner points
    corner_points = np.ones((4 * num_gts, 3))
    corner_points[:, :2] = targets[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(
        4 * num_gts, 2
    )  # x1y1, x2y2, x1y2, x2y1
    corner_points = corner_points @ M.T  # apply affine transform
    corner_points = corner_points.reshape(num_gts, 4*2)

    # create new boxes
    corner_xs = corner_points[:, 0::2]
    corner_ys = corner_points[:, 1::2]
    new_bboxes = (
        np.concatenate(
            (corner_xs.min(1), corner_ys.min(1), corner_xs.max(1), corner_ys.max(1))
        )
        .reshape(4, num_gts)
        .T
    )

    targets[:, :4] = new_bboxes

    return targets


def apply_affine_to_kpts(targets, target_size, M, scale, num_kpts=0):
    num_gts = len(targets)

    # warp corner points
    xy_kpts = np.ones((num_kpts * num_gts, 3))
    xy_kpts[:, :2] = targets[:, 5:5+num_kpts*2].reshape(
        num_kpts * num_gts, 2
    )  # x1y1, x2y2, x3y3...
    xy_kpts = xy_kpts @ M.T  # transform
    xy_kpts = xy_kpts.reshape(num_gts, num_kpts*2)

    targets[:, 5:5+num_kpts*2] = xy_kpts

    return targets


def random_affine(
    img,
    targets=(),
    target_size=(416, 416),
    degrees=10,
    translate=0.1,
    scales=0.1,
    shear=10,
    num_kpts=0
):
    M, scale = get_affine_matrix(target_size, degrees, translate, scales, shear)

    img = cv2.warpAffine(img, M, dsize=target_size, borderValue=(114, 114, 114))

    # Transform label coordinates
    if len(targets) > 0:
        targets = apply_affine_to_bboxes(targets, target_size, M, scale)
        if num_kpts > 0:
            targets = apply_affine_to_kpts(targets, target_size, M, scale, num_kpts)

    return img, targets


def _mirror_bboxes(image, boxes, prob=0.5):
    height, width, _ = image.shape
    if random.random() < prob:
        image = image[:, ::-1]
        boxes[:, 0::2] = width - boxes[:, -2::-2]
    
    if random.random() < prob:
        image = image[::-1, :]
        boxes[:, 1::2] = height - boxes[:, -1::-2]

    return image, boxes


def preproc(img, input_size, swap=(2, 0, 1)):
    if len(img.shape) == 3:
        padded_img = np.ones((input_size[0], input_size[1], 3), dtype=np.uint8) * 114
    else:
        padded_img = np.ones(input_size, dtype=np.uint8) * 114

    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])

    inter_modes = [cv2.INTER_LINEAR, cv2.INTER_NEAREST]
    inter_idx = random.randint(0,len(inter_modes)-1)

    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=inter_modes[inter_idx],
    ).astype(np.uint8)

    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img.astype(np.uint8)

    padded_img = padded_img.transpose(swap)
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
    return padded_img, r


class TrainTransform:
    def __init__(self, max_labels=50, 
            flip_prob=0.5, 
            hsv_prob=0.5, 
            blur_prob=0.5,
            erase_prob=0.5,
            num_kpts=0
        ):
        self.max_labels = max_labels
        self.flip_prob = flip_prob
        self.hsv_prob = hsv_prob
        self.blur_prob = blur_prob
        self.erase_prob = erase_prob
        self.num_kpts = num_kpts

    def __call__(self, image, targets, input_dim):
        boxes = targets[:, :4].copy()
        labels = targets[:, 4].copy()
        if self.num_kpts > 0:
            kpts = targets[:, 5:5+self.num_kpts*2].copy()
            kpts_conf = targets[:, 5+self.num_kpts*2:].copy()
            target_size = 5+2*self.num_kpts+self.num_kpts
        else:
            target_size = 5

        if len(boxes) == 0:
            targets = np.zeros((self.max_labels, target_size), dtype=np.float32)
            image, r_o = preproc(image, input_dim)
            return image, targets

        image_o = image.copy()
        targets_o = targets.copy()
        boxes_o = targets_o[:, :4]
        # bbox_o: [xyxy] to [c_x,c_y,w,h]
        boxes_o = xyxy2cxcywh(boxes_o)
        labels_o = targets_o[:, 4]
        if self.num_kpts > 0:
            kpts_o = targets_o[:, 5:5+self.num_kpts*2]
            kpts_conf_o = targets_o[:, 5+self.num_kpts*2:]

        if random.random() < self.hsv_prob:
            image = augment_hsv(image)
        if random.random() < self.blur_prob:
            image = augment_blur(image)
        if random.random() < self.erase_prob:
            image = augment_erase(image)

        if self.num_kpts > 0: # keypoints don't support flip
            image_t, boxes = image, boxes
        else:
            image_t, boxes = _mirror_bboxes(image, boxes, self.flip_prob)

        image_t, r_ = preproc(image_t, input_dim)
        # boxes [xyxy] 2 [cx,cy,w,h]
        boxes = xyxy2cxcywh(boxes)
        boxes *= r_
        if self.num_kpts > 0: # kpts rescale
            kpts *= r_

        clip_b = np.minimum(boxes[:, 2], boxes[:, 3]) > 1
        boxes_t = boxes[clip_b]
        labels_t = labels[clip_b]
        if self.num_kpts > 0:
            kpts_t = kpts[clip_b]
            kpts_conf_t = kpts_conf[clip_b]

        if len(boxes_t) == 0:
            image_t, r_o = preproc(image_o, input_dim)
            boxes_t = boxes_o
            boxes_t *= r_o
            labels_t = labels_o
            if self.num_kpts > 0:
                kpts_t = kpts_o
                kpts_t *= r_o
                kpts_conf_t = kpts_conf_o

        labels_t = np.expand_dims(labels_t, 1)
        if self.num_kpts > 0:
            targets_t = np.hstack((labels_t, boxes_t, kpts_t, kpts_conf_t))
        else:
            targets_t = np.hstack((labels_t, boxes_t))
        padded_labels = np.zeros((self.max_labels, target_size))
        padded_labels[range(len(targets_t))[: self.max_labels]] = targets_t[
            : self.max_labels
        ]
        padded_labels = np.ascontiguousarray(padded_labels, dtype=np.float32)
        return image_t, padded_labels


class ValTransform:
    """
    Defines the transformations that should be applied to test PIL image
    for input into the network

    dimension -> tensorize -> color adj

    Arguments:
        resize (int): input dimension to SSD
        rgb_means ((int,int,int)): average RGB of the dataset
            (104,117,123)
        swap ((int,int,int)): final order of channels

    Returns:
        transform (transform) : callable transform to be applied to test/val
        data
    """

    def __init__(self, swap=(2, 0, 1)):
        self.swap = swap

    # assume input is cv2 img for now
    def __call__(self, img, res, input_dim):
        img, _ = preproc(img, input_dim, self.swap)
        return img, np.zeros((1, 5))
