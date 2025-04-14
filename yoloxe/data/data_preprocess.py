#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Hanqtech, Inc. and its affiliates.

import cv2
import numpy as np

# static resize
def static_resize(img, image_size):
    padded_img = np.ones((image_size[0], image_size[1], 3), dtype=np.uint8) * 114
    ratio = min(image_size[0] / img.shape[0], image_size[1] / img.shape[1])
    # 随机缩放模式
    interpolation_types = [
            cv2.INTER_NEAREST,
            cv2.INTER_LINEAR,
        ]
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * ratio), int(img.shape[0] * ratio)),
        interpolation=np.random.choice(interpolation_types),
    ).astype(np.uint8)

    padded_img[: int(img.shape[0] * ratio), : int(img.shape[1] * ratio)] = resized_img.astype(np.uint8)

    return padded_img, ratio


def data_preprocess(
        input_image, 
        input_annos,
        image_size=(416, 416)
    ):
    
    image = input_image.copy()
    annos = input_annos.copy()

    # 静态缩放
    image, ratio = static_resize(image, image_size)
    # annos: list[category, xmin, ymin, xmax, ymax]
    annos[:, 1:5] = annos[:, 1:5] * ratio
    
    return image, annos
