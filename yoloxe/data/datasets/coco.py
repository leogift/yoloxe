#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import os
from loguru import logger

import cv2
import numpy as np
from pycocotools.coco import COCO

from ..dataloading import get_yoloxe_datadir
from .datasets_wrapper import CacheDataset, cache_read_img

import random
import copy


class COCODataset(CacheDataset):
    """
    COCO dataset class.
    """

    def __init__(
        self,
        data_dir=None,
        json_file="instances_train2017.json",
        name="train2017",
        img_size=(416, 416),
        preproc=None,
        cache=False,
    ):
        """
        COCO dataset initialization. Annotation data are read into memory by COCO API.
        Args:
            data_dir (str): dataset root directory
            json_file (str): COCO json file name
            name (str): COCO data name (e.g. 'train2017' or 'val2017')
            img_size (int): target image size after pre-processing
            preproc: data augmentation strategy
        """
        if data_dir is None:
            data_dir = os.path.join(get_yoloxe_datadir(), "COCO")
        else:
            data_dir = os.path.join(get_yoloxe_datadir(), data_dir)
        self.data_dir = data_dir
        self.json_file = json_file

        self.coco = COCO(os.path.join(self.data_dir, "annotations", self.json_file))
        self.ids = self.coco.getImgIds()
        self.class_ids = sorted(self.coco.getCatIds())
        cats = self.coco.loadCats(self.coco.getCatIds())
        self._classes = tuple([c["name"] for c in cats])
        self.name = name
        self.img_size = img_size
        self.preproc = preproc

        self.annotations = self._load_coco_annotations()

        path_filename = [os.path.join(name, anno[2]) for anno in self.annotations]
        super().__init__(
            img_size,
            num_imgs=len(self.ids),
            data_dir=data_dir,
            cache_dir_name=f"cache_{name}",
            path_filename=path_filename,
            cache=cache,
        )

    def __len__(self):
        return len(self.ids)

    def _load_coco_annotations(self):
        return [self.load_anno_from_ids(_ids) for _ids in self.ids]

    def load_anno_from_ids(self, id_):
        im_ann = self.coco.loadImgs(id_)[0]

        width = im_ann["width"]
        height = im_ann["height"]

        anno_ids = self.coco.getAnnIds(imgIds=[int(id_)])
        annotations = self.coco.loadAnns(anno_ids)
        objs = []
        for ann in annotations:
            x1 = ann["bbox"][0]
            y1 = ann["bbox"][1]
            x2 = x1 + ann["bbox"][2]
            y2 = y1 + ann["bbox"][3]

            if x2 >= x1 and y2 >= y1:
                ann["clean_bbox"] = [x1, y1, x2, y2]
                objs.append(ann)

        len_objs = len(objs)

        target = np.zeros((len_objs, 5)) # x1,y1,x2,y2, cls

        for ix, obj in enumerate(objs):
            cls = self.class_ids.index(obj["category_id"])
            target[ix, 0:4] = obj["clean_bbox"]
            target[ix, 4] = cls

        r = min(self.img_size[0] / height, self.img_size[1] / width)
        target[:, :4] *= r

        img_info = (height, width)

        file_name = (
            im_ann["file_name"]
            if "file_name" in im_ann
            else "{:012}".format(id_) + ".jpg"
        )

        return target, img_info, file_name

    def load_anno(self, index):
        return self.annotations[index][0]

    def load_resized_img(self, index, interpolation=None):
        img = self.load_image(index)
        r = min(self.img_size[0] / img.shape[0], self.img_size[1] / img.shape[1])

        if interpolation is None:
            inter_modes = [cv2.INTER_LINEAR, cv2.INTER_NEAREST]
            inter_idx = random.randint(0,len(inter_modes)-1)
            interpolation = inter_modes[inter_idx]

        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * r), int(img.shape[0] * r)),
            interpolation=interpolation,
        ).astype(np.uint8)
        return resized_img

    def load_image(self, index):
        file_name = self.annotations[index][2]

        img_file = os.path.join(self.data_dir, self.name, file_name)

        img = cv2.imread(img_file)

        assert img is not None, f"file named {img_file} not found"

        return img

    @cache_read_img(use_cache=True)
    def read_img(self, index):
        return self.load_resized_img(index)

    def pull_item(self, index):

        try:
            id_ = self.ids[index]

        except Exception as e:
            logger.error(str(e))
            logger.error("pull_item:"+str(index)+"/"+str(len(self.ids)))
            index = random.randint(0, len(self.ids)-1)
            id_ = self.ids[index]
            logger.error("new pull_item:"+str(index)+"/"+str(len(self.ids)))
    
        target, img_info, _ = self.annotations[index]
        
        img = self.read_img(index)
        return img, copy.deepcopy(target), img_info, np.array([id_])

    @CacheDataset.wrap_getitem
    def __getitem__(self, index):
        """
        One image / label pair for the given index is picked up and pre-processed.

        Args:
            index (int): data index

        Returns:
            img (numpy.ndarray): pre-processed image
            padded_labels (torch.Tensor): pre-processed label data.
                The shape is :math:`[max_labels, 5]`.
                each label consists of [class, xc, yc, w, h]:
                    class (float): class index.
                    xc, yc (float) : center of bbox whose values range from 0 to 1.
                    w, h (float) : size of bbox whose values range from 0 to 1.
            info_img : tuple of h, w.
                h, w (int): original shape of the image
            img_id (int): same as the input index. Used for evaluation.
        """
        img, target, img_info, img_id = self.pull_item(index)

        if self.preproc is not None:
            img, target = self.preproc(img, target, self.input_dim)

        return img, target, img_info, img_id
