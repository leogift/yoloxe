#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import numpy as np

from .coco import COCODataset

class COCOKPTSDataset(COCODataset):
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
        num_kpts=0,
    ):
        assert num_kpts > 0, "num_kpts should be greater than 0."
        self.num_kpts = num_kpts

        super().__init__(
            data_dir=data_dir,
            json_file=json_file,
            name=name,
            img_size=img_size,
            preproc=preproc,
            cache=cache,
        )
        
    def __len__(self):
        return len(self.ids)

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
                ann["clean_kpts"] = np.array(ann["keypoints"]).reshape(-1, 3)
                objs.append(ann)

        len_objs = len(objs)

        target = np.zeros((len_objs, 5 + self.num_kpts*2 + self.num_kpts)) # x1,y1,x2,y2,cls, kpts_xy,kpts_conf

        for ix, obj in enumerate(objs):
            cls = self.class_ids.index(obj["category_id"])
            target[ix, 0:4] = obj["clean_bbox"]
            target[ix, 4] = cls
            for ikpt in range(self.num_kpts):
                target[ix, 5+ikpt*2] = obj["clean_kpts"][ikpt, 0]
                target[ix, 5+ikpt*2 + 1] = obj["clean_kpts"][ikpt, 1]
                target[ix, 5+self.num_kpts*2 + ikpt] = 1 if obj["clean_kpts"][ikpt, 2] > 0 else 0

        r = min(self.img_size[0] / height, self.img_size[1] / width)
        target[:, :4] *= r
        if self.num_kpts > 0:
            target[:, 5:5+self.num_kpts*2] *= r

        img_info = (height, width)

        file_name = (
            im_ann["file_name"]
            if "file_name" in im_ann
            else "{:012}".format(id_) + ".jpg"
        )

        return target, img_info, file_name
