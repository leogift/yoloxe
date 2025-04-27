#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import contextlib
import io
import itertools
import json
import tempfile
import time
from loguru import logger
from tabulate import tabulate
from tqdm import tqdm

import numpy as np

import torch

from yoloxe.data.datasets import COCO_CLASSES
from yoloxe.utils import (
    is_main_process,
    postprocess,
    xyxy2xywh
)


def per_class_AP_table(coco_eval, class_names=COCO_CLASSES, headers=["class", "AP"], colums=6):
    per_class_AP = {}
    precisions = coco_eval.eval["precision"]
    # dimension of precisions: [TxRxKxAxM]
    # precision has dims (iou, recall, cls, area range, max dets)
    assert len(class_names) == precisions.shape[2]

    for idx, name in enumerate(class_names):
        # area range index 0: all area ranges
        # max dets index -1: typically 100 per image
        precision = precisions[:, :, idx, 0, -1]
        precision = precision[precision > -1]
        ap = np.mean(precision) if precision.size else float("nan")
        per_class_AP[name] = float(ap * 100)

    num_cols = min(colums, len(per_class_AP) * len(headers))
    result_pair = [x for pair in per_class_AP.items() for x in pair]
    row_pair = itertools.zip_longest(*[result_pair[i::num_cols] for i in range(num_cols)])
    table_headers = headers * (num_cols // len(headers))
    table = tabulate(
        row_pair, tablefmt="pipe", floatfmt=".3f", headers=table_headers, numalign="left",
    )
    return table


class COCOEvaluator:
    """
    COCO AP Evaluation class.  All the data in the val2017 dataset are processed
    and evaluated by COCO API.
    """

    def __init__(
        self,
        dataloader,
        img_size: int,
        confthre: float,
        nmsthre: float,
        num_classes: int,
        per_class_AP: bool = True,
        num_kpts: int = 0,
    ):
        """
        Args:
            dataloader (Dataloader): evaluate dataloader.
            img_size: image size after preprocess. images are resized
                to squares whose shape is (img_size, img_size).
            confthre: confidence threshold ranging from 0 to 1, which
                is defined in the config file.
            nmsthre: IoU threshold of non-max supression ranging from 0 to 1.
            per_class_AP: Show per class AP during evalution or not. Default to True.
        """
        self.dataloader = dataloader
        self.img_size = img_size
        self.confthre = confthre
        self.nmsthre = nmsthre
        self.num_classes = num_classes
        self.per_class_AP = per_class_AP
        self.num_kpts = num_kpts

    def evaluate(
        self, 
        model, 
        decoder=None,
    ):
        """
        COCO average precision (AP) Evaluation. Iterate inference on the test dataset
        and the results are evaluated by COCO API.

        NOTE: This function will change training mode to False, please save states if needed.

        Args:
            model : model to evaluate.

        Returns:
            ap50_95 (float) : COCO AP of IoU=50:95
            ap50 (float) : COCO AP of IoU=50
            summary (sr): summary info of evaluation.
        """
        if not is_main_process():
            return None
	
        model = model.eval()
        ids = []
        outputs_list = []

        inference_time = 0
        nms_time = 0
        n_samples = max(len(self.dataloader) - 1, 1)

        print("Start evaluate ...")
        for cur_iter, (imgs, _, info_imgs, ids) in enumerate(
            tqdm(self.dataloader)
        ):
            with torch.no_grad():
                imgs = imgs.type(torch.cuda.FloatTensor)

                # skip the last iters since batchsize might be not enough for batch inference
                is_time_record = cur_iter < len(self.dataloader) - 1
                if is_time_record:
                    start = time.time()

                outputs = model(imgs)
                if decoder is not None:
                    outputs = decoder(outputs, dtype=outputs.type())

                if is_time_record:
                    infer_end = time.time()
                    inference_time += infer_end - start

                if self.num_kpts > 0:
                    outputs = postprocess(
                        outputs, self.num_classes, self.confthre, self.nmsthre, self.num_kpts
                    )
                else:
                    outputs = postprocess(
                        outputs, self.num_classes, self.confthre, self.nmsthre
                    )
                
                if is_time_record:
                    nms_end = time.time()
                    nms_time += nms_end - infer_end

            outputs_list_elem = self.convert_to_coco_format(
                outputs, info_imgs, ids)
            outputs_list.extend(outputs_list_elem)

        a_infer_time = 1000 * inference_time / (n_samples * self.dataloader.batch_size)
        a_nms_time = 1000 * nms_time / (n_samples * self.dataloader.batch_size)

        time_info = ", ".join(
            [
                "Average {} time: {:.2f} ms".format(k, v)
                for k, v in zip(
                    ["forward", "NMS", "inference"],
                    [a_infer_time, a_nms_time, (a_infer_time + a_nms_time)],
                )
            ]
        )
        
        dash = "--------"*10+"\n"

        eval_results = {
            "eval_info": time_info + "\n" + dash,
            "eval_metrics": {}
        }

        eval_results = self.evaluate_prediction(outputs_list, eval_results)

        return eval_results


    def evaluate_prediction(self, data_dict, eval_results):
        if not is_main_process():
            return 0, 0, None

        logger.info("Evaluate in main process...")

        annType = ["segm", "bbox", "keypoints"]

        # Evaluate the Dt (detection) json comparing with the ground truth
        if len(data_dict) > 0:
            cocoGt = self.dataloader.dataset.coco
            _, tmp = tempfile.mkstemp()
            json.dump(data_dict, open(tmp, "w"))
            cocoDt = cocoGt.loadRes(tmp)
            try:
                from yoloxe.layers import COCOeval_opt as COCOeval
            except ImportError:
                from pycocotools.cocoeval import COCOeval

                logger.warning("Use standard COCOeval.")

            if self.num_kpts > 0:
                cocoEval = COCOeval(cocoGt, cocoDt, annType[2])
                cocoEval.params.kpt_oks_sigmas = np.array([0.89]*self.num_kpts) / 10
            else:
                cocoEval = COCOeval(cocoGt, cocoDt, annType[1])

            cocoEval.evaluate()
            cocoEval.accumulate()
            redirect_string = io.StringIO()
            with contextlib.redirect_stdout(redirect_string):
                cocoEval.summarize()
            eval_results["eval_info"] += redirect_string.getvalue()

            if self.per_class_AP:
                cat_ids = list(cocoGt.cats.keys())
                cat_names = [cocoGt.cats[catId]['name'] for catId in sorted(cat_ids)]
                AP_table = per_class_AP_table(cocoEval, class_names=cat_names)
                eval_results["eval_info"] += "per class AP:\n" + AP_table + "\n"

            eval_results["eval_metrics"]["ap50_95"] = cocoEval.stats[0]
            eval_results["eval_metrics"]["ap50"] = cocoEval.stats[1]
			
            if self.num_kpts > 0:
                eval_results["eval_metrics"]["ar50_95"] = cocoEval.stats[5]
            else:
                eval_results["eval_metrics"]["ar50_95"] = cocoEval.stats[8]

        return eval_results


    def convert_to_coco_format(self, outputs, info_imgs, ids):
        outputs_list = []
        for (output, img_h, img_w, img_id) in zip(
            outputs, info_imgs[0], info_imgs[1], ids
        ):
            if output is None:
                continue
            output = output.cpu()
            
            bboxes = output[:, 0:4]
            cls = output[:, 6]
            scores = output[:, 4] * output[:, 5]

            # preprocessing: resize
            scale = min(
                self.img_size[0] / float(img_h), self.img_size[1] / float(img_w)
            )
            bboxes /= scale
            bboxes = xyxy2xywh(bboxes)

            if self.num_kpts > 0:
                kpts = output[:, 7 : 7+self.num_kpts*2] # N x 2k
                kpts /= scale
                kpts_conf = output[:, 7+self.num_kpts*2 : 7+self.num_kpts*2+self.num_kpts] # N x k
                keypoints = torch.zeros([kpts.shape[0], self.num_kpts*3])
                for i in range(self.num_kpts):
                    keypoints[:, i*3] = kpts[:, i*2]
                    keypoints[:, i*3+1] = kpts[:, i*2+1]
                    keypoints[:, i*3+2] = kpts_conf[:, i]

            for ind in range(bboxes.shape[0]):
                label = self.dataloader.dataset.class_ids[int(cls[ind])]
                pred_data = {
                    "image_id": int(img_id),
                    "category_id": label,
                    "bbox": bboxes[ind].numpy().tolist(),
                    "score": scores[ind].numpy().item(),
                    "keypoints": [],
                    "segmentation": [],
                }  # COCO json format
                if self.num_kpts > 0:
                    pred_data["keypoints"] = keypoints[ind].numpy().tolist()
                outputs_list.append(pred_data)

        return outputs_list


