#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Hanqtech Inc. All rights reserved.

import os
import random

import torch
import torch.distributed as dist
import torch.nn as nn

from .base_exp import BaseExp


class Exp(BaseExp):
    def __init__(self):
        super().__init__()

        # ---------------- model config ---------------- #
        # detect classes number of model
        self.num_classes = 0
        # detect keypoints number of model
        self.num_kpts = 0
        self.kpts_weight = None

        # model task, in ["dt2d", "kpts"]
        self.task = "dt2d"  # 2d object detection
        # self.task = "kpts"  # keypoints detection

        # activation name. For example, if using "relu", then "silu" will be replaced to "relu".
        self.act = "silu"

        # ---------------- dataloader config ---------------- #
        # set worker to 4 for shorter dataloader init time
        # If your training process cost many memory, reduce this value.
        self.data_num_workers = 4
        self.input_size = (416, 416)  # (height, width)
        # Actual multiscale ranges: [416 - 6 * 32, 416 + 6 * 32].
        # To disable multiscale training, set the value to 0.
        self.multiscale_range = 3
        # dir of dataset images, if data_dir is None, this project will use `datasets` dir
        self.data_dir = "COCO"
        # name of annotation file for training
        self.train_ann = "instances_train2017.json"
        self.train_name = "train2017"
        # name of annotation file for evaluation
        self.val_ann = "instances_val2017.json"
        self.val_name = "val2017"

        # --------------- transform config ----------------- #
        self.no_aug = False
        # prob of applying mosaic aug
        self.mosaic_prob = 0.5
        # prob of applying mixup aug
        self.mixup_prob = 0.2
        # prob of applying hsv aug
        self.hsv_prob = 0.8
        # prob of applying blur aug
        self.blur_prob = 0.2
        # prob of applying erase aug
        self.erase_prob = 0.5
        # prob of applying flip augmosaic
        self.flip_prob = 0.5
        # rotation angle range, for example, if set to 5, the true range is (-5, 5)
        self.degrees = 5.0
        # translate range, for example, if set to 0.1, the true range is (-0.2, 0.2)
        self.translate = 0.2
        # scale range
        self.scale = (0.5, 2.0)
        # shear angle range, for example, if set to 2, the true range is (-2, 2)
        self.shear = 2.0

        # --------------  training config --------------------- #
        # epoch number used for warmup
        self.warmup_epochs = 10
        # max training epoch
        self.max_epoch = 240
        # minimum learning rate during warmup
        self.warmup_lr = 1e-6
        self.min_lr_ratio = 0.001
        # learning rate for one image. During training, lr will multiply batchsize.
        self.basic_lr_per_img = 1e-3 / 64.0 # SGD: 0.01 / 64.0, Adam: 0.001 / 64.0
        # name of LRScheduler
        self.scheduler = "warmcos"
        # last #epoch to close augmention like mosaic
        self.no_aug_epochs = 10
        # number of grad accumulation steps
        self.grad_accum = 4
        # apply EMA during training
        self.ema = True
        # Optimizer name
        self.opt_name = "AdamW"
        # weight decay of optimizer
        self.weight_decay = 5e-4
        # momentum of optimizer
        self.momentum = 0.9
        # log period in iter, for example,
        # if set to 1, user could see log every iteration.
        self.print_iter_interval = 50
        # eval period in epoch, for example,
        # if set to 1, model will be evaluate after every epoch.
        self.eval_epoch_interval = 5
        # save history checkpoint or not.
        # If set to False, yolo will only save latest and best ckpt.
        self.save_history_ckpt = False
        # Wether only train the head
        self.only_train_head = False
        # name of experiment
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        # -----------------  testing config ------------------ #
        # output image size during evaluation/test
        self.test_size = (416, 416)
        # confidence threshold during evaluation/test,
        # boxes whose scores are less than test_conf will be filtered
        self.test_conf = 0.1
        # nms threshold
        self.nmsthre = 0.5

    def get_model(self):
        pass

    def get_dataset(self, cache: bool = False):
        """
        Get dataset according to cache and cache_type parameters.
        Args:
            cache (bool): Whether to cache imgs to ram or disk.
            cache_type (str, optional): Defaults to "ram".
                "ram" : Caching imgs to ram for fast training.
                "disk": Caching imgs to disk for fast training.
        """
        from yoloxe.data import COCODataset, COCOKPTSDataset, \
            TrainTransform
        
        if self.task == "kpts":
            assert self.num_kpts > 0, "num_kpts should be greater than 0."
            return COCOKPTSDataset(
                data_dir=self.data_dir,
                json_file=self.train_ann,
                name=self.train_name,
                img_size=self.input_size,
                preproc=TrainTransform(
                    max_labels=50,
                    flip_prob=self.flip_prob,
                    hsv_prob=self.hsv_prob,
                    blur_prob=self.blur_prob,
                    erase_prob=self.erase_prob,
                    num_kpts=self.num_kpts
                ),
                cache=cache,
                num_kpts=self.num_kpts
            )
        
        else:
            return COCODataset(
                data_dir=self.data_dir,
                json_file=self.train_ann,
                name=self.train_name,
                img_size=self.input_size,
                preproc=TrainTransform(
                    max_labels=50,
                    flip_prob=self.flip_prob,
                    hsv_prob=self.hsv_prob,
                    blur_prob=self.blur_prob,
                    erase_prob=self.erase_prob
                ),
                cache=cache,
            )

    def get_train_loader(
        self, batch_size, is_distributed, cache=False
    ):
        from yoloxe.data import (
            TrainTransform,
            YoloBatchSampler,
            DataLoader,
            InfiniteSampler,
            AugmentDetection,
            worker_init_reset_seed,
        )
        from yoloxe.utils import (
            wait_for_the_master,
            get_rank,
        )

        local_rank = get_rank()

        with wait_for_the_master(local_rank):
            dataset = self.get_dataset(cache=cache)

        self.dataset = AugmentDetection(
            dataset,
            enable_aug=not self.no_aug,
            img_size=self.input_size,
            preproc=TrainTransform(
                max_labels=200,
                flip_prob=self.flip_prob if not self.no_aug else 0,
                hsv_prob=self.hsv_prob if not self.no_aug else 0,
                blur_prob=self.blur_prob if not self.no_aug else 0,
                erase_prob=self.erase_prob if not self.no_aug else 0,
                num_kpts=self.num_kpts,
            ),
            degrees=self.degrees,
            translate=self.translate,
            shear=self.shear,
            mosaic_prob=self.mosaic_prob,
            scale=self.scale,
            mixup_prob=self.mixup_prob,
            num_kpts=self.num_kpts
        )

        batch_size = batch_size // self.grad_accum
        if is_distributed:
            batch_size = batch_size // dist.get_world_size()

        sampler = InfiniteSampler(len(self.dataset), seed=self.seed if self.seed else 0)

        batch_sampler = YoloBatchSampler(
            sampler=sampler,
            batch_size=batch_size,
            drop_last=False,
            aug=not self.no_aug,
        )

        dataloader_kwargs = {
            "num_workers": self.data_num_workers,
            "pin_memory": True,
            "batch_sampler": batch_sampler,
            "worker_init_fn": worker_init_reset_seed,
        }
        train_loader = DataLoader(self.dataset, **dataloader_kwargs)

        return train_loader

    def random_resize(self, rank, is_distributed):
        tensor = torch.LongTensor(2).cuda()

        if rank == 0:
            size_factor = self.input_size[1] * 1.0 / self.input_size[0]
            min_size = int(self.input_size[0] / 32) - self.multiscale_range
            max_size = int(self.input_size[0] / 32) + self.multiscale_range
            size = random.randint(min_size, max_size)
            size = (int(32 * size), 32 * int(size * size_factor))
            tensor[0] = size[0]
            tensor[1] = size[1]

        if is_distributed:
            dist.barrier()
            dist.broadcast(tensor, 0)

        input_size = (tensor[0].item(), tensor[1].item())
        return input_size

    def preprocess(self, inputs, targets, tsize):
        scale_y = tsize[0] / self.input_size[0]
        scale_x = tsize[1] / self.input_size[1]
        if scale_x != 1 or scale_y != 1:

            inter_modes = ['nearest', 'bilinear']
            align_modes = [None, False]
            inter_idx = random.randint(0,len(inter_modes)-1)
    
            inputs = nn.functional.interpolate(
                inputs, size=tsize, mode=inter_modes[inter_idx], align_corners=align_modes[inter_idx]
            )
            targets[..., 1:5:2] = targets[..., 1:5:2] * scale_x
            targets[..., 2:5:2] = targets[..., 2:5:2] * scale_y
            if self.task == "kpts": # rescale keypoints
                for ikpt in range(self.num_kpts):
                    targets[..., 5+ikpt*2] = targets[..., 5+ikpt*2] * scale_x
                    targets[..., 5+ikpt*2 + 1] = targets[..., 5+ikpt*2+1] * scale_y
        
        return inputs, targets

    def get_optimizer(self, batch_size):
        if "optimizer" not in self.__dict__:
            if self.warmup_epochs > 0:
                lr = self.warmup_lr
            else:
                lr = self.basic_lr_per_img * batch_size

            g = [], [], []  # optimizer parameter groups
            norm = tuple(v for k, v in nn.__dict__.items() if "Norm" in k)  # normalization layers, i.e. BatchNorm2d()
            for v in self.model.modules():
                for p_name, p in v.named_parameters(recurse=0):
                    if "bias" in p_name:
                        g[2].append(p)
                    elif p_name == "weight" and isinstance(v, norm):  # weight (no decay)
                        g[1].append(p)
                    else:
                        g[0].append(p)  # weight (with decay)

            # add uncetainty parameters
            g[0].append(self.model.head.cls_uncertainty.weight)
            g[0].append(self.model.head.obj_uncertainty.weight)
            g[0].append(self.model.head.bbox_uncertainty.weight)
            if self.task == "kpts":
                g[0].append(self.model.head.kpts_uncertainty.weight)
                g[0].append(self.model.head.kpts_conf_uncertainty.weight)

            if self.opt_name == "Adam":
                optimizer = torch.optim.Adam(g[2], lr=lr, betas=(self.momentum, 0.999))  # adjust beta1 to momentum
            elif self.opt_name == "AdamW":
                optimizer = torch.optim.AdamW(g[2], lr=lr, betas=(self.momentum, 0.999), amsgrad=True)
            elif self.opt_name == "SGD":
                optimizer = torch.optim.SGD(g[2], lr=lr, momentum=self.momentum, nesterov=True)
            else:
                raise NotImplementedError(f"Optimizer {self.opt_name} not implemented.")

            optimizer.add_param_group({"params": g[0], "weight_decay": self.weight_decay})  # add g0 with weight_decay
            optimizer.add_param_group({"params": g[1], "weight_decay": 0.0})  # add g1 (BatchNorm2d weights)

            self.optimizer = optimizer

        return self.optimizer

    def get_lr_scheduler(self, lr, iters_per_epoch):
        from yoloxe.utils import LRScheduler

        scheduler = LRScheduler(
            self.scheduler,
            lr,
            iters_per_epoch,
            self.max_epoch,
            warmup_epochs=self.warmup_epochs,
            warmup_lr_start=self.warmup_lr,
            no_aug_epochs=self.no_aug_epochs,
            min_lr_ratio=self.min_lr_ratio,
        )
        return scheduler

    def _get_val_loader(self, batch_size, is_distributed):
        from yoloxe.data import COCODataset, COCOKPTSDataset, ValTransform

        if self.task == "kpts":
            valdataset = COCOKPTSDataset(
                data_dir=self.data_dir,
                json_file=self.val_ann,
                name=self.val_name,
                img_size=self.test_size,
                preproc=ValTransform(),
                num_kpts=self.num_kpts
            )
        else:
            valdataset = COCODataset(
                data_dir=self.data_dir,
                json_file=self.val_ann,
                name=self.val_name,
                img_size=self.test_size,
                preproc=ValTransform(),
            )
        
        batch_size = batch_size // self.grad_accum
        if is_distributed:
            batch_size = batch_size // dist.get_world_size()
        sampler = torch.utils.data.SequentialSampler(valdataset)

        dataloader_kwargs = {
            "num_workers": self.data_num_workers,
            "pin_memory": True,
            "sampler": sampler,
            "batch_size": batch_size,
            "drop_last": True,
        }
        val_loader = torch.utils.data.DataLoader(valdataset, **dataloader_kwargs)

        return val_loader

    def get_evaluator(self, batch_size, is_distributed):
        from yoloxe.evaluators import COCOEvaluator

        val_loader = self._get_val_loader(batch_size, is_distributed)

        return COCOEvaluator(
            dataloader=val_loader,
            img_size=self.test_size,
            confthre=self.test_conf,
            nmsthre=self.nmsthre,
            num_classes=self.num_classes,
            num_kpts=self.num_kpts,
        )

    def eval(self, model, evaluator, half=False):
        return evaluator.evaluate(model, half)
