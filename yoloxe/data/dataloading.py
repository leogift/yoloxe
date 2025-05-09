#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import os
import random
import uuid

import numpy as np

import torch
from torch.utils.data.dataloader import DataLoader as torchDataLoader
from torch.utils.data.dataloader import default_collate

from .samplers import YoloBatchSampler

def get_yoloxe_datadir():
    """
    get dataset dir of YOLOXE. If environment variable named `YOLOXE_DATADIR` is set,
    this function will return value of the environment variable. Otherwise, use data
    """
    yoloxe_datadir = os.getenv("YOLOXE_DATADIR", None)
    if yoloxe_datadir is None:
        import yoloxe

        yoloxe_path = os.path.dirname(os.path.dirname(yoloxe.__file__))
        yoloxe_datadir = os.path.join(yoloxe_path, "datasets")
    return yoloxe_datadir


class DataLoader(torchDataLoader):
    """
    Lightnet dataloader that enables on the fly resizing of the images.
    See :class:`torch.utils.data.DataLoader` for more information on the arguments.
    Check more on the following website:
    https://gitlab.com/EAVISE/lightnet/-/blob/master/lightnet/data/_dataloading.py
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__initialized = False
        shuffle = False
        batch_sampler = None
        if len(args) > 5:
            shuffle = args[2]
            sampler = args[3]
            batch_sampler = args[4]
        elif len(args) > 4:
            shuffle = args[2]
            sampler = args[3]
            if "batch_sampler" in kwargs:
                batch_sampler = kwargs["batch_sampler"]
        elif len(args) > 3:
            shuffle = args[2]
            if "sampler" in kwargs:
                sampler = kwargs["sampler"]
            if "batch_sampler" in kwargs:
                batch_sampler = kwargs["batch_sampler"]
        else:
            if "shuffle" in kwargs:
                shuffle = kwargs["shuffle"]
            if "sampler" in kwargs:
                sampler = kwargs["sampler"]
            if "batch_sampler" in kwargs:
                batch_sampler = kwargs["batch_sampler"]

        # Use custom BatchSampler
        if batch_sampler is None:
            if sampler is None:

                sampler = torch.utils.data.sampler.RandomSampler(self.dataset)
            batch_sampler = YoloBatchSampler(
                sampler,
                self.batch_size,
                self.drop_last,
                input_dimension=self.dataset.input_dim,
            )
            # batch_sampler = IterationBasedBatchSampler(batch_sampler, num_iterations =

        self.batch_sampler = batch_sampler

        self.__initialized = True

    def open_aug(self):
        self.batch_sampler.aug = True

    def close_aug(self):
        self.batch_sampler.aug = False


def list_collate(batch):
    """
    Function that collates lists or tuples together into one list (of lists/tuples).
    Use this as the collate function in a Dataloader, if you want to have a list of
    items as an output, as opposed to tensors (eg. Brambox.boxes).
    """
    items = list(zip(*batch))

    for i in range(len(items)):
        if isinstance(items[i][0], (list, tuple)):
            items[i] = list(items[i])
        else:
            items[i] = default_collate(items[i])

    return items


def worker_init_reset_seed(worker_id):
    import time
    seed = (uuid.uuid4().int | int(time.time())) % 2**32
    random.seed(seed)
    torch.set_rng_state(torch.manual_seed(seed).get_state())
    np.random.seed(seed)
