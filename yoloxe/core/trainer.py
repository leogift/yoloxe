#!/usr/bin/env python3
# Copyright (c) Hanqtech, Inc. and its affiliates.

import datetime
import os
import time
from loguru import logger

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

from yoloxe.data import DataPrefetcher
from yoloxe.exp import Exp
from yoloxe.utils import (
    MeterBuffer,
    ModelEMA,
    adjust_status,
    all_reduce_norm,
    get_model_info,
    get_rank,
    get_world_size,
    gpu_mem_usage,
    is_parallel,
    load_ckpt,
    mem_usage,
    occupy_mem,
    save_checkpoint,
    setup_logger,
    synchronize,
)


class Trainer:
    def __init__(self, exp: Exp, args):
        # init function only defines some basic attr, other attrs like model, optimizer are built in
        # before_train methods.
        self.exp = exp
        self.args = args

        # training related attr
        self.max_epoch = exp.max_epoch
        self.warmup_epochs = exp.warmup_epochs
        self.amp_training = args.fp16
        self.scaler = torch.amp.GradScaler("cuda", enabled=args.fp16)
        self.is_distributed = get_world_size() > 1
        self.rank = get_rank()
        self.device = "cuda:{}".format(self.rank)
        self.use_model_ema = exp.ema
        self.save_history_ckpt = exp.save_history_ckpt

        # data/dataloader related attr
        self.data_type = torch.float16 if args.fp16 else torch.float32
        self.input_size = exp.input_size
        self.best_metric_mean = 0

        # metric record
        self.meter = MeterBuffer(window_size=exp.print_iter_interval)
        self.file_name = os.path.join(exp.output_dir, exp.exp_name)

        if self.rank == 0:
            os.makedirs(self.file_name, exist_ok=True)

        setup_logger(
            self.file_name,
            distributed_rank=self.rank,
            filename="train_log.txt",
            mode="a",
        )

    def train(self):
        self.before_train()
        try:
            self.trainning()
        except Exception as e:
            logger.error("Exception in training: ", e)
            raise
        finally:
            self.after_train()

    def trainning(self):
        for self.epoch in range(self.start_epoch, self.max_epoch):
            self.before_epoch()
            self.trainning_epoch()
            self.after_epoch()

    def trainning_epoch(self):
        for self.iter in range(self.max_iter):
            self.before_iter()
            self.trainning_iter()
            self.after_iter()

    def trainning_iter(self):
        data_time = 0
        iter_start_time = time.time()
        self.optimizer.zero_grad()
        for _ in range(self.exp.grad_accum):
            data_start_time = time.time()

            inps, targets = self.prefetcher.next()

            inps = inps.type(self.data_type)
            targets = targets.type(self.data_type)
            targets.requires_grad = False
            inps, targets = self.exp.preprocess(inps, targets, self.input_size)
            
            data_end_time = time.time()
            data_time += (data_end_time - data_start_time)

            with torch.amp.autocast("cuda", enabled=self.amp_training):
                outputs = self.model(inps, targets)
                loss = outputs["total_loss"]
            
            self.scaler.scale(loss).backward()

        self.scaler.step(self.optimizer)
        self.scaler.update()

        if self.use_model_ema:
            self.ema_model.update(self.model)

        lr = self.lr_scheduler.update_lr(self.progress_in_iter + 1)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

        iter_end_time = time.time()
        self.meter.update(
            iter_time=iter_end_time - iter_start_time,
            data_time=data_time,
            lr=lr,
            **outputs,
        )

    def before_train(self):
        logger.info("args: {}".format(self.args))
        logger.info("exp value:\n{}".format(self.exp))

        if self.args.batch_size % self.exp.grad_accum != 0:
            self.exp.grad_accum = 1
            logger.warning("grad_accum should divide batch_size, set to default 1")

        # model related init
        torch.cuda.set_device(self.rank)
        origin_model = self.exp.get_model()
        logger.info(
            "Model Summary: {}".format(get_model_info(origin_model,
                                                    self.exp.test_size
                                                ))
        )
        origin_model.to(self.device)

        # solver related init
        self.optimizer = self.exp.get_optimizer(self.args.batch_size)

        # value of epoch will be set in `resume_train`
        model = self.resume_train(origin_model)

        # data related init
        self.train_loader = self.exp.get_train_loader(
            batch_size=self.args.batch_size,
            is_distributed=self.is_distributed,
            cache=self.args.cache
        )
        logger.info("init prefetcher, this might take one minute or less...")
        self.prefetcher = DataPrefetcher(self.train_loader)
        # max_iter means iters per epoch
        self.max_iter = len(self.train_loader) // self.exp.grad_accum

        self.lr_scheduler = self.exp.get_lr_scheduler(
            self.exp.basic_lr_per_img * self.args.batch_size, self.max_iter
        )
        if self.args.occupy:
            occupy_mem(self.rank)

        if self.is_distributed:
            model = DDP(model, device_ids=[self.rank], broadcast_buffers=False, find_unused_parameters=True)

        if self.use_model_ema:
            self.ema_model = ModelEMA(model, 0.9998)
            self.ema_model.updates = self.max_iter * self.start_epoch

        self.model = model

        if self.rank == 0:
            self.evaluator = self.exp.get_evaluator(
                batch_size=self.args.batch_size,
                is_distributed=self.is_distributed
            )

        # Tensorboard loggers
        if self.rank == 0:
            self.tblogger = SummaryWriter(os.path.join(self.file_name, "tensorboard"))

        logger.info("Training start...")
        logger.info("\n{}".format(model))

    def after_train(self):
        logger.info(
            "Training of experiment is done. The best metric is {:.2f}".format(self.best_metric_mean * 100)
        )

    def before_epoch(self):
        from yoloxe.utils import freeze_module, unfreeze_module
        # trainning mode
        self.model.train()

        logger.info("--->Start train epoch{}".format(self.epoch + 1))

        if self.epoch == 0:
            self.save_ckpt(ckpt_name="first")

        if self.is_distributed:
            compat_model = self.model.module
        else:
            compat_model = self.model

        if self.epoch + 1 > self.max_epoch - self.exp.no_aug_epochs or self.exp.no_aug:
            if self.exp.only_train_head:
                logger.info("--->Freeze all!")
                freeze_module(compat_model.backbone)
                freeze_module(compat_model.neck)
            else:
                logger.info("--->Unfreeze all!")
                unfreeze_module(compat_model.backbone)
                unfreeze_module(compat_model.neck)

            logger.info("--->No aug now!")
            self.train_loader.close_aug()
            self.exp.eval_epoch_interval = 1

        elif self.epoch + 1 <= self.warmup_epochs//2:
            logger.info("--->Freeze all!")
            freeze_module(compat_model.backbone)
            freeze_module(compat_model.neck)

            logger.info("--->No aug now!")
            self.train_loader.close_aug()

        else:
            if self.exp.only_train_head:
                logger.info("--->Freeze all!")
                freeze_module(compat_model.backbone)
                freeze_module(compat_model.neck)
            else:
                logger.info("--->Unfreeze all!")
                unfreeze_module(compat_model.backbone)
                unfreeze_module(compat_model.neck)

            logger.info("--->Use aug now!")
            self.train_loader.open_aug()

    def after_epoch(self):
        self.save_ckpt(ckpt_name="latest")
        if (self.epoch + 1) % self.exp.eval_epoch_interval == 0:
            all_reduce_norm(self.model)
            if self.rank == 0:
                self.evaluate_and_save_model()

        synchronize()

    def before_iter(self):
        pass

    def after_iter(self):
        """
        `after_iter` contains two parts of logic:
            * log information
            * reset setting of resize
        """
        # log needed information
        if (self.iter + 1) % self.exp.print_iter_interval == 0:
            # TODO check ETA logic
            left_iters = self.max_iter * self.max_epoch - (self.progress_in_iter + 1)
            eta_seconds = self.meter["iter_time"].global_avg * left_iters
            eta_str = "ETA: {}".format(datetime.timedelta(seconds=int(eta_seconds)))

            progress_str = "epoch: {}/{}, iter: {}/{}".format(
                self.epoch + 1, self.max_epoch, self.iter + 1, self.max_iter
            )
            loss_meter = self.meter.get_filtered_meter("loss")
            loss_str = ", ".join(
                ["{}: {:.2f}".format(k, v.latest) for k, v in loss_meter.items()]
            )

            time_meter = self.meter.get_filtered_meter("time")
            time_str = ", ".join(
                ["{}: {:.2f}s".format(k, v.avg) for k, v in time_meter.items()]
            )

            mem_str = "gpu mem: {:.0f}Mb, mem: {:.1f}Gb".format(gpu_mem_usage(), mem_usage())

            logger.info(
                "{}, {}, {}, {}, lr: {:.3e}".format(
                    progress_str,
                    mem_str,
                    time_str,
                    loss_str,
                    self.meter["lr"].latest,
                )
                + (", size: {:d}, {}".format(self.input_size[0], eta_str))
            )

            if self.rank == 0:
                self.tblogger.add_scalar(
                    "train/lr", self.meter["lr"].latest, self.progress_in_iter)
                for k, v in loss_meter.items():
                    self.tblogger.add_scalar(
                        f"train/{k}", v.latest, self.progress_in_iter)

            self.meter.clear_meters()

        # random resizing
        if (self.progress_in_iter + 1) % 10 == 0:
            self.input_size = self.exp.random_resize(
                self.rank, self.is_distributed
            )

    @property
    def progress_in_iter(self):
        return self.epoch * self.max_iter + self.iter

    def resume_train(self, model):
        if self.args.resume:
            logger.info("resume training")
            if self.args.ckpt is None:
                ckpt_file = os.path.join(self.file_name, "latest" + "_ckpt.pth")
            else:
                ckpt_file = self.args.ckpt

            ckpt = torch.load(ckpt_file, map_location=self.device, weights_only=False)
            # resume the model/optimizer state dict
            model.load_state_dict(ckpt["model"])
            self.optimizer.load_state_dict(ckpt["optimizer"])
            self.best_metric_mean = ckpt.pop("best_metric_mean", 0)

            # resume the training states variables
            start_epoch = (
                self.args.start_epoch - 1
                if self.args.start_epoch is not None
                else ckpt["start_epoch"]
            )
            self.start_epoch = start_epoch
            logger.info(
                "loaded checkpoint '{}' (epoch {})".format(
                    self.args.resume, self.start_epoch
                )
            )  # noqa
        else:
            if self.args.ckpt is not None:
                logger.info("loading checkpoint for fine tuning")
                ckpt_file = self.args.ckpt
                ckpt = torch.load(ckpt_file, map_location=self.device)["model"]
                model = load_ckpt(model, ckpt)
            self.start_epoch = 0

        return model

    def evaluate_and_save_model(self):
        if self.use_model_ema:
            evalmodel = self.ema_model.ema
        else:
            evalmodel = self.model
            if is_parallel(evalmodel):
                evalmodel = evalmodel.module

        with adjust_status(evalmodel, training=False):
            eval_results = self.exp.eval(
                    evalmodel, self.evaluator
                )
        
        if self.rank == 0:
            eval_info = eval_results["eval_info"]
            eval_metrics = eval_results["eval_metrics"]

            inverse_sum = 0
            for k in eval_metrics.keys():
               self.tblogger.add_scalar(f"val/{k}", eval_metrics[k], self.epoch + 1)
               inverse_sum += 1 / max(eval_metrics[k], 1e-8)
            logger.info("\n" + eval_info)

            metric_mean = len(eval_metrics.keys()) / max(inverse_sum, 1e-8)
            update_best_ckpt = metric_mean >= self.best_metric_mean
            self.best_metric_mean = max(self.best_metric_mean, metric_mean)

            logger.info("Current metric score is  {}".format(metric_mean))
            logger.info("Best metric score is  {}".format(self.best_metric_mean))

            self.save_ckpt("last_epoch", update_best_ckpt)
            if self.save_history_ckpt:
                self.save_ckpt(f"epoch_{self.epoch + 1}")

    def save_ckpt(self, ckpt_name, update_best_ckpt=False):
        if self.rank == 0:
            save_model = self.ema_model.ema if self.use_model_ema else self.model
            logger.info("Save weights to {}".format(self.file_name))
            ckpt_state = {
                "start_epoch": self.epoch + 1,
                "model": save_model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "best_metric_mean": self.best_metric_mean,
            }
            save_checkpoint(
                ckpt_state,
                update_best_ckpt,
                self.file_name,
                ckpt_name,
            )
