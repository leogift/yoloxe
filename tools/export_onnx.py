#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import os
from loguru import logger

import torch
from torch import nn

from yoloxe.exp import get_exp
from yoloxe.utils import optimize_model

def make_parser():
    parser = argparse.ArgumentParser("YOLOXE onnx deploy")
    parser.add_argument(
        "-o", "--output-name", type=str, default="yoloxe.onnx", help="output name of models"
    )
    parser.add_argument(
        "--input", default="images", type=str, help="input node name of onnx model"
    )
    parser.add_argument(
        "--output", default="output", type=str, help="output node name of onnx model"
    )
    parser.add_argument(
        "-s", "--opset", default=16, type=int, help="onnx opset version"
    )
    parser.add_argument("-b", "--batch-size", type=int, default=1, help="batch size")
    parser.add_argument("--no-onnxsim", action="store_true", help="use onnxsim or not")
    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="experiment description file",
    )
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt path")
    parser.add_argument(
        "--decode_in_inference",
        action="store_true",
        help="decode in inference or not"
    )

    return parser


@logger.catch
def main():
    args = make_parser().parse_args()
    logger.info("args value: {}".format(args))
    exp = get_exp(args.exp_file)

    model = exp.get_model()
    model = model.cpu()
    ckpt_file = args.ckpt

    # load the model state dict
    ckpt = torch.load(ckpt_file, map_location="cpu")

    if "model" in ckpt:
        ckpt = ckpt["model"]

    model.load_state_dict(ckpt, strict=True)
    model = optimize_model(model)
    model.head.decode_in_inference = args.decode_in_inference

    model.eval()
    
    args.output = args.output.split(",")

    logger.info("loading checkpoint done.")
    dummy_input = torch.randn(args.batch_size, 3, exp.test_size[0], exp.test_size[1])

    torch.onnx.export(
        model,
        dummy_input,
        args.output_name,
        input_names=[args.input],
        output_names=args.output,
        opset_version=args.opset,
    )
    logger.info("generated onnx model named {}".format(args.output_name))

    if not args.no_onnxsim:
        import onnx
        from onnxsim import simplify

        # use onnx-simplifier to reduce reduent model.
        onnx_model = onnx.load(args.output_name)
        model_simp, check = simplify(onnx_model)
        assert check, "Simplified ONNX model could not be validated"
        onnx.save(model_simp, args.output_name)
        logger.info("generated simplified onnx model named {}".format(args.output_name))

if __name__ == "__main__":
    main()
