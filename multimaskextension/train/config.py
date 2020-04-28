# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from detectron2.config import CfgNode as CN


# TODO(allie): Implement this!

def add_multimask_config(cfg):
    """
    Add config for Multimask head. (refer to densepose/config.py for an example)
    """
    _C = cfg

    _C.MODEL.DENSEPOSE_ON = True

    raise NotImplementedError
