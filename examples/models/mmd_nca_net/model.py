# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging

import torch
from torch.autograd import Variable

from ..model_base import EagerModelBase
from .mmd_nca_net import MMD_NCA_Net

class MmdNcaNetModel(EagerModelBase):
    def __init__(self):
        pass

    def get_eager_model(self) -> torch.nn.Module:
        logging.info("loading mmd_nca_net model")
        mmd_nca_model = MMD_NCA_Net(sequence_size=30)
        logging.info("loaded mmd_nca_net model")
        return mmd_nca_model

    def get_example_inputs(self) :
        input = Variable(torch.ones(30, 13, 3)).float().squeeze()\
                .view(-1, 30,39).permute(1,0,2)
        return (input,)
