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
        mmd_nca_model = MMD_NCA_Net(frames_num=30, 
                                    joints_num=11, 
                                    dim_num=2)
        logging.info("loaded mmd_nca_net model")
        '''
        mmd_nca_model.load_weights(
            '/workspace/workspace/executorch/examples/models/mmd_nca_net/weights/sequential_250_128_meta_quest_30_13_mobile_refactor_63499.pth')
        '''
        '''
        mmd_nca_model.load_weights(
            '/workspace/workspace/executorch/examples/models/mmd_nca_net/weights/sequential_250_128_meta_quest_2d_proj_30_14_synced_12999.pth')
        '''
        mmd_nca_model.load_weights(
            '/workspace/workspace/executorch/examples/models/mmd_nca_net/weights/sequential_250_128_mocopi_2d_proj_30_11_synced_1249.pth')
        logging.info("loaded mmd_nca_net weights")
        return mmd_nca_model

    def get_example_inputs(self) :
        '''
        input = Variable(torch.ones(30, 13, 3)).float().squeeze()\
                .view(-1, 30,39).permute(1,0,2)
        '''
        '''
        input = Variable(torch.ones(30, 14, 2)).float().squeeze()\
                .view(-1, 30,28).permute(1,0,2)
        '''
        input = Variable(torch.ones(30, 11, 2)).float().squeeze()\
                .view(-1, 30,22).permute(1,0,2)
        return (input,)
