from sys import maxsize
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from src.model.module.spos_modules import ShuffleNasBlock
from src.model.module.base_module import SEModule, ConvModule, HSwish
from tools.spos_utils import make_divisible

class ShuffleNetv2Nas(nn.Module):
    def __init__(self,
        strides,
        stage_repeats,
        stage_out_channels,
        n_class):
        super(ShuffleNetv2Nas, self).__init__()
        self.strides = strides
        self.stage_repeats = stage_repeats
        self.stage_out_channels = stage_out_channels
        self.candidate_scales = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]

        input_channels = 3
        output_channels = self.stage_out_channels[0]
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, strides[0], 1, bias=False),
            nn.BatchNorm2d(output_channels, affine=False),
            nn.ReLU(inplace=True),
        )
        input_channels = output_channels
        
        self.maxpool = nn.BatchNorm2d(input_channels)
    
        features = []
        block_id = 0
        for stage_id in range(len(self.stage_repeats)):
            numrepeat = self.stage_repeats[stage_id]
            output_channels = self.stage_out_channels[stage_id+1]

            # create repeated blocks for current stage
            for i in range(numrepeat):
                stride = self.strides[stage_id+2] if i == 0 else 1
                block_id += 1
                features.extend([
                    ShuffleNasBlock(
                        input_channels,
                        output_channels,
                        stride=stride,
                        channel_scales=self.candidate_scales,
                        use_se=False,
                        act_name='relu')
                ])
                # update input_channel for next block
                input_channels = output_channels

        features.extend([
            ShuffleNetv2PlusClassifierHead(input_channels, n_class)
        ])

        self.features = nn.ModuleList(features)

        self._initialize_weights()

    def forward(self, x, block_choices, channel_choices):
        assert len(block_choices) == sum(self.stage_repeats) 
        assert len(channel_choices) == sum(self.stage_repeats)
        x = self.stem(x)
        block_idx = 0
        for m in self.features:
            if isinstance(m ,ShuffleNasBlock):
                x = m(x, block_choices[block_idx], channel_choices[block_idx])
                block_idx += 1
            else:
                x = m(x)
        assert block_idx == len(block_choices)
        return x

    def _initialize_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                if 'stem' in name or 'SE' in name:
                    nn.init.normal_(m.weight, 0, 0.01)
                else:
                    nn.init.normal_(m.weight, 0, 1.0 / m.weight.shape[1])
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

class ShuffleNetv2PlusClassifierHead(nn.Module):
    def __init__(self, in_channels, num_classes, featc=1024):
        super(ShuffleNetv2PlusClassifierHead, self).__init__()
        featc = int(featc * 0.75)
        self.v3_conv = ConvModule(in_channels, featc, 1, activation='hs')
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.v3_se = SEModule(featc)
        self.v3_fc = nn.Linear(featc, featc, bias=False)
        self.v3_hs = HSwish()
        self.dropout = nn.Dropout(0.2)
        self.v3_fc2 = nn.Linear(featc, num_classes, bias=False)

    def forward(self, x):
        x = self.v3_conv(x)
        x = self.gap(x)
        x = self.v3_se(x)
        x = x.contiguous().view(x.size(0), -1)
        x = self.v3_fc(x)
        x = self.dropout(x)
        x = self.v3_fc2(x)
        return x

