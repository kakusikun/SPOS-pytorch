from sys import maxsize
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from src.model.module.spos_modules import HardSwish, ShuffleNasBlock
from src.model.module.base_module import SEModule
from tools.spos_utils import make_divisible

class ShuffleNasOneShot(nn.Module):
    def __init__(self,
        strides,
        stage_repeats,
        n_class=1000,
        use_se=False,
        last_conv_after_pooling=False,
        stage_out_channels=None,
        candidate_scales=None,
        last_conv_out_channel=512):
        super(ShuffleNasOneShot, self).__init__()
        self.strides = strides
        self.stage_repeats = stage_repeats
        self.stage_out_channels = stage_out_channels
        self.candidate_scales = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
        self.use_se = use_se

        input_channel = 16
        self.last_conv_after_pooling = last_conv_after_pooling
        self.last_conv_out_channel = last_conv_out_channel

        assert len(self.stage_repeats) == len(self.stage_out_channels)

        
        self.stem = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=input_channel,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False),
            nn.BatchNorm2d(input_channel, affine=False),
            HardSwish()
        )

        features = []
        block_id = 0
        for stage_id in range(len(self.stage_repeats)):
            numrepeat = self.stage_repeats[stage_id]
            output_channel = self.stage_out_channels[stage_id]

            if self.use_se:
                act_name = 'hard_swish' if stage_id >= 1 else 'relu'
                block_use_se = True if stage_id >= 2 else False
            else:
                act_name = 'relu'
                block_use_se = False
            # create repeated blocks for current stage
            for i in range(numrepeat):
                stride = self.strides[stage_id] if i == 0 else 1
                block_id += 1
                features.extend([
                    ShuffleNasBlock(
                        input_channel,
                        output_channel,
                        stride=stride,
                        max_channel_scale=self.candidate_scales[-1],
                        use_se=block_use_se,
                        act_name=act_name)
                ])
                # update input_channel for next block
                input_channel = output_channel

        if self.last_conv_after_pooling:
            # MobileNet V3 approach
            features.extend([
                nn.AdaptiveAvgPool2d(1),
                # no last SE for MobileNet V3 style
                nn.Conv2d(input_channel, self.last_conv_out_channel, kernel_size=1, stride=1,
                            padding=0, bias=True),
                # No bn for the conv after pooling
                HardSwish()
            ])
        else:
            if self.use_se:
                # ShuffleNetV2+ approach
                features.extend([
                    nn.Conv2d(input_channel, make_divisible(self.last_conv_out_channel * 0.75),
                                kernel_size=1, stride=1,
                                padding=0, bias=False),
                    nn.BatchNorm2d(make_divisible(self.last_conv_out_channel * 0.75), affine=False),
                    HardSwish(),
                    nn.AdaptiveAvgPool2d(1),
                    SEModule(make_divisible(self.last_conv_out_channel * 0.75)),
                    nn.Conv2d(self.last_conv_out_channel, make_divisible(self.last_conv_out_channel * 0.75),
                                kernel_size=1, stride=1,
                                padding=0, bias=True),
                    # No bn for the conv after pooling
                    HardSwish()
                ])
            else:
                # original Oneshot Nas approach
                features.extend([
                    nn.Conv2d(input_channel, self.last_conv_out_channel,
                                kernel_size=1, stride=1,
                                padding=0, bias=False),
                    nn.BatchNorm2d(self.last_conv_out_channel, affine=False),
                    nn.ReLU(inplace=True),
                    nn.AdaptiveAvgPool2d(1)
                ])
        features.extend([nn.Dropout(0.2 if self.use_se else 0.1)])
        self.features = nn.ModuleList(features)

        self.output = nn.Sequential(
            nn.Conv2d(self.last_conv_out_channel, n_class,
                        kernel_size=1, stride=1,
                        padding=0, bias=True),
        )

        self._initialize_weights()

    def forward(self, x, block_choices, channel_masks):
        assert len(block_choices) == sum(self.stage_repeats) 
        assert channel_masks.shape[1] == sum(self.stage_repeats)
        x = self.stem(x)
        block_idx = 0
        for m in self.features:
            if isinstance(m ,ShuffleNasBlock):
                x = m(x, block_choices[block_idx], channel_masks.squeeze()[block_idx])
                block_idx += 1
            else:
                x = m(x)
        assert block_idx == len(block_choices)
        x = self.output(x).view(x.size(0), -1)
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


def shufflenas_oneshot(
    strides,
    stage_repeats,
    stage_out_channels,
    last_conv_out_channel,
    n_class=1000,
    use_se=False,
    last_conv_after_pooling=False,
    channels_layout='OneShot'):
    '''
    architecture(list):
        parsed from string which delimited by space, indicating the all block type in model.
        The length is same as number of blocks.
    scale_ids(list):
        parsed from string which delimited by space, indicating the mid channel scale of blocks in model.
        The length is same as number of blocks.
    use_se, last_conv_after_pooling(bool):
        trick in MobileNetv3
    channels_layout:
        OneShot model template
    '''

    if channels_layout == 'OneShot':
        candidate_scales = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
    else:
        raise ValueError("Unrecognized channels_layout: {}. "
                         "Please choose from ['OneShot']".format(channels_layout))

        # Nothing about architecture is specified, do random block selection and channel selection.
    return ShuffleNasOneShot(
        strides=strides,
        stage_repeats=stage_repeats,
        n_class=n_class,
        use_se=use_se,
        last_conv_after_pooling=last_conv_after_pooling,
        stage_out_channels=stage_out_channels,
        candidate_scales=candidate_scales,
        last_conv_out_channel=last_conv_out_channel)


if __name__ == '__main__':
    model = shufflenas_oneshot(
        n_class=10,
        use_se=True,
        last_conv_after_pooling=True,
        channels_layout='OneShot'
    )

    all_channel_mask, choice = model.random_channel_mask(epoch_after_cs=1)
    block_choices = model.random_block_choices()
    x = torch.rand(2,3,112,112)
    _ = model(x, block_choices, all_channel_mask)