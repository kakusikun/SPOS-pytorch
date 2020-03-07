import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from src.model.module.base_module import SEModule

class HardSwish(nn.Module):
    def __init__(self):
        super(HardSwish, self).__init__()

    def forward(self, x, **kwargs):
        clip = torch.clamp(x + 3, 0, 6) / 6
        return x * clip

class ShufflenetCS(nn.Module):
    def __init__(self, inp, oup, base_mid_channels, ksize, stride, activation='relu', useSE=False, affine=False):
        super(ShufflenetCS, self).__init__()
        self.stride = stride
        assert stride in [1, 2]
        assert ksize in [3, 5, 7]

        self.base_mid_channel = base_mid_channels
        self.ksize = ksize
        pad = ksize // 2
        self.pad = pad
        self.inc = inp // 2 if stride == 1 else inp
        self.midc = base_mid_channels
        self.projc = inp // 2 if stride == 1 else inp
        self.ouc = oup - self.projc

        branch_main = [
            # pw
            nn.Conv2d(self.inc, self.midc, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.midc, affine=affine),
            None,
            # dw
            nn.Conv2d(self.midc, self.midc, ksize, stride, pad, groups=self.midc, bias=False),
            nn.BatchNorm2d(self.midc, affine=affine),
            # pw-linear
            nn.Conv2d(self.midc, self.ouc, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.ouc, affine=affine),
            None,
        ]
        if activation == 'relu':
            assert useSE == False
            '''This model should not have SE with ReLU'''
            branch_main[2] = nn.ReLU(inplace=True)
            branch_main[-1] = nn.ReLU(inplace=True)
        else:
            branch_main[2] = HardSwish()
            branch_main[-1] = HardSwish()
            if useSE:
                branch_main.append(SEModule(self.ouc))
        self.branch_main = nn.Sequential(*branch_main)

        if stride == 2:
            branch_proj = [
                # dw
                nn.Conv2d(self.projc, self.projc, ksize, stride, pad, groups=self.projc, bias=False),
                nn.BatchNorm2d(self.projc, affine=affine),
                # pw-linear
                nn.Conv2d(self.projc, self.projc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(self.projc, affine=affine),
                None,
            ]
            if activation == 'relu':
                branch_proj[-1] = nn.ReLU(inplace=True)
            else:
                branch_proj[-1] = HardSwish()
            self.branch_proj = nn.Sequential(*branch_proj)
        else:
            self.branch_proj = None

    def forward(self, old_x):
        if self.stride==1:
            x_proj, x = channel_shuffle(old_x)
            return torch.cat((x_proj, self.branch_main(x)), 1)
        elif self.stride==2:
            x_proj = old_x
            x = old_x
            return torch.cat((self.branch_proj(x_proj), self.branch_main(x)), 1)

class ShuffleXceptionCS(nn.Module):

    def __init__(self, inp, oup, base_mid_channels, stride, activation='relu', useSE=False, affine=False):
        super(ShuffleXceptionCS, self).__init__()

        assert stride in [1, 2]

        self.base_mid_channel = base_mid_channels
        self.stride = stride
        self.ksize = 3
        self.pad = 1
        self.inc = inp // 2 if stride == 1 else inp
        self.midc = base_mid_channels
        self.projc = inp // 2 if stride == 1 else inp
        self.ouc = oup - self.projc

        branch_main = [
            # dw
            nn.Conv2d(self.inc, self.inc, 3, stride, 1, groups=self.inc, bias=False),
            nn.BatchNorm2d(self.inc, affine=affine),
            # pw
            nn.Conv2d(self.inc, self.midc, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.midc, affine=affine),
            None,
            # dw
            nn.Conv2d(self.midc, self.midc, 3, 1, 1, groups=self.midc, bias=False),
            nn.BatchNorm2d(self.midc, affine=affine),
            # pw
            nn.Conv2d(self.midc, self.midc, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.midc, affine=affine),
            None,
            # dw
            nn.Conv2d(self.midc, self.midc, 3, 1, 1, groups=self.midc, bias=False),
            nn.BatchNorm2d(self.midc, affine=affine),
            # pw
            nn.Conv2d(self.midc, self.ouc, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.ouc, affine=affine),
            None,
        ]

        if activation == 'relu':
            branch_main[4] = nn.ReLU(inplace=True)
            branch_main[9] = nn.ReLU(inplace=True)
            branch_main[-1] = nn.ReLU(inplace=True)
        else:
            branch_main[4] = HardSwish()
            branch_main[9] = HardSwish()
            branch_main[-1] = HardSwish()
        assert None not in branch_main

        if useSE:
            assert activation != 'relu'
            branch_main.append(SEModule(self.ouc))

        self.branch_main = nn.Sequential(*branch_main)

        if self.stride == 2:
            branch_proj = [
                # dw
                nn.Conv2d(self.projc, self.projc, 3, stride, 1, groups=self.projc, bias=False),
                nn.BatchNorm2d(self.projc, affine=affine),
                # pw-linear
                nn.Conv2d(self.projc, self.projc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(self.projc, affine=affine),
                None,
            ]
            if activation == 'relu':
                branch_proj[-1] = nn.ReLU(inplace=True)
            else:
                branch_proj[-1] = HardSwish()
            self.branch_proj = nn.Sequential(*branch_proj)

    def forward(self, old_x):
        if self.stride==1:
            x_proj, x = channel_shuffle(old_x)
            return torch.cat((x_proj, self.branch_main(x)), 1)
        elif self.stride==2:
            x_proj = old_x
            x = old_x
            return torch.cat((self.branch_proj(x_proj), self.branch_main(x)), 1)

def channel_shuffle(x):
    batchsize, num_channels, height, width = x.data.size()
    assert (num_channels % 4 == 0)
    x = x.reshape(batchsize * num_channels // 2, 2, height * width)
    x = x.permute(1, 0, 2)
    x = x.reshape(2, -1, num_channels // 2, height, width)
    return x[0], x[1]

class ShuffleNetCSBlock(nn.Module):
    def __init__(self, 
        input_channel, 
        output_channel, 
        channel_scales, 
        ksize, 
        stride, 
        block_mode='ShuffleNetV2', 
        act_name='relu', 
        use_se=False, 
        **kwargs):
        super(ShuffleNetCSBlock, self).__init__()
        assert stride in [1, 2]
        assert ksize in [3, 5, 7]
        assert block_mode in ['ShuffleNetV2', 'ShuffleXception']

        self.stride = stride
        self.ksize = ksize
        self.block_mode = block_mode
        self.input_channel = input_channel
        self.output_channel = output_channel
        """
        Regular block: (We usually have the down-sample block first, then followed by repeated regular blocks)
        Input[64] -> split two halves -> main branch: [32] --> mid_channels (final_output_C[64] // 2 * scale[1.4])
                        |                                       |--> main_out_C[32] (final_out_C (64) - input_C[32]
                        |
                        |-----> project branch: [32], do nothing on this half
        Concat two copies: [64 - 32] + [32] --> [64] for final output channel

        =====================================================================

        In "Single path one shot nas" paper, Channel Search is searching for the main branch intermediate #channel.
        And the mid channel is controlled / selected by the channel scales (0.2 ~ 2.0), calculated from:
            mid channel = block final output # channel // 2 * scale

        Since scale ~ (0, 2), this is guaranteed: main mid channel < final output channel
        """
        self.block = nn.ModuleList()
        for i in range(len(channel_scales)):
            mid_channel = make_divisible(int(output_channel // 2 * channel_scales[i]))
            if block_mode == 'ShuffleNetV2':
                self.block.append(ShufflenetCS(
                    self.input_channel, 
                    self.output_channel, 
                    mid_channel,
                    ksize=ksize,
                    stride=stride,
                    activation=act_name,
                    useSE=use_se,
                    affine=False
                ))
            elif block_mode == 'ShuffleXception':
                self.block.append(ShuffleXceptionCS(
                    self.input_channel, 
                    self.output_channel, 
                    mid_channel,
                    stride=stride,
                    activation=act_name,
                    useSE=use_se,
                    affine=False
                ))
    def forward(self, x, channel_choice):
        return self.block[channel_choice](x)

class ShuffleNasBlock(nn.Module):
    def __init__(self, 
        input_channel, 
        output_channel, 
        stride, 
        channel_scales, 
        act_name='relu', 
        use_se=False):
        super(ShuffleNasBlock, self).__init__()
        assert stride in [1, 2]
        """
        Four pre-defined blocks
        """
        self.block_sn_3x3 = ShuffleNetCSBlock(input_channel, output_channel, channel_scales,
                                                3, stride, 'ShuffleNetV2', act_name=act_name, use_se=use_se)
        self.block_sx_3x3 = ShuffleNetCSBlock(input_channel, output_channel, channel_scales,
                                                3, stride, 'ShuffleXception', act_name=act_name, use_se=use_se)

    def forward(self, x, block_choice, channel_choice):
        # ShuffleNasBlock has three inputs and passes two inputs to the ShuffleNetCSBlock
        if block_choice == 0:
            x = self.block_sn_3x3(x, channel_choice)
        elif block_choice == 3:
            x = self.block_sx_3x3(x, channel_choice)
        return x

def make_divisible(x, divisible_by=8):
    return int(np.ceil(x * 1. / divisible_by) * divisible_by)