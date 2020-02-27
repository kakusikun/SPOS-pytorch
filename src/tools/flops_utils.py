import os
import sys
import json

import torch
import torch.nn as nn
from src.model.module.spos_modules import *
from src.model.module.base_module import *
from thop import profile
import logging
logger = logging.getLogger('logger')

def count_hs(m, x, y):
    x = x[0]

    nelements = x.numel()

    m.total_ops += torch.Tensor([int(nelements * 2)])

def get_flop_params(all_block_choice, all_channel_choice, lookup_table):
    assert isinstance(all_block_choice, list) and isinstance(all_channel_choice, list)
    flops = lookup_table['flops']['input_block'] + lookup_table['flops']['output_block']
    params = lookup_table['params']['input_block'] + lookup_table['params']['output_block']

    for block_idx, (block_choice, channel_choice) in enumerate(zip(all_block_choice, all_channel_choice)):
        choice_id = f"{block_idx}-{block_choice}-{channel_choice}"
        flops += lookup_table['flops']['nas_block'][choice_id]
        params += lookup_table['params']['nas_block'][choice_id]
    return flops, params

def get_flops_table(
    input_size,
    n_class,
    use_se,
    last_conv_after_pooling,
    last_conv_out_channel,
    channels_layout='OneShot'
    ):
    root = os.getcwd()
    file_path = os.path.join(root, 'external/OneShot_flops.json')   

    if not os.path.exists(file_path):
        logger.info("FLOPs Table is not found, generating ...")
        make_OneShot_flops_table(
            input_size=input_size, 
            n_class=n_class, 
            use_se=use_se,
            last_conv_after_pooling=last_conv_after_pooling,
            last_conv_out_channel=last_conv_out_channel
        )
        logger.info(f"Done. FLOPs Table is placed at {file_path}")
    
    with open(file_path, 'r') as f:
        lookup_table = json.load(f)
    
    return lookup_table

def make_OneShot_flops_table(
    input_size,
    n_class,
    use_se,
    last_conv_after_pooling,
    last_conv_out_channel,
    channels_layout='OneShot'
    ):
    
    input_channel = 16
    stage_repeats = [4, 4, 8, 4]
    stage_out_channels = [64, 160, 320, 640]
    channel_scales = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0] 
    block_choices = [0, 1, 2, 3]

    lookup_table = dict()
    lookup_table['config'] = dict()
    lookup_table['config']['use_se'] = use_se
    lookup_table['config']['last_conv_after_pooling'] = last_conv_after_pooling
    lookup_table['config']['channels_layout'] = channels_layout
    lookup_table['config']['stage_repeats'] = stage_repeats
    lookup_table['config']['stage_out_channels'] = stage_out_channels
    lookup_table['config']['channel_scales'] = channel_scales
    lookup_table['config']['block_choices'] = ['ShuffleNetV2_3x3', 'ShuffleNetV2_5x5', 'ShuffleNetV2_7x7', 'ShuffleXception']
    lookup_table['config']['first_conv_out_channel'] = input_channel
    lookup_table['config']['input_size'] = input_size
    lookup_table['config']['last_conv_out_channel'] = last_conv_out_channel
    lookup_table['flops'] = dict()
    lookup_table['params'] = dict()
    lookup_table['flops']['input_block'] = 0.0
    lookup_table['params']['input_block'] = 0.0
    lookup_table['flops']['nas_block'] = {}
    lookup_table['params']['nas_block'] = {}
    lookup_table['flops']['output_block'] = 0.0
    lookup_table['params']['output_block'] = 0.0

    inp = torch.rand(1, 3, input_size, input_size)
    input_block = nn.Sequential(            
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
    input_block_flops, input_block_params = profile(input_block, inputs=(inp,), custom_ops={HardSwish:count_hs})
    input_block.eval()
    inp = input_block(inp)
    lookup_table['flops']['input_block'] = input_block_flops / 1e6
    lookup_table['params']['input_block'] = input_block_params / 1e6

    block_idx = 0
    global_max_length = make_divisible(int(stage_out_channels[-1] // 2 * channel_scales[-1]))
    for stage_id in range(len(stage_repeats)):
        numrepeat = stage_repeats[stage_id]
        output_channel = stage_out_channels[stage_id]

        if use_se:
            act_name = 'hard_swish' if stage_id >= 1 else 'relu'
            block_use_se = True if stage_id >= 2 else False
        else:
            act_name = 'relu'
            block_use_se = False
        # create repeated blocks for current stage
        for i in range(numrepeat):
            stride = 2 if i == 0 else 1
            for block_choice in block_choices:
                for channel_choice, scale in enumerate(channel_scales):     
                    local_mask = [0] * global_max_length
                    mid_channel = make_divisible(int(output_channel // 2 * scale))
                    for j in range(mid_channel):
                        local_mask[j] = 1
                        local_mask = torch.Tensor(local_mask)
                    # SNB 3x3
                    if block_choice == 0:
                        block = ShuffleNetCSBlock(
                            input_channel, output_channel, mid_channel,
                            3, stride, 'ShuffleNetV2', act_name=act_name, use_se=block_use_se)
                    if block_choice == 1:
                        block = ShuffleNetCSBlock(
                            input_channel, output_channel, mid_channel,
                            5, stride, 'ShuffleNetV2', act_name=act_name, use_se=block_use_se)
                    if block_choice == 2:
                        block = ShuffleNetCSBlock(
                            input_channel, output_channel, mid_channel,
                            7, stride, 'ShuffleNetV2', act_name=act_name, use_se=block_use_se)
                    if block_choice == 3:
                        block = ShuffleNetCSBlock(
                            input_channel, output_channel, mid_channel,
                            3, stride, 'ShuffleXception', act_name=act_name, use_se=block_use_se)
                    # fill the table
                    choice_id = f"{block_idx}-{block_choice}-{channel_choice}"
                    block_flops, block_params = profile(block, inputs=(inp, local_mask), custom_ops={HardSwish:count_hs})
                    lookup_table['flops']['nas_block'][choice_id] = block_flops / 1e6
                    lookup_table['params']['nas_block'][choice_id] = block_params / 1e6
            if stride == 2:
                input_size //= 2
            block.eval()
            inp = block(inp, torch.ones(global_max_length))
            input_channel = output_channel
            block_idx += 1
    
    if last_conv_after_pooling:
        # MobileNet V3 approach
        output_block = [
            nn.AdaptiveAvgPool2d(1),
            # no last SE for MobileNet V3 style
            nn.Conv2d(input_channel, last_conv_out_channel, kernel_size=1, stride=1,
                        padding=0, bias=True),
            # No bn for the conv after pooling
            HardSwish()
        ]
    else:
        if use_se:
            # ShuffleNetV2+ approach
            output_block = [
                nn.Conv2d(input_channel, make_divisible(last_conv_out_channel * 0.75), 
                            kernel_size=1, stride=1,
                            padding=0, bias=False),
                nn.BatchNorm2d(make_divisible(last_conv_out_channel * 0.75), affine=False),
                HardSwish(),
                nn.AdaptiveAvgPool2d(1),
                SEModule(make_divisible(last_conv_out_channel * 0.75)),
                nn.Conv2d(last_conv_out_channel, make_divisible(last_conv_out_channel * 0.75), 
                            kernel_size=1, stride=1,
                            padding=0, bias=True),
                # No bn for the conv after pooling
                HardSwish()
            ]
        else:
            # original Oneshot Nas approach
            output_block = [
                nn.Conv2d(input_channel, last_conv_out_channel, 
                            kernel_size=1, stride=1,
                            padding=0, bias=False),
                nn.BatchNorm2d(last_conv_out_channel, affine=False),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d(1)
            ]
            
    output_block.append(nn.Dropout(0.2 if use_se else 0.1))
    output_block.append(nn.Conv2d(last_conv_out_channel, n_class, 
                    kernel_size=1, stride=1,
                    padding=0, bias=True),
                )
    output_block = nn.Sequential(*output_block)
    output_block_flops, output_block_params = profile(output_block, inputs=(inp,), custom_ops={HardSwish:count_hs})
    output_block.eval()
    inp = output_block(inp)
    lookup_table['flops']['output_block'] = output_block_flops / 1e6
    lookup_table['params']['output_block'] = output_block_params / 1e6

    root = os.getcwd()
    with open(os.path.join(root, 'external/OneShot_flops.json'), 'w') as f:
        json.dump(lookup_table, f)

if __name__ == '__main__':
    make_OneShot_flops_table(
        input_size=224, 
        n_class=10, 
        use_se=True,
        last_conv_after_pooling=True,
        last_conv_out_channel=1024
    )