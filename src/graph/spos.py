from sys import maxsize
from src.graph import *
from tools.spos_utils import make_divisible
from src.model.module.loss_module import CrossEntropyLossLS
import random

class SPOS(BaseGraph):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.stage_repeats = [4, 4, 8, 4]
        self.stage_out_channels = [64, 160, 320, 640]
        self.candidate_scales = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
        self.last_conv_out_channel = 1024  

    def build(self):
        self.model = shufflenas_oneshot(
            n_class=self.cfg.DB.NUM_CLASSES,
            use_se=self.cfg.SPOS.USE_SE,
            last_conv_after_pooling=self.cfg.SPOS.LAST_CONV_AFTER_POOLING,
            channels_layout=self.cfg.SPOS.CHANNELS_LAYOUT)
    
        self.crit = {}
        self.crit['cels'] = CrossEntropyLossLS(self.cfg.DB.NUM_CLASSES)

        def loss_head(feat, batch):
            losses = {'cels':self.crit['cels'](feat, batch['target'])}
            loss = losses['cels']
            return loss, losses

        self.loss_head = loss_head


    def random_block_choices(self, num_of_block_choices=4, select_predefined_block=False):
        if select_predefined_block:
            block_choices = [0, 0, 3, 1, 1, 1, 0, 0, 2, 0, 2, 1, 1, 0, 2, 0, 2, 1, 3, 2]
        else:
            block_number = sum(self.stage_repeats)
            block_choices = []
            for _ in range(block_number):
                block_choices.append(random.randint(0, num_of_block_choices - 1))

        return block_choices

    def random_channel_choices(self, epoch_after_cs=maxsize):
        """
        candidate_scales = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
        mode: str, "dense" or "sparse". Sparse mode select # channel from candidate scales. Dense mode selects
              # channels between randint(min_channel, max_channel).
        """
        assert len(self.stage_repeats) == len(self.stage_out_channels)
        # From [1.0, 1.2, 1.4, 1.6, 1.8, 2.0] to [0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0], warm-up stages are
        # not just 1 epoch, but 2, 3, 4, 5 accordingly.

        epoch_delay_early = {0: 0,  # 8
                             1: 1, 2: 1,  # 7
                             3: 2, 4: 2, 5: 2,  # 6
                             6: 3, 7: 3, 8: 3, 9: 3,  # 5
                             10: 4, 11: 4, 12: 4, 13: 4, 14: 4,
                             15: 5, 16: 5, 17: 5, 18: 5, 19: 5, 20: 5,
                             21: 6, 22: 6, 23: 6, 24: 6, 25: 6, 27: 6, 28: 6,
                             29: 6, 30: 6, 31: 6, 32: 6, 33: 6, 34: 6, 35: 6, 36: 7,
                           }
        epoch_delay_late = {0: 0,
                            1: 1,
                            2: 2,
                            3: 3,
                            4: 4, 5: 4,  # warm up epoch: 2 [1.0, 1.2, ... 1.8, 2.0]
                            6: 5, 7: 5, 8: 5,  # warm up epoch: 3 ...
                            9: 6, 10: 6, 11: 6, 12: 6,  # warm up epoch: 4 ...
                            13: 7, 14: 7, 15: 7, 16: 7, 17: 7,  # warm up epoch: 5 [0.4, 0.6, ... 1.8, 2.0]
                            18: 8, 19: 8, 20: 8, 21: 8, 22: 8, 23: 8}  # warm up epoch: 6, after 17, use all scales
        select_all_channels = False
        if epoch_after_cs < 0:
            select_all_channels = True
        else:
            if 0 <= epoch_after_cs <= 23 and self.stage_out_channels[0] >= 64:
                delayed_epoch_after_cs = epoch_delay_late[epoch_after_cs]
            elif 0 <= epoch_after_cs <= 36 and self.stage_out_channels[0] < 64:
                delayed_epoch_after_cs = epoch_delay_early[epoch_after_cs]
            else:
                delayed_epoch_after_cs = epoch_after_cs

        min_scale_id = 0

        channel_choices = []
        for i in range(len(self.stage_out_channels)):
            for _ in range(self.stage_repeats[i]):
                if select_all_channels:
                    channel_choices = [len(self.candidate_scales) - 1] * sum(self.stage_repeats)
                else:
                    channel_scale_start = max(min_scale_id, len(self.candidate_scales) - delayed_epoch_after_cs - 2)
                    channel_choice = random.randint(channel_scale_start, len(self.candidate_scales) - 1)
                    # In sparse mode, channel_choices is the indices of candidate_scales
                    channel_choices.append(channel_choice)
        return channel_choices

    def get_channel_masks(self, channel_choices):
        """
        candidate_scales = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
        mode: str, "dense" or "sparse". Sparse mode select # channel from candidate scales. Dense mode selects
              # channels between randint(min_channel, max_channel).
        """
        assert len(self.stage_repeats) == len(self.stage_out_channels)

        channel_masks = []
        global_max_length = make_divisible(int(self.stage_out_channels[-1] // 2 * self.candidate_scales[-1]))
        for i in range(len(self.stage_out_channels)):
            for _ in range(self.stage_repeats[i]):
                local_mask = [0] * global_max_length
                # this is for channel selection warm up: channel choice ~ (8, 9) -> (7, 9) -> ... -> (0, 9)
                select_channel = int(self.stage_out_channels[i] // 2 *
                                            self.candidate_scales[channel_choices[i]])
                # To take full advantages of acceleration, # of channels should be divisible to 8.
                select_channel = make_divisible(select_channel)
                for j in range(select_channel):
                    local_mask[j] = 1
                channel_masks.append(local_mask)
                
        return torch.Tensor(channel_masks).expand(len(self.cfg.MODEL.GPU), sum(self.stage_repeats), global_max_length)

