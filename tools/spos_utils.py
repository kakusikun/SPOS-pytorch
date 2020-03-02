import random
import numpy as np
from tools.flops_utils import get_flops_table, get_flop_params

class Evolution:
    def __init__(self, 
        cfg,
        graph, 
        pool_target_size=20, 
        children_size=10, 
        parent_size=5, 
        num_batches=20000,
        mutate_ratio=0.1,  
        flops_cuts=7,
        children_pick_interval=3, 
        logger=None):
        self.graph = graph
        self.pool_target_size = pool_target_size
        self.children_size = children_size
        self.parent_size = parent_size
        self.mutate_ratio = mutate_ratio
        self.flops_cuts = flops_cuts
        self.parents = []
        self.children = []       

        self.lookup_table = get_flops_table(
            input_size=cfg.INPUT.RESIZE[0],
            n_class=cfg.DB.NUM_CLASSES,
            use_se=cfg.SPOS.USE_SE, 
            last_conv_after_pooling=cfg.SPOS.LAST_CONV_AFTER_POOLING,
            last_conv_out_channel=self.graph.model.last_conv_out_channel,
            channels_layout=cfg.SPOS.CHANNELS_LAYOUT
            )
        upper_flops, bottom_flops, upper_params, bottom_params = self.set_flops_params_bound()
        # [top to bottom then bottom to top] 
        self.flops_interval = (upper_flops - bottom_flops) / flops_cuts
        self.flops_ranges = [max(upper_flops - i * self.flops_interval, 0) for i in range(flops_cuts)] + \
                            [max(upper_flops - i * self.flops_interval, 0) for i in range(flops_cuts)][::-1]

        # Use worse children of the good parents
        # If the children are too outstanding, the distribution coverage ratio will be low
        # [0, 3, 6, 9, 9, 6, 3, 0] => [6, 6, 6, 9, 9, 6, 6, 6]
        children_pick_ids = list(range(0, children_size, children_pick_interval)) + \
                                 list(reversed(range(0, children_size, children_pick_interval)))
        self.children_pick_ids = [6 if idx == 0 or idx == 3 else idx for idx in children_pick_ids]

        self.sample_counts = num_batches // len(self.flops_ranges) // len(self.children_pick_ids)

        param_interval = (upper_params - bottom_params) / (len(self.children_pick_ids) - 1)
        # [top to bottom] 
        self.param_range = [upper_params - i * param_interval for i in range(len(self.children_pick_ids))]

        self.cur_step = 0
        self.source_choice = {'channel_choices': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                            'block_choices': [0, 1, 2, 3]}



        p = next(iter(self.graph.model.parameters()))
        if p.is_cuda:
            self.use_gpu = True

    def evolve(self, epoch_after_cs, pick_id, find_max_param, max_flops, upper_params, bottom_params, logger=None):
        '''
        Returns:
            selected_child(dict):
                a candidate that 
        '''
        # Prepare random parents for the initial evolution
        while len(self.parents) < self.parent_size:
            block_choices = self.graph.random_block_choices()
            channel_choices = self.graph.random_channel_choices(epoch_after_cs)
            flops, param = get_flop_params(block_choices, channel_choices, self.lookup_table)
            candidate = dict()
            candidate['block_choices'] = block_choices
            candidate['channel_choices'] = channel_choices
            candidate['flops'] = flops
            candidate['param'] = param
            self.parents.append(candidate)

        generation = 0.0
        # Breed children
        while len(self.children) < self.children_size:
            candidate = dict()
            # randomly select parents from current pool
            mother = random.choice(self.parents)
            father = random.choice(self.parents)

            # make sure mother and father are different
            while father is mother:
                mother = random.choice(self.parents)

            # breed block choice
            block_choices = [0] * len(father['block_choices'])
            for i in range(len(block_choices)):
                block_choices[i] = random.choice([mother['block_choices'][i], father['block_choices'][i]])
                # Mutation: randomly mutate some of the children.
                if random.random() < self.mutate_ratio:
                    block_choices[i] = random.choice(self.source_choice['block_choices'])

            # breed channel choice
            channel_choices = [0] * len(father['channel_choices'])
            for i in range(len(channel_choices)):
                channel_choices[i] = random.choice([mother['channel_choices'][i], father['channel_choices'][i]])
                # Mutation: randomly mutate some of the children.
                if random.random() < self.mutate_ratio:
                    channel_choices[i] = random.choice(self.source_choice['channel_choices'])

            flops, param = get_flop_params(block_choices, channel_choices, self.lookup_table)

            # if flops > max_flop or model_size > upper_params:
            if flops < (max_flops-self.flops_interval) or flops > max_flops \
                    or param < bottom_params or param > upper_params:
                continue

            candidate['block_choices'] = block_choices
            candidate['channel_choices'] = channel_choices
            candidate['flops'] = flops
            candidate['param'] = param
            generation += 1
            self.children.append(candidate)
            if logger:
                logger.info(f"Get child after {generation} generations")
        # Set target and select
        self.children.sort(key=lambda cand: cand['param'], reverse=find_max_param)
        selected_child = self.children[pick_id]

        # Update step for the strolling evolution
        self.cur_step += 1

        # prepare for next evolve
        self.parents = self.children[:self.parent_size]
        self.children = []
        return selected_child

    def get_cur_evolve_state(self):
        '''
        walk(cur_step) on the map(flop x param) from large param to small given flop from which large to small then
        walk back.
        '''
        self.cur_step = self.cur_step % (self.sample_counts * len(self.children_pick_ids) * len(self.flops_ranges))
        i = self.cur_step // (len(self.children_pick_ids) * self.sample_counts)
        j = self.cur_step % (len(self.children_pick_ids) * self.sample_counts) // self.sample_counts
        range_id = j if i % 2 == 0 else len(self.children_pick_ids) - 1 - j
        find_max_param = False
        if (i % 2 == 0 and j < len(self.children_pick_ids) // 2) or \
                (not i % 2 == 0 and j >= len(self.children_pick_ids) // 2):
            find_max_param = True
        return self.flops_ranges[i], self.children_pick_ids[j], range_id, find_max_param

    def maintain(self, epoch_after_cs, pool, lock, finished_flag, logger=None):
        while not finished_flag.value:
            if len(pool) < self.pool_target_size:
                max_flops, pick_id, range_id, find_max_param = self.get_cur_evolve_state()
                if find_max_param:
                    info = f"[Evolution] Find max params   Max Flops [{max_flops:.2f}]   Child Pick ID [{pick_id}]   Upper model size [{self.param_range[range_id]:.2f}]   Bottom model size [{self.param_range[-1]:.2f}]" 
                    if logger and self.cur_step % self.sample_counts == 0:
                        logger.info('-' * 40 + '\n' + info)
                    candidate = self.evolve(
                        epoch_after_cs,
                        pick_id, 
                        find_max_param, 
                        max_flops,
                        upper_params=self.param_range[range_id],
                        bottom_params=self.param_range[-1],
                        logger=logger
                    )
                else:
                    info = f"[Evolution] Find min params   Max Flops [{max_flops:.2f}]   Child Pick ID [{pick_id}]   Upper model size [{self.param_range[range_id]:.2f}]   Bottom model size [{self.param_range[-1]:.2f}]" 
                    if logger and self.cur_step % self.sample_counts == 0:
                        logger.info('-' * 40 + '\n' + info)
                    candidate = self.evolve(
                        epoch_after_cs,
                        pick_id, 
                        find_max_param, 
                        max_flops,
                        upper_params=self.param_range[0],
                        bottom_params=self.param_range[range_id],
                        logger=logger
                    )
                with lock:
                    candidate['channel_masks'] = self.graph.get_channel_masks(candidate['channel_choices'])
                    pool.append(candidate)
        logger.info("[Evolution] Ends")
    
    def set_flops_params_bound(self):
        block_choices = [3] * sum(self.graph.stage_repeats)
        channel_choices = [9] * sum(self.graph.model.stage_repeats)
        upper_flops, upper_params = get_flop_params(block_choices, channel_choices, self.lookup_table)
        block_choices = [0] * sum(self.graph.model.stage_repeats)
        channel_choices = [0] * sum(self.graph.model.stage_repeats)        
        bottom_flops, bottom_params = get_flop_params(block_choices, channel_choices, self.lookup_table)
        return upper_flops*0.5, bottom_flops, upper_params*0.5, bottom_params

def make_divisible(x, divisible_by=8):
    return int(np.ceil(x * 1. / divisible_by) * divisible_by)

def recalc_bn(graph, block_choices, channel_masks, bndata, use_gpu, bn_recalc_imgs=20000):
    graph.model.train()
    count = 0
    for batch in bndata:
        if use_gpu:
            img = batch['inp'].cuda()
        graph.model(img, block_choices, channel_masks)

        count += img.size(0)        
        if count > bn_recalc_imgs:
            break

if __name__ == "__main__":
    import torch
    from config.config_factory import _C as cfg
    from src.model.backbone.shufflenas import shufflenas_oneshot

    cfg.INPUT.RESIZE = (224,224)
    cfg.DB.NUM_CLASSES = 10
    cfg.SPOS.USE_SE = True
    cfg.SPOS.LAST_CONV_AFTER_POOLING = True
    cfg.SPOS.CHANNELS_LAYOUT = "OneShot"

    model = shufflenas_oneshot(
        n_class=10,
        use_se=True,
        last_conv_after_pooling=True,
        channels_layout='OneShot')

    darwin = Evolution(cfg, model)

    max_flops, pick_id, range_id, find_max_param = darwin.get_cur_evolve_state()

    if find_max_param:    
        candidate = darwin.evolve(pick_id, find_max_param, max_flops,
                                upper_params=darwin.param_range[range_id],
                                bottom_params=darwin.param_range[-1])
    else:   
        candidate = darwin.evolve(pick_id, find_max_param, max_flops,
                                upper_params=darwin.param_range[0],
                                bottom_params=darwin.param_range[range_id])