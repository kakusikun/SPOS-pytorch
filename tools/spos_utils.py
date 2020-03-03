import random
import time
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
        flops_cuts=10,
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
        max_flops, min_flops, max_params, min_params = self.set_flops_params_bound()
        # [top to bottom then bottom to top] 
        self.flops_interval = (max_flops - min_flops) / flops_cuts
        self.flops_ranges = [max(max_flops - i * self.flops_interval, 0) for i in range(flops_cuts)] + \
                            [max(max_flops - i * self.flops_interval, 0) for i in range(flops_cuts)][::-1]

        # Use worse children of the good parents
        # If the children are too outstanding, the distribution coverage ratio will be low
        # [0, 3, 6, 9, 9, 6, 3, 0] => [6, 6, 6, 9, 9, 6, 6, 6]
        children_pick_ids = list(range(0, children_size, children_pick_interval)) + \
                                 list(reversed(range(0, children_size, children_pick_interval)))
        self.children_pick_ids = [6 if idx == 0 or idx == 3 else idx for idx in children_pick_ids]

        self.sample_counts = num_batches // len(self.flops_ranges) // len(self.children_pick_ids)

        self.param_interval = (max_params - min_params) / (len(self.children_pick_ids) - 1)
        # [top to bottom] 
        self.param_range = [max_params - i * self.param_interval for i in range(len(self.children_pick_ids))]

        self.cur_step = 0
        self.source_choice = {'channel_choices': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                            'block_choices': [0, 1, 2, 3]}



        p = next(iter(self.graph.model.parameters()))
        if p.is_cuda:
            self.use_gpu = True

    def evolve(self, epoch_after_cs, pick_id, find_max_param, max_flops, max_params, min_params, logger=None):
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

        # min/max_flop/param_count
        min_f_c = max_f_c = 0
        # Breed children
        start = time.time()
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
                    or param < min_params or param > max_params:
                if flops < (max_flops-self.flops_interval):
                    min_f_c += 1
                elif flops > max_flops:
                    max_f_c += 1

                duration = (time.time() - start) // 60
                if duration > 10: # cost too much time in evolution
                    if min_f_c > max_f_c:
                        info = f"{max_flops:.2f} => "
                        max_flops -= self.flops_interval
                        info += f"{max_flops:.2f}"
                        if logger:
                            logger.info("Max FLOPs is too large, adjusted: " + info)
                        min_f_c = 0

                    if max_f_c > min_f_c:
                        info = f"{max_flops:.2f} => "
                        max_flops += self.flops_interval
                        info += f"{max_flops:.2f}"
                        if logger:
                            logger.info("Max FLOPs is too small, adjusted: " + info)
                        max_f_c = 0
                    start = time.time()    
                else:
                    info = ". " * int(duration)
                    logger.info("\r" + info)
                continue

            candidate['block_choices'] = block_choices
            candidate['channel_choices'] = channel_choices
            candidate['flops'] = flops
            candidate['param'] = param
            self.children.append(candidate)
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
                        max_params=self.param_range[range_id],
                        min_params=self.param_range[-1],
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
                        max_params=self.param_range[0],
                        min_params=self.param_range[range_id],
                        logger=logger
                    )
                with lock:
                    candidate['channel_masks'] = self.graph.get_channel_masks(candidate['channel_choices'])
                    pool.append(candidate)
        logger.info("[Evolution] Ends")
    
    def set_flops_params_bound(self):
        block_choices = [3] * sum(self.graph.stage_repeats)
        channel_choices = [9] * sum(self.graph.model.stage_repeats)
        max_flops, max_params = get_flop_params(block_choices, channel_choices, self.lookup_table)
        block_choices = [0] * sum(self.graph.model.stage_repeats)
        channel_choices = [0] * sum(self.graph.model.stage_repeats)        
        min_flops, min_params = get_flop_params(block_choices, channel_choices, self.lookup_table)
        return max_flops, min_flops, max_params, min_params

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
    from src.factory.config_factory import _C as cfg
    from src.graph.spos import SPOS

    cfg.INPUT.RESIZE = (112, 112)
    cfg.DB.NUM_CLASSES = 10
    cfg.SPOS.USE_SE = True
    cfg.SPOS.LAST_CONV_AFTER_POOLING = True
    cfg.SPOS.CHANNELS_LAYOUT = "OneShot"

    graph = SPOS(cfg)

    evolution = Evolution(cfg, graph)

    max_flops, pick_id, range_id, find_max_param = evolution.get_cur_evolve_state()

    if find_max_param:    
        candidate = evolution.evolve(0 - cfg.SPOS.EPOCH_TO_CS, pick_id, find_max_param, max_flops,
                                max_params=evolution.param_range[range_id],
                                min_params=evolution.param_range[-1])
    else:   
        candidate = evolution.evolve(0 - cfg.SPOS.EPOCH_TO_CS, pick_id, find_max_param, max_flops,
                                max_params=evolution.param_range[0],
                                min_params=evolution.param_range[range_id])