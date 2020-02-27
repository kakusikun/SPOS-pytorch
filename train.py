import sys
import time
import argparse
import random

import multiprocessing
from multiprocessing import Value
from ctypes import c_bool

import torch
import torch.nn as nn
from torch.utils import data
from config.config_factory import _C as cfg
from config.config_factory import build_output
from tools.logger import setup_logger
from tools.utils import deploy_macro, print_config

from src.graph.spos import SPOS
from src.tools.spos_utils import Evolution, recalc_bn
from src.factory.loader_factory import LoaderFactory
from src.solver.solver import Solver

def tensor_to_scalar(tensor):
    if isinstance(tensor, list):
        scalar = []
        for _tensor in tensor:
            scalar.append(_tensor.item())
    elif isinstance(tensor, dict):
        scalar = {}
        for _tensor in tensor:
            scalar[_tensor] = tensor[_tensor].item()
    elif isinstance(tensor, torch.Tensor) and tensor.dim() != 0:
        if tensor.is_cuda:
            scalar = tensor.cpu().detach().numpy().tolist()
        else:
            scalar = tensor.detach().numpy().tolist()
    else:
        scalar = tensor.item()
    return scalar

def train_once(logger, epoch, graph, tdata, solver, pool=None, pool_lock=None, shared_finished_flag=None):
    accus = []
    graph.model.train()
    for i, batch in enumerate(tdata):
        for key in batch:
            batch[key] = batch[key].cuda()

        cand = None
        while cand is None:
            if len(pool) > 0:
                with pool_lock:
                    cand = pool.pop()
                    if i % 50 == 0:
                        logger.info('[Trainer] ' + '-' * 40)
                        logger.info("[Trainer] block_choices choice: {}".format(cand['block_list']))
                        logger.info("[Trainer] Channel choice: {}".format(cand['channel_list']))
                        logger.info("[Trainer] Flop: {}M, param: {}M".format(cand['flops'], cand['graph.model_size']))
            else:
                time.sleep(1)

        solver.zero_grad()
        all_channel_masks = cand['channel_masks'].cuda()
        all_block_choices = cand['block_choices']            
        outputs = graph.model(batch['inp'], all_block_choices, all_channel_masks)
        loss = graph.loss_head(outputs, batch['target'])
        loss.backward()
        solver.step()
        if i % 50 == 0:
            logger.info(f"Epoch [{epoch:03}]   Step [{i:04}]   loss [{loss:3.3f}]")
        accus.append((outputs.max(1)[1] == batch['target']).float().mean())
    train_accu = tensor_to_scalar(torch.stack(accus).mean())
    logger.info(f"Epoch [{epoch:03}]   Train Accuracy [{train_accu:3.3f}]")
    shared_finished_flag.value = True
    return

def main():
    parser = argparse.ArgumentParser(description="PyTorch Deep Learning")
    parser.add_argument("--config", default="", help="path to config file", type=str)
    parser.add_argument('--list', action='store_true',
                        help='list available config in factories')
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    if args.config != "":
        cfg.merge_from_file(args.config)
    cfg.merge_from_list(args.opts)    
    build_output(cfg, args.config)
    logger = setup_logger(cfg.OUTPUT_DIR)
    deploy_macro(cfg)    

    loader = LoaderFactory.produce(cfg)

    graph = SPOS(cfg)
        
    solver = Solver(cfg, graph.model.parameters())

    evolution = Evolution(cfg, graph, logger=logger)

    manager = multiprocessing.Manager()
    cand_pool = manager.list()
    lock = manager.Lock()

    for epoch in range(cfg.SOLVER.START_EPOCH, cfg.SOLVER.MAX_EPOCHS):

        finished = Value(c_bool, False)
        pool_process = multiprocessing.Process(target=evolution.maintain,
                                        args=[cand_pool, lock, finished, logger])
        pool_process.start()
        train_once(logger, epoch, graph, loader['train'], solver, pool=cand_pool, pool_lock=lock, shared_finished_flag=finished)
        pool_process.join()



    