import sys
import time
import argparse
import random

import multiprocessing
from multiprocessing import Value
from ctypes import c_bool
from copy import deepcopy

import torch
import torch.nn as nn
from torch.utils import data
from src.factory.config_factory import _C as cfg
from src.factory.config_factory import build_output
from tools.logger import setup_logger
from tools.utils import deploy_macro, print_config

from src.graph.spos import SPOS
from tools.spos_utils import Evolution, recalc_bn
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
                        logger.info('-' * 40)
                        logger.info(f"[Train] Block Choices: {cand['block_choices']}")
                        logger.info(f"[Train] Channel Choice: {cand['channel_choices']}")
                        logger.info(f"[Train] Flop: {cand['flops']:.2f}M, param: {cand['param']:.2f}M")
            else:
                time.sleep(1)

        solver.zero_grad()
        channel_masks = cand['channel_masks'].cuda()
        block_choices = cand['block_choices']            
        outputs = graph.model(batch['inp'], block_choices, channel_masks)
        loss, _ = graph.loss_head(outputs, batch)
        loss.backward()
        solver.step()
        accu = (outputs.max(1)[1] == batch['target']).float().mean()
        if i % 50 == 0:
            logger.info(f"Epoch [{epoch:03}]   Step [{i:04}]   loss [{loss:3.3f}]   accu [{accu:.3f}]")
        accus.append(accu)
    train_accu = tensor_to_scalar(torch.stack(accus).mean())
    logger.info(f"Epoch [{epoch:03}]   Train Accuracy [{train_accu:3.3f}]")
    shared_finished_flag.value = True
    return

def test_once(logger, epoch, graph, vdata, bndata):
    block_choices = graph.random_block_choices()
    channel_choices = graph.random_channel_choices()
    channel_masks = graph.get_channel_masks(channel_choices)
    raw_model_state = deepcopy(graph.model.state_dict())
    recalc_bn(graph, block_choices, channel_masks, bndata, True)
    graph.model.eval()
    accus = []
    for batch in vdata:
        with torch.no_grad():
            for key in batch:
                batch[key] = batch[key].cuda()
            outputs = graph.model(batch['inp'], block_choices, channel_masks)
        accus.append((outputs.max(1)[1] == batch['target']).float().mean())
    accu = tensor_to_scalar(torch.stack(accus).mean())
    logger.info(f"Epoch [{epoch:03}]   Validation Accuracy [{accu:3.3f}]")
    graph.model.load_state_dict(raw_model_state)
    return accu
    
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
    evolution = Evolution(
        cfg=cfg,
        graph=graph,
        num_batches=len(loader['train']),
        logger=logger
    )

    graph.use_multigpu()        
    solver = Solver(cfg, graph.model.named_parameters())

    manager = multiprocessing.Manager()
    cand_pool = manager.list()
    lock = manager.Lock()

    best_accu = 0.0

    for epoch in range(cfg.SOLVER.START_EPOCH, cfg.SOLVER.MAX_EPOCHS):

        finished = Value(c_bool, False)
        pool_process = multiprocessing.Process(target=evolution.maintain,
                                        args=[epoch - cfg.SPOS.EPOCH_TO_CS, cand_pool, lock, finished, logger])
        pool_process.start()
        train_once(logger, epoch, graph, loader['train'], solver, pool=cand_pool, pool_lock=lock, shared_finished_flag=finished)
        pool_process.join()

        test_accu = test_once(logger, epoch, graph, loader['val'], loader['train'])
        if test_accu > best_accu:
            best_accu = test_accu
            logger.info(f"Epoch [{epoch:03}]   Best Accuracy [{best_accu:3.3f}]")

        graph.save(graph.save_path, graph.model, solvers={'main': solver}, epoch=epoch, metric=test_accu)


    
if __name__ == "__main__":
    main()