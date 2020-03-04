import sys
import argparse
import heapq

from src.factory.config_factory import _C as cfg
from src.factory.config_factory import build_output
from tools.logger import setup_logger
from tools.utils import deploy_macro, print_config

from src.graph.spos import SPOS
from tools.spos_utils import SearchEvolution
from src.factory.loader_factory import LoaderFactory

class LeaderBoard(object):
    def __init__(self, k):
        self.k = k
        self.data = []

    def push(self, elem):
        if len(self.data) < self.k:
            heapq.heappush(self.data, elem)
        else:
            topk_small = self.data[0][0]
            target_score = elem[0]
            if target_score < topk_small:
                heapq.heapreplace(self.data, elem)
    def topk(self):
        return [heapq.heappop(self.data) for _ in range(len(self.data))]

def genetic_search(cfg, graph, vdata, bndata, logger, search_iters=20):
    leader_board = LeaderBoard(100)
    evolver = SearchEvolution(cfg, graph, vdata, bndata, logger=logger)
    population = evolver.build_population()
    for i in range(search_iters):
        population = evolver.evolve(population, leader_board, i)


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
    graph.use_multigpu()
    genetic_search(cfg, graph, loader['val'], loader['train'], logger)

if __name__ == '__main__':
    # main()
    
