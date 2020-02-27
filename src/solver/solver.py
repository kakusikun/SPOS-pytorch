import sys
import torch
import math
import numpy as np 
import src.solver.optimizers as opts
import src.solver.lr_schedulers as lr_schedulers
import logging
logger = logging.getLogger("logger")

class Solver(): 
    def __init__(self, cfg, params):  
        self.lr = cfg.SOLVER.BASE_LR
        self.bias_lr_factor = cfg.SOLVER.BIAS_LR_FACTOR
        self.momentum = cfg.SOLVER.MOMENTUM
        self.wd = cfg.SOLVER.WEIGHT_DECAY
        self.wd_factor = cfg.SOLVER.WEIGHT_DECAY_BIAS_FACTOR
        self.lr_policy = cfg.SOLVER.LR_POLICY
        self.opt_name = cfg.SOLVER.OPTIMIZER

        # cosine annealing
        self.T_0 = cfg.SOLVER.T_0
        self.T_mult = cfg.SOLVER.T_MULT
        self.num_iter_per_epoch = cfg.SOLVER.ITERATIONS_PER_EPOCH

        # warmup
        self.warmup = cfg.SOLVER.WARMUP
        self.warmup_factor = cfg.SOLVER.WARMUP_FACTOR
        self.warmup_iters = cfg.SOLVER.WARMUP_SIZE * self.num_iter_per_epoch

        # plateau
        self.gamma = cfg.SOLVER.GAMMA
        self.patience = cfg.SOLVER.PLATEAU_SIZE * self.num_iter_per_epoch 
        self.monitor_lr = 0.0
            
        self._model_analysis(params, custom=cfg.SOLVER.CUSTOM)

        if self.opt_name == 'SGD':
            self.opt = torch.optim.SGD(self.params, momentum=self.momentum, nesterov=cfg.SOLVER.NESTEROV)
        elif self.opt_name == 'Adam':
            self.opt = torch.optim.Adam(self.params)
        elif self.opt_name == 'AdamW':
            self.opt = torch.optim.AdamW(self.params)
        elif self.opt_name == 'SGDW':
            self.opt = opts.SGDW(self.params, momentum=self.momentum, nesterov=cfg.SOLVER.NESTEROV)

        if not self.warmup:
            self.warmup_iters = 0

        if self.lr_policy == "plateau":
            self.scheduler = lr_schedulers.WarmupReduceLROnPlateau(
                optimizer=self.opt, 
                mode="min",
                gamma=self.gamma,
                patience=self.patience,
                warmup_factor=1.0/3,#self.warmup_factor,
                warmup_iters=self.warmup_iters,
            )
        elif self.lr_policy == "cosine":
            self.scheduler = lr_schedulers.WarmupCosineLR(
                optimizer=self.opt,
                num_iter_per_epoch=self.num_iter_per_epoch,
                warmup_factor=1.0/3,
                warmup_iters=self.warmup_iters,
                anneal_mult=self.T_mult,
                anneal_period=self.T_0,
            )
        elif self.lr_policy == "none":
            self.scheduler = None
            logger.info("LR policy is not used")
        else:
            logger.info("LR policy is not specified")
            sys.exit(1)    

    def _model_analysis(self, params, custom=[]):
        self.params = []
        # self.params = [{"params": params, "lr": self.lr, "weight_decay": self.wd}]
        num_params = 0.0
        
        for layer, p in params:
            #  try:
            if not p.requires_grad:
                continue
            lr = self.lr
            wd = self.wd
            if "bias" in layer:
                lr = self.lr * self.bias_lr_factor
                wd = self.wd * self.wd_factor    
            for name, target, value in custom:
                if name in layer:
                    if target == 'lr':
                        lr = value
                    elif target == 'wd':
                        wd = value
                    else:
                        logger.info("Unsupported optimizer parameter: {}".format(target))

            self.params += [{"params": p, "lr": lr, "weight_decay": wd}]
            num_params += p.numel()
        
        logger.info("Trainable parameters: {:.2f}M".format(num_params / 1000000.0))
    
    def lr_adjust(self, metrics, iters):
        if self.scheduler is not None:
            self.scheduler.step(metrics, iters)   
            self.monitor_lr = self.scheduler.monitor_lrs[0]
        else:
            self.monitor_lr = self.lr
    
    def zero_grad(self):        
        self.opt.zero_grad()

    def step(self):
        self.opt.step()





        

