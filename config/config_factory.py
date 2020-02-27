import os
import os.path as osp
import sys
import datetime
import shutil

from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Convention about Training / Test specific parameters
# -----------------------------------------------------------------------------
# Whenever an argument can be either used for training or for testing, the
# corresponding name will be post-fixed by a _TRAIN for a training parameter,
# or _TEST for a test-specific parameter.
# For example, the number of images during training will be
# IMAGES_PER_BATCH_TRAIN, while the number of images for testing will be
# IMAGES_PER_BATCH_TEST

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------
_C = CN()
_C.IO = True
_C.SEED = 42
_C.EXPERIMENT = ""
_C.TASK = ""
_C.ENGINE = ""
_C.GRAPH = ""
_C.TRAINER = ""
_C.TRAIN_TRANSFORM = ""
_C.TEST_TRANSFORM = ""
_C.NUM_WORKERS = 16

# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
_C.OUTPUT_DIR = ""
_C.RESUME = ""
_C.EVALUATE = False
_C.SAVE = True

# ---------------------------------------------------------------------------- #
# Model options
# ---------------------------------------------------------------------------- #

_C.MODEL = CN()
_C.MODEL.GPU = []
_C.MODEL.BACKBONE = ""
_C.MODEL.HEAD = ""
_C.MODEL.STRIDE = 1
_C.MODEL.EVALUATE_FREQ = 0

# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
_C.INPUT = CN()
_C.INPUT.PAD = 0
_C.INPUT.RESIZE = (0, 0)
_C.INPUT.CROP_SIZE = (0, 0)
# Size of the image during training
_C.INPUT.TRAIN_BS = 32
# Size of the image during test
_C.INPUT.TEST_BS = 32
# Values to be used for image normalization
_C.INPUT.MEAN = []
# Values to be used for image normalization
_C.INPUT.STD = []
# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DB = CN()
_C.DB.PATH = ""
_C.DB.DATA = ""
_C.DB.DATA_FORMAT = ""
_C.DB.USE_TRAIN = True
_C.DB.USE_TEST = True
_C.DB.TRAIN_TRANSFORM = ""
_C.DB.TEST_TRANSFORM = ""
_C.DB.LOADER = ""
_C.DB.NUM_CLASSES = 0

# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()
_C.SOLVER.OPTIMIZER = "SGD"
_C.SOLVER.START_EPOCH = 1
_C.SOLVER.MAX_EPOCHS = 50
_C.SOLVER.BASE_LR = 0.001
_C.SOLVER.BIAS_LR_FACTOR = 1.0
_C.SOLVER.CUSTOM = [] 
_C.SOLVER.MOMENTUM = 0.9
_C.SOLVER.WEIGHT_DECAY = 0.0005
_C.SOLVER.WEIGHT_DECAY_BIAS_FACTOR = 1.0
_C.SOLVER.NESTEROV = False
_C.SOLVER.LR_POLICY = ""

# for plateau
_C.SOLVER.MIN_LR = 0.0
_C.SOLVER.PLATEAU_SIZE = 10.0
_C.SOLVER.GAMMA = 0.1

# for cosine
_C.SOLVER.NUM_RESTART = 4
_C.SOLVER.T_MULT = 2
_C.SOLVER.T_0 = 10
_C.SOLVER.ITERATIONS_PER_EPOCH = -1
# _C.SOLVER. = 1

# for warmup
_C.SOLVER.WARMUP = False 
_C.SOLVER.WARMUP_FACTOR = 1.0 / 3
_C.SOLVER.WARMUP_SIZE = 10.0
_C.SOLVER.MODEL_FREEZE_PEROID = 0

# ---------------------------------------------------------------------------- #
# SPOS
# ---------------------------------------------------------------------------- #
_C.SPOS = CN()
_C.SPOS.USE_SE = True
_C.SPOS.LAST_CONV_AFTER_POOLING = True
_C.SPOS.CHANNELS_LAYOUT = "OneShot"




def build_output(cfg, config_file=""):
    time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if cfg.EVALUATE:
        cfg.OUTPUT_DIR = osp.join("evaluation", cfg.TASK, cfg.EXPERIMENT, time)
    else:
        cfg.OUTPUT_DIR = osp.join(os.getcwd(), "result", cfg.TASK, cfg.EXPERIMENT, time)
    if cfg.OUTPUT_DIR and not osp.exists(cfg.OUTPUT_DIR):
        os.makedirs(cfg.OUTPUT_DIR)
        if config_file != "":
            shutil.copy(config_file, osp.join(cfg.OUTPUT_DIR, config_file.split("/")[-1]))
