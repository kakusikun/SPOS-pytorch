from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import sys
import math
from src.base_graph import BaseGraph
#TODO use backbone factory
from src.model.backbone.shufflenas import shufflenas_oneshot
from src.model.backbone.shufflenetv2nas import ShuffleNetv2Nas
import logging
logger = logging.getLogger("logger")