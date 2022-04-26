import sys
sys.path.append("..")
sys.path.append(".")
import os
import json
import argparse

from layers import *
from blocks import *
from utils import *

from stable import *

DEBUG = 0

import numpy as np

if not os.path.exists("log_dense"):
    os.mkdir("log_dense")

def parseargs():
    """Multi-gpu version
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", type=str)
    parser.add_argument("-n", type=str)
    return parser.parse_args()

# Regnet parameters
# size of featuremap, after downsampling
widths = [56, 28, 14, 7] if not DEBUG else [56]

def regnet_parameters(num):
    channels = None
    group = None

    if num == "002":
        channels = [24, 56, 152, 368]
        group = 8
    elif num == "004":
        channels = [32, 64, 160, 384]
        group = 16
    elif num == "006":
        channels = [48, 96, 240, 528]
        group = 24
    elif num == "008":
        channels = [64, 128, 288, 672]
        group = 16
    else:
        raise NotImplementedError

    return channels, group if not DEBUG else ([64], 16)

def get_factors(num):
    factors = []
    if num == 7:
        factors = [1, 7]
    elif num == 14:
        factors = [1, 2, 7]
    elif num == 28:
        factors = [1, 2, 4, 7, 14]
    elif num == 56:
        factors = [1, 2, 4, 7, 8, 14, 28]
    return factors if not DEBUG else [1]


if __name__ == "__main__":
    args = parseargs()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.g
    
    # test dense model's speed
    num = args.n
    channels, group = regnet_parameters(num)

    # dense
    for i, width in enumerate(widths):
        channel = channels[i]
        n_trial = 300 if not DEBUG else 10  # no refreshing, so 4 processes are cooperating
        bs = TVMDynamicBlockEvaluator(channel, width, group, f"log_dense/c{channel}_w{width}_g{group}", n_trial=n_trial)
        print(bs())
    
    # 008 sparse
    ss = [0.2, 0.4, 0.6, 0.8]
    s = ss[int(args.g)]

    channels, group = regnet_parameters("008")
    for i, width in enumerate(widths):
        channel = channels[i]
        n_trial = 300 if not DEBUG else 10  # no refreshing, so 4 processes are cooperating
        bs = TVMDynamicBlockEvaluator(channel, width, group, f"log_dense/c{channel}_w{width}_g{group}", n_trial=n_trial)

        factors = get_factors(width)
        for g in factors:
            sl = int(s * ((width // g) ** 2))
            sl = (sl // 8) * 8  # better for scheduling
            print(bs("sparse", sl, g, True, True, True))