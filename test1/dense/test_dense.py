import os
import sys
sys.path.append("../..")
sys.path.append("..")
sys.path.append(".")
import argparse

from regnet_param import *
from layers import *
from blocks import *
from utils import *

if not os.path.exists("log"):
    os.mkdir("log")

def parseargs():
    """Multi-gpu version
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", type=str)  # which gpu to use
    parser.add_argument("-n", type=str)  # which network to run
    return parser.parse_args()

if __name__ == "__main__":
    args = parseargs()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.g
    
    # test dense model's speed
    num = args.n
    widths, channels, group = regnet_parameters(num)

    for i in range(4):
        width, channel = widths[i], channels[i]
        n_trial = 10  # debug
        bs = TVMDynamicBlockEvaluator(channel, width, group, f"log/c{channel}_w{width}_g{group}", n_trial)
        bs()
