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
    widths, channels, group_width = regnet_parameters(num)

    for i in range(4):
        width, channel = widths[i], channels[i]
        n_trial = 300  # debug
        bs = TVMDynamicBlockEvaluator(channel, width, group_width, f"log/c{channel}_w{width}_g{group_width}", n_trial)
        # bs()

        # only evaluate add
        # bs.setup()
        # layer = bs.layers["add"]
        # layer.autotune(refresh="True")
        
        # only evaluate conv2
        bs.setup()
        layer = bs.layers["conv2"]
        layer.autotune(refresh="True")
    print("test dense complete!")