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
    return parser.parse_args()

if __name__ == "__main__":
    args = parseargs()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.g
    
    # test dense model's speed
    widths, channels, group_width = regnet_parameters("008")

    ss = [0.2, 0.4, 0.6, 0.8]
    s = ss[int(args.g)]

    for i in range(4):
        width, channel = widths[i], channels[i]
        n_trial = 300  # debug
        save_path = f"log/c{channel}_w{width}_g{group_width}"
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        bs = TVMDynamicBlockEvaluator(channel, width, group_width, save_path, n_trial)

        # sparse
        gg = get_factors(width)
        for g in gg:
            sl = find_best_sparselen(width//g, s-0.01, s+0.01)
            # bs.setup_sparse(sl, g, True, True, True, True)
            bs.setup_sparse(sl, g, False, False, False, False, True, False)
            bs.autotune("sparse")

    print("autotune complete!")

