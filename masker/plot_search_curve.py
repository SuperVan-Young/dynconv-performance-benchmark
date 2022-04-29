import os
import sys
import numpy as np
import tvm
from tvm import autotvm
import tvm.te as te
import tvm.testing
from masker_scheduler import *
import matplotlib.pyplot as plt
import json
import math

gpu_id=3
os.environ['CUDA_VISIBLE_DEVICES'] = f'{gpu_id}'
refresh=0 # clear prev search
target="cuda"
device=tvm.device(target, 0)
task_name="dyconv/masker"

# widths, channels, group_width = regnet_parameters("008")
# regnet 008
widths = [56, 28, 14, 7]
channels = [64, 128, 288, 672]
group_width = 16
batch_size=1

test_densities = [0.2]

for density in test_densities:
    for i in range(4):
        width, channel = widths[i], channels[i]
        height=width
        n_trial = 300
        eval_repeat=100
        granul_groups=1
        save_dir = f"log/c{channel}_w{width}_g{group_width}"
        
        granul_sizes = get_factors(width)
        for granul_size in granul_sizes:
            n_blocks = find_best_n_blocks(width//granul_size, density-0.01, density+0.01)
            save_path=f"{save_dir}/maskconv_and_conv1.json"
            log2_latencies=[]
            last_latency=1
            with open(save_path) as f:
                for l in f.readlines():
                    r=json.loads(l)
                    latency=r['result'][0][0]
                    if latency>100:
                        latency=last_latency
                    # print(latency)
                    last_latency=latency
                    log2_latencies.append(math.log2(latency))
            maskconv_and_conv1_latency=2**min(log2_latencies)

            save_path=f"{save_dir}/maskconv_reduce_g{granul_size}_nb{n_blocks}.json"
            if not os.path.exists(save_path):
                continue
            log2_latencies=[]
            last_latency=1
            with open(save_path) as f:
                for l in f.readlines():
                    r=json.loads(l)
                    latency=r['result'][0][0]
                    if latency>100:
                        latency=last_latency
                    # print(latency)
                    last_latency=latency
                    log2_latencies.append(math.log2(latency))
            plt.plot(log2_latencies)
            plt.savefig(save_path+".jpg")
            plt.clf()
            print(save_path)
            maskconv_reduce_latency=2**min(log2_latencies)
            print(maskconv_and_conv1_latency,maskconv_reduce_latency,maskconv_and_conv1_latency+maskconv_reduce_latency)