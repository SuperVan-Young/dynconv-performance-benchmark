import os
import sys
import numpy as np
import tvm
from tvm import autotvm
import tvm.te as te
import tvm.testing
from masker_scheduler import *

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
            save_path=f"{save_dir}/maskconv_g{granul_size}_nb{n_blocks}.json"
            if not os.path.exists(save_path):
                continue
            args=[batch_size,granul_groups,channel,width,height,granul_size]
            task = autotvm.task.create(
                task_name=task_name, args=args, target=target
            ) # Create a tuning task and initialize its search space
            with open(save_path+'.best','r') as f:
                l=f.readline().split(" ")
                ms=float(l[1])*1000
                print(",".join([str(_) for _ in [channel,width,group_width,granul_size,density,ms]]))
                # f.write(task.print_best(save_path))
                        
            # measure_option = autotvm.measure_option(
            #     builder=autotvm.LocalBuilder(),
            #     runner=autotvm.LocalRunner(repeat=3, min_repeat_ms=150, timeout=4),
            # )
            # tuner = autotvm.tuner.XGBTuner(task)
            # tuner.tune(
            #     n_trial=n_trial,
            #     measure_option=measure_option,
            #     callbacks=[autotvm.callback.log_to_file(save_path)],
            # )
            # print(f"autotune {save_path} complete!")