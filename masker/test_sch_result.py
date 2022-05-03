import os
import sys
import numpy as np
import tvm
from tvm import autotvm
import tvm.te as te
import tvm.testing
from masker_scheduler import *

gpu_id=1
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
        
        granul_sizes = get_factors(width)[:1]
        for granul_size in granul_sizes:
            n_blocks = find_best_n_blocks(width//granul_size, density-0.01, density+0.01)
            # save_path=f"{save_dir}/maskconv_g{granul_size}_nb{n_blocks}.json"
            save_path=f"{save_dir}/maskconv_and_conv1.json"
            if not os.path.exists(save_path):
                continue
            args=[batch_size,channel,channel,width,height,granul_size]
            task = autotvm.task.create(
                task_name=task_name, args=args, target=target
            ) # Create a tuning task and initialize its search space
            with autotvm.apply_history_best(save_path):
                with tvm.target.Target(target):
                    s, arg_bufs = masker_and_conv1(*args)
                    func = tvm.build(s, arg_bufs)
                    code=tvm.lower(s,arg_bufs,simple_mode=True)
                    # print()
                    with open(f"{save_path}.cu",'w') as f:
                        f.write(func.imported_modules[0].get_source())
                        f.write(f"\n\n{code}")
                    
                    evaluator=func.time_evaluator(func.entry_name,device,eval_repeat)
                    input = np.random.randn(batch_size, channel, height, width).astype("float32")
                    weight = np.random.randn(channel+1, channel,1,1).astype("float32")
                    output = np.zeros([batch_size,channel+1,height,width]).astype("float32")
                    sample=[input, weight,output]
                    sample = [tvm.nd.array(_, device) for _ in sample]
                    t=evaluator(*sample).mean
                    print(t*1000)
                    with open(save_path+'.best','w') as f:
                        f.write(f"time {t} \n")
                        f.write(f"{tvm.lower(s, arg_bufs, simple_mode=True)}")
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
            print(f"autotune {save_path} complete!")