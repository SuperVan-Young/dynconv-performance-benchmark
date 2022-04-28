import os
import sys
import numpy as np
import tvm
from tvm import autotvm
import tvm.te as te
import tvm.testing
import time

if not os.path.exists("log"):
    os.mkdir("log")

os.environ['CUDA_VISIBLE_DEVICES'] = '3'
refresh=True # clear prev search
target="cuda"
task_name="dyconv/masker"
max_time=(3600*16)/(16*4)
# max_time=10

# widths, channels, group_width = regnet_parameters("008")
# regnet 008
widths = [56, 28, 14, 7]
channels = [64, 128, 288, 672]
group_width = 16
batch_size=1


def get_factors(num):
    factors = []
    if num == 7:
        factors = [1]
    elif num == 14:
        factors = [1, 2, 7]
    elif num == 28:
        factors = [1, 2, 4, 7, 14]
    elif num == 56:
        factors = [1, 2, 4, 7, 8, 14, 28]
    return factors

def find_best_n_blocks(width, slow, shigh):
    """Find best sparselen given sparsity lower limit and higher limit
    """
    def factor(num):
        ls = [i for i in range(1, num+1) if num % i == 0]
        return len(ls)

    sparselens = range(int(width*width*slow), int(width*width*shigh)+1)
    factors = [factor(i) for i in sparselens]
    idx = factors.index(max(factors))
    # import pdb;pdb.set_trace()
    return max(sparselens[idx], 1) # minimum 1

@autotvm.template(task_name)
def masker_conv(batch_size,Cout, Cin, W, H, granul_size):
    # dataflow
    # print("batch_size,Cout, Cin, W, H, granul_size",batch_size,Cout, Cin, W, H, granul_size)
    input = te.placeholder((batch_size, Cin, H, W), dtype="float32", name="input")
    weight = te.placeholder((Cout, Cin), dtype="float32", name="weight")

    rc = te.reduce_axis((0, Cin), "rc")
    ry = te.reduce_axis((0, granul_size), "ry")
    rx = te.reduce_axis((0, granul_size), "rx")
    output = te.compute(
        (batch_size, Cout, H//granul_size, W//granul_size),
        lambda n, f, y, x: te.sum(
            input[n, rc, y*granul_size + ry, x*granul_size + rx] * weight[f, rc],
            axis=[rc, ry, rx]
        ),
        name="output"
    )

    s = te.create_schedule(output.op)

    # schedule
    cfg = autotvm.get_config()
    n, f, y, x = s[output].op.axis
    # no tile on n
    cfg.define_split("tile_f", f, num_outputs=4)  # with fi
    cfg.define_split("tile_y", y, num_outputs=4)  # by relieve memory ref
    cfg.define_split("tile_x", x, num_outputs=4)  # bx relieve memory ref
    cfg.define_split("tile_rc", rc, num_outputs=3)
    cfg.define_split("tile_ry", ry, num_outputs=3)
    cfg.define_split("tile_rx", rx, num_outputs=3)

    # caching
    OL = s.cache_write(output, "local")

    AA = s.cache_read(input, "shared", [OL])
    WW = s.cache_read(weight, "shared", [OL])
    AL = s.cache_read(AA, "local", [OL])
    WL = s.cache_read(WW, "local", [OL])

    # tile and bind spatial axes
    bf, vf, tf, fi = cfg["tile_f"].apply(s, output, f)
    by, vy, ty, yi = cfg["tile_y"].apply(s, output, y)
    bx, vx, tx, xi = cfg["tile_x"].apply(s, output, x)

    s[output].bind(bf, te.thread_axis("blockIdx.z"))
    s[output].bind(by, te.thread_axis("blockIdx.y"))
    s[output].bind(bx, te.thread_axis("blockIdx.x"))
    s[output].bind(vf, te.thread_axis("vthread"))
    s[output].bind(vy, te.thread_axis("vthread"))
    s[output].bind(vx, te.thread_axis("vthread"))
    s[output].bind(tf, te.thread_axis("threadIdx.z"))
    s[output].bind(ty, te.thread_axis("threadIdx.y"))
    s[output].bind(tx, te.thread_axis("threadIdx.x"))
    s[output].reorder(n, bf, by, bx, vf, vy, vx, tf, ty, tx, fi, yi, xi)
    s[OL].compute_at(s[output], tx)

    # tile reduction axes
    n, f, y, x = s[OL].op.axis
    rc, ry, rx = s[OL].op.reduce_axis
    rco, rcm, rci = cfg["tile_rc"].apply(s, OL, rc)
    ryo, rym, ryi = cfg["tile_ry"].apply(s, OL, ry)
    rxo, rxm, rxi = cfg["tile_rx"].apply(s, OL, rx)
    s[OL].reorder(rco, ryo, rxo, rcm, rym, rxm, rci, ryi, rxi, n, f, y, x)
    s[AA].compute_at(s[OL], rxo)
    s[WW].compute_at(s[OL], rxo)
    s[AL].compute_at(s[OL], rxm)
    s[WL].compute_at(s[OL], rxm)

    # cooperative fetching
    for load in [AA, ]:
        n, f, y, x = s[load].op.axis
        fused = s[load].fuse(n, f, y, x)
        tz, fused = s[load].split(fused, nparts=cfg["tile_f"].size[2])  # tf
        ty, fused = s[load].split(fused, nparts=cfg["tile_y"].size[2])  # ty
        tx, fused = s[load].split(fused, nparts=cfg["tile_x"].size[2])  # tx
        s[load].bind(tz, te.thread_axis("threadIdx.z"))
        s[load].bind(ty, te.thread_axis("threadIdx.y"))
        s[load].bind(tx, te.thread_axis("threadIdx.x"))
    
    # cooperative fetching
    for load in [ WW]:
        n, f = s[load].op.axis
        fused = s[load].fuse(n, f)
        tz, fused = s[load].split(fused, nparts=cfg["tile_f"].size[2])  # tf
        s[load].bind(tz, te.thread_axis("threadIdx.z"))

    return s, [input, weight, output]

class MyXGBTuner(autotvm.tuner.XGBTuner):
    def __init__(self, task, plan_size=64, feature_type="itervar", loss_type="rank", num_threads=None, optimizer="sa", diversity_filter_ratio=None, log_interval=50):
        super().__init__(task, plan_size, feature_type, loss_type, num_threads, optimizer, diversity_filter_ratio, log_interval)
        self.st_time=time.time()

    def has_next(self):
        if time.time()-self.st_time>max_time:
            print("Time out, break autotuner")
            return False
        return len(self.visited) < len(self.space)

if __name__=='__main__':

    test_densities = [0.2, 0.4, 0.6, 0.8]

    for density in test_densities[0:1]:
        for i in range(4):
            width, channel = widths[i], channels[i]
            height=width
            n_trial = 600
            granul_groups=1
            save_dir = f"log/c{channel}_w{width}_g{group_width}"
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            
            granul_sizes = get_factors(width)
            for granul_size in granul_sizes:
                n_blocks = find_best_n_blocks(width//granul_size, density-0.01, density+0.01)
                save_path=f"{save_dir}/maskconv_g{granul_size}_nb{n_blocks}.json"
                if os.path.exists(save_path) and refresh:
                    os.remove(save_path)
                args=[batch_size,granul_groups,channel,width,height,granul_size]
                task = autotvm.task.create(
                    task_name=task_name, args=args, target=target
                ) # Create a tuning task and initialize its search space
                measure_option = autotvm.measure_option(
                    builder=autotvm.LocalBuilder(),
                    runner=autotvm.LocalRunner(repeat=3, min_repeat_ms=150, timeout=4),
                )
                tuner = MyXGBTuner(task)
                tuner.tune(
                    n_trial=n_trial,
                    measure_option=measure_option,
                    early_stopping=80,
                    callbacks=[autotvm.callback.log_to_file(save_path)],
                )
                print(f"autotune {save_path} complete!")