import numpy as np
import tvm
from tvm import autotvm
import tvm.te as te
import tvm.testing
import torch
import torch.nn.functional as F

from .base import BaseLayerScheduler


@autotvm.template("dynconv/add")
def schedule_add(C, H, W):
    # dataflow
    a = te.placeholder((1, C, H, W), dtype="float32", name="a")
    b = te.placeholder((1, C, H, W), dtype="float32", name="b")

    output = te.compute(
        (1, C, H, W),
        lambda n, f, y, x: a[n, f, y, x] + b[n, f, y, x],
        name="output"
    )

    s = te.create_schedule(output.op)

    # schedule
    cfg = autotvm.get_config()
    n, f, y, x = s[output].op.axis
    cfg.define_split("tile_f", f, num_outputs=4)  # with fi
    cfg.define_split("tile_y", y, num_outputs=4)  # by relieve memory ref
    cfg.define_split("tile_x", x, num_outputs=4)  # bx relieve memory ref

    OL = s.cache_write(output, "local")

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

    return s, [a, b, output]


class AddScheduler(BaseLayerScheduler):
    def __init__(self, channel, width, save_path):
        super().__init__()

        self.channel = channel
        self.width = width
        self.save_path = save_path

    def __repr__(self) -> str:
        return "AddScheduler"

    @property
    def _task_name(self):
        return "dynconv/add"

    @property
    def _task_args(self):
        return [self.channel, self.width, self.width]

    @property
    def _schedule(self):
        return schedule_add(*self._task_args)

    def _generate_sample(self):
        H, W = self.width, self.width
        C = self.channel
        a = np.random.randn(1, C, H, W).astype("float32")
        b = np.random.randn(1, C, H, W).astype("float32")
        return [a, b]

    def _convert_sample(self, sample):
        H, W = self.width, self.width
        C = self.channel
        sample = [tvm.nd.array(_, self.device) for _ in sample]
        output = tvm.nd.empty(
            (1, C, H, W), dtype="float32", device=self.device)
        sample.append(output)
        return sample

    def _run_numpy(self, inputs):
        a, b = inputs
        return a + b

    def _run_pytorch(self, inputs):
        return NotImplementedError
    
    def _run_tvm(self, inputs):
        inputs = self._convert_sample(inputs)
        self.func(*inputs)
        return inputs[-1].numpy()
