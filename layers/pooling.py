import numpy as np
import tvm
from tvm import autotvm
import tvm.te as te
import tvm.testing
import torch
import torch.nn.functional as F

from .base import BaseLayerScheduler


@autotvm.template("dynconv/pooling")
def schedule_pooling(C, H, W, G):
    """This only works for H=W and H%G==0
    """
    # dataflow
    input = te.placeholder((1, C, H, W), dtype="float32", name="input")

    ry = te.reduce_axis((0, G), "ry")
    rx = te.reduce_axis((0, G), "rx")
    output = te.compute(
        (1, C, H//G, W//G),
        lambda n, f, y, x: te.sum(
            input[n, f, y*G+ry, x*G+rx], axis=[ry, rx]
        ),
        name="output",
    ) # actually not average, but it's fine

    s = te.create_schedule(output.op)

    # schedule
    cfg = autotvm.get_config()
    n, f, y, x = s[output].op.axis
    cfg.define_split("tile_f", f, num_outputs=4)  # with fi
    cfg.define_split("tile_y", y, num_outputs=4)  # by relieve memory ref
    cfg.define_split("tile_x", x, num_outputs=4)  # bx relieve memory ref
    cfg.define_split("tile_ry", ry, num_outputs=3)
    cfg.define_split("tile_rx", rx, num_outputs=3)

    # caching
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

    return s, [input, output]


class PoolingScheduler(BaseLayerScheduler):
    """Adaptive Average Pooling 2d in masker

    stride = floor(input_size / output_size) = granularity
    kernel_size = input_size - (output_size - 1) * stride = granularity
    We only schedule this particular scenario

    Not on average, just summing a grid; but it doesn't matter.
    """

    def __init__(self, channel, width, granularity, save_path):
        super().__init__()
        assert width % granularity == 0

        self.channel = channel
        self.width = width
        self.granularity = granularity
        self.save_path = save_path

    def __repr__(self) -> str:
        return "PoolingScheduler"

    @property
    def _task_name(self):
        return "dynconv/pooling"

    @property
    def _task_args(self):
        return [self.channel, self.width, self.width, self.granularity]

    @property
    def _schedule(self):
        return schedule_pooling(*self._task_args)

    def _generate_sample(self):
        H, W = self.width, self.width
        C = self.channel
        G = self.granularity
        input = np.random.randn(1, C, H, W).astype("float32")
        return [input]

    def _convert_sample(self, sample):
        H, W = self.width, self.width
        C = self.channel
        G = self.granularity
        sample = [tvm.nd.array(_, self.device) for _ in sample]
        output = tvm.nd.empty(
            (1, C, H//G, W//G), dtype="float32", device=self.device)
        sample.append(output)
        return sample

    def _run_tvm(self, inputs):
        inputs = self._convert_sample(inputs)
        self.func(*inputs)
        return inputs[-1].numpy()

    def _run_numpy(self, inputs):
        raise NotImplementedError

    def _run_pytorch(self, inputs):
        [input, ] = inputs
        input = torch.tensor(input, dtype=torch.float32, device="cuda")
        G = self.granularity
        output = F.adaptive_avg_pool2d(input, self.width//self.granularity) * G * G
        output = output.cpu().numpy()
        return output
