import numpy as np
import tvm
from tvm import autotvm
import tvm.te as te
import tvm.testing
import torch
import torch.nn.functional as F

from .base import BaseLayerScheduler


@autotvm.template("dynconv/conv_dense")
def schedule_conv_dense(Cout, Cin, H, W, KH, KW):
    # dataflow
    input = te.placeholder((1, Cin, H, W), dtype="float32", name="input")
    weight = te.placeholder((Cout, Cin, KH, KW), dtype="float32", name="weight")

    rc = te.reduce_axis((0, Cin), "rc")
    ry = te.reduce_axis((0, KH), "ry")
    rx = te.reduce_axis((0, KW), "rx")
    output = te.compute(
        (1, Cout, H-KH+1, W-KW+1),
        lambda n, f, y, x: te.sum(
            input[n, rc, y + ry, x + rx] * weight[f, rc, ry, rx],
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
    for load in [AA, WW]:
        n, f, y, x = s[load].op.axis
        fused = s[load].fuse(n, f, y, x)
        tz, fused = s[load].split(fused, nparts=cfg["tile_f"].size[2])  # tf
        ty, fused = s[load].split(fused, nparts=cfg["tile_y"].size[2])  # ty
        tx, fused = s[load].split(fused, nparts=cfg["tile_x"].size[2])  # tx
        s[load].bind(tz, te.thread_axis("threadIdx.z"))
        s[load].bind(ty, te.thread_axis("threadIdx.y"))
        s[load].bind(tx, te.thread_axis("threadIdx.x"))

    return s, [input, weight, output]


class ConvDenseScheduler(BaseLayerScheduler):
    def __init__(self, channel_out, channel_in, width, kernel, save_path):
        super().__init__()

        self.channel_out = channel_out
        self.channel_in = channel_in
        self.width = width
        self.kernel = kernel
        self.save_path = save_path

    def __repr__(self) -> str:
        return "ConvDenseScheduler"

    @property
    def _task_name(self):
        return "dynconv/conv_dense"

    @property
    def _task_args(self):
        return [self.channel_out, self.channel_in, self.width, self.width, self.kernel, self.kernel]

    @property
    def _schedule(self):
        return schedule_conv_dense(*self._task_args)

    def _generate_sample(self):
        H, W = self.width, self.width
        KH, KW = self.kernel, self.kernel
        Cout, Cin = self.channel_out, self.channel_in
        input = np.random.randn(1, Cin, H, W).astype("float32")
        weight = np.random.randn(Cout, Cin, KH, KW).astype("float32")
        return [input, weight]

    def _convert_sample(self, sample):
        H, W = self.width, self.width
        KH, KW = self.kernel, self.kernel
        Cout, Cin = self.channel_out, self.channel_in
        sample = [tvm.nd.array(_, self.device) for _ in sample]
        output = tvm.nd.empty(
            (1, Cout, H-KH+1, W-KW+1), dtype="float32", device=self.device)
        sample.append(output)
        return sample

    def _run_tvm(self, inputs):
        inputs = self._convert_sample(inputs)
        self.func(*inputs)
        return inputs[-1].numpy()

    def _run_numpy(self, inputs):
        raise NotImplementedError

    def _run_pytorch(self, inputs):
        input, weight = inputs
        input = torch.tensor(input, dtype=torch.float32, device="cuda")
        weight = torch.tensor(weight, dtype=torch.float32, device="cuda")
        output = F.conv2d(input, weight)
        output = output.cpu().numpy()
        return output
