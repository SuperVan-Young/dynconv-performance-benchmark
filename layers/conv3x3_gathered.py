import numpy as np
import tvm
from tvm import autotvm
import tvm.te as te
import tvm.testing
import torch
import torch.nn.functional as F

from .base import BaseLayerScheduler


@autotvm.template("dynconv/conv3x3_gathered")
def schedule_conv3x3_gathered(C, L, G, group):
    # dataflow
    input = te.placeholder((L, C, (G+2)*(G+2)), dtype="float32", name="input")
    weight = te.placeholder((C, C//group, 3, 3),
                            dtype="float32", name="weight")

    rc = te.reduce_axis((0, C//group), name="rc")
    ry = te.reduce_axis((0, 3), name="ry")
    rx = te.reduce_axis((0, 3), name="rx")

    def find_group(f): return f - (f % (C//group))

    def find_input(v, ry, rx):
        y, x = v // G, v % G
        y_, x_ = y + ry, x + rx
        v_ = y_ * (G+2) + x_
        return v_

    output = te.compute(
        (L, C, G*G),
        lambda n, f, x: te.sum(
            input[n, find_group(f)+rc, find_input(x, ry, rx)
                  ] * weight[f, rc, ry, rx],
            axis=[rc, ry, rx]),
        name="output"
    )

    s = te.create_schedule(output.op)

    # schedule
    cfg = autotvm.get_config()

    n, f, x = s[output].op.axis
    cfg.define_split("tile_n", n, num_outputs=2)  # no ni! don't need vn
    cfg.define_split("tile_f", f, num_outputs=3)  # no fi!
    cfg.define_split("tile_x", x, num_outputs=3)
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
    bn, tn = cfg["tile_n"].apply(s, output, n)
    bf, vf, tf = cfg["tile_f"].apply(s, output, f)
    vx, tx, xi = cfg["tile_x"].apply(s, output, x)

    s[output].bind(bn, te.thread_axis("blockIdx.y"))
    s[output].bind(bf, te.thread_axis("blockIdx.x"))
    s[output].bind(vf, te.thread_axis("vthread"))
    s[output].bind(vx, te.thread_axis("vthread"))
    s[output].bind(tn, te.thread_axis("threadIdx.z"))
    s[output].bind(tf, te.thread_axis("threadIdx.y"))
    s[output].bind(tx, te.thread_axis("threadIdx.x"))
    s[output].reorder(bn, bf, vf, vx, tn, tf, tx, xi)
    s[OL].compute_at(s[output], tx)

    # tile reduction axes
    n, f, x = s[OL].op.axis
    rc, ry, rx = s[OL].op.reduce_axis
    rco, rcm, rci = cfg["tile_rc"].apply(s, OL, rc)
    ryo, rym, ryi = cfg["tile_ry"].apply(s, OL, ry)
    rxo, rxm, rxi = cfg["tile_rx"].apply(s, OL, rx)
    s[OL].reorder(rco, ryo, rxo, rcm, rym, rxm, rci, ryi, rxi, n, f, x)
    s[AA].compute_at(s[OL], rxo)
    s[WW].compute_at(s[OL], rxo)
    s[AL].compute_at(s[OL], rxm)
    s[WL].compute_at(s[OL], rxm)

    # cooperative fetching
    n, f, x = s[AA].op.axis
    fused = s[AA].fuse(n, f, x)
    tz, fused = s[AA].split(fused, nparts=cfg["tile_n"].size[1])  # tn
    ty, fused = s[AA].split(fused, nparts=cfg["tile_f"].size[2])  # tf
    tx, fused = s[AA].split(fused, nparts=cfg["tile_x"].size[1])  # tx
    s[AA].bind(tz, te.thread_axis("threadIdx.z"))
    s[AA].bind(ty, te.thread_axis("threadIdx.y"))
    s[AA].bind(tx, te.thread_axis("threadIdx.x"))

    n, f, y, x = s[WW].op.axis
    fused = s[WW].fuse(n, f, y, x)
    tz, fused = s[WW].split(fused, nparts=cfg["tile_n"].size[1])  # tn
    ty, fused = s[WW].split(fused, nparts=cfg["tile_f"].size[2])  # tf
    tx, fused = s[WW].split(fused, nparts=cfg["tile_x"].size[1])  # tx
    s[WW].bind(tz, te.thread_axis("threadIdx.z"))
    s[WW].bind(ty, te.thread_axis("threadIdx.y"))
    s[WW].bind(tx, te.thread_axis("threadIdx.x"))

    return s, [input, weight, output]


class Conv3x3GatheredScheduler(BaseLayerScheduler):
    def __init__(self, channel, group, sparselen, granularity, save_path):
        super().__init__()
        assert channel % group == 0

        self.channel = channel
        self.group = group
        self.sparselen = sparselen
        self.granularity = granularity
        self.save_path = save_path
        
        self.rtol = 1e-4
        self.atol = 1

    def __repr__(self) -> str:
        return "Conv3x3GatheredScheduler"

    @property
    def _task_name(self):
        return "dynconv/conv3x3_gathered"

    @property
    def _task_args(self):
        return [self.channel, self.sparselen, self.granularity, self.group]

    @property
    def _schedule(self):
        return schedule_conv3x3_gathered(*self._task_args)

    def _generate_sample(self):
        L, C, G = self.sparselen, self.channel, self.granularity
        input = np.random.randn(L, C, (G+2)*(G+2)).astype("float32")
        weight = np.random.randn(C, C//self.group, 3, 3).astype("float32")
        return [input, weight]

    def _convert_sample(self, sample):
        L, C, G = self.sparselen, self.channel, self.granularity
        sample = [tvm.nd.array(_, self.device) for _ in sample]
        output = tvm.nd.empty((L, C, G*G), dtype="float32", device=self.device)
        sample.append(output)
        return sample

    def _run_tvm(self, inputs):
        inputs = self._convert_sample(inputs)
        self.func(*inputs)
        return inputs[-1].numpy()

    def _run_numpy(self, inputs):
        raise NotImplementedError

    def _run_pytorch(self, inputs):
        L, C, G = self.sparselen, self.channel, self.granularity
        input, weight = inputs
        input = torch.tensor(input.reshape(L, C, G+2, G+2),
                             dtype=torch.float32, device="cuda")
        weight = torch.tensor(weight, dtype=torch.float32, device="cuda")
        output = F.conv2d(input, weight, None, groups=self.group)
        output = output.cpu().numpy().reshape(L, C, -1)
        return output