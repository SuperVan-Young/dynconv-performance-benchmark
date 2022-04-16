import numpy as np
import tvm
from tvm import autotvm
import tvm.te as te
import tvm.testing

from .base import BaseLayerScheduler


@autotvm.template("dynconv/gather")
def schedule_gather(C, H, W, L, G):
    # dataflow
    input = te.placeholder((1, C, H, W), dtype="float32", name="input")
    rows = te.placeholder((L, ), dtype="int32", name="rows")
    cols = te.placeholder((L, ), dtype="int32", name="cols")

    input_padded = te.compute(
        (1, C, H+2, W+2),
        lambda n, c, y, x: tvm.tir.if_then_else(
            tvm.tir.all(y >= 1, y - 1 < H, x >= 1, x - 1 < W),
            input[n, c, y - 1, x - 1],
            tvm.tir.const(0.0)),
        name="input_padded")

    def vec2row(row, x): return row * G + x // (G + 2)
    def vec2col(col, x): return col * G + x % (G + 2)

    output = te.compute(
        (L, C, (G+2)*(G+2)),
        lambda b, c, x: input_padded[0, c, vec2row(
            rows[b], x), vec2col(cols[b], x)],
        name="output")

    s = te.create_schedule(output.op)

    # scheduling
    cfg = autotvm.get_config()

    n, f, x = s[output].op.axis
    cfg.define_split("tile_n", n, num_outputs=3)
    cfg.define_split("tile_f", f, num_outputs=4)
    cfg.define_split("tile_x", x, num_outputs=3)

    # caching
    s[input_padded].compute_inline()
    OL = s.cache_write(output, "local")

    # tile and bind spatial axes
    bn, tn, ni = cfg["tile_n"].apply(s, output, n)  # don't need vn
    bf, vf, tf, fi = cfg["tile_f"].apply(s, output, f)
    vx, tx, xi = cfg["tile_x"].apply(s, output, x)  # 1 thraed-block for a grid

    s[output].bind(bn, te.thread_axis("blockIdx.y"))
    s[output].bind(bf, te.thread_axis("blockIdx.x"))
    s[output].bind(vf, te.thread_axis("vthread"))
    s[output].bind(vx, te.thread_axis("vthread"))
    s[output].bind(tn, te.thread_axis("threadIdx.z"))
    s[output].bind(tf, te.thread_axis("threadIdx.y"))
    s[output].bind(tx, te.thread_axis("threadIdx.x"))

    s[output].reorder(bn, bf, vf, vx, tn, tf, tx, ni, fi, xi)

    s[OL].compute_at(s[output], tx)

    return s, [input, rows, cols, output]


class GatherScheduler(BaseLayerScheduler):
    def __init__(self, width, channel, sparselen, granularity, save_path):
        super().__init__()
        assert width % granularity == 0
        assert sparselen < width * width

        self.width = width
        self.channel = channel
        self.sparselen = sparselen
        self.granularity = granularity
        self.save_path = save_path

    def __repr__(self) -> str:
        return "GatherScheduler"

    @property
    def _task_name(self):
        return "dynconv/gather"

    @property
    def _task_args(self):
        return [self.channel, self.width, self.width, self.sparselen, self.granularity]

    @property
    def _schedule(self):
        return schedule_gather(*self._task_args)

    def _generate_sample(self):
        H, W = self.width, self.width
        C = self.channel
        L = self.sparselen
        G = self.granularity

        input = np.random.randn(1, C, H, W).astype("float32")
        perm = np.arange(0, (H//G)*(W//G))
        np.random.shuffle(perm)
        rows = (perm[:L] // (W//G)).astype("int32")
        cols = (perm[:L] % (W//G)).astype("int32")

        return [input, rows, cols]

    def _convert_sample(self, sample):
        L, C, G = self.sparselen, self.channel, self.granularity
        sample = [tvm.nd.array(_, self.device) for _ in sample]
        output = tvm.nd.empty((L, C, (G+2)*(G+2)), "float32", self.device)
        sample.append(output)
        return sample

    def _run_tvm(self, inputs):
        inputs = self._convert_sample(inputs)
        self.func(*inputs)
        return inputs[-1].numpy()

    def _run_numpy(self, inputs):
        C = self.channel
        L = self.sparselen
        G = self.granularity

        input, rows, cols = inputs
        input_pad = np.pad(input, ((0, 0), (0, 0), (1, 1),
                           (1, 1)), mode="constant", constant_values=0)
        output = np.zeros((L, C, G+2, G+2), dtype="float32")

        for i in range(L):
            rg = rows[i] * G
            rc = cols[i] * G
            output[i:i+1, :, :, :] = input_pad[:, :, rg:rg+G+2, rc:rc+G+2]
        return output.reshape((L, C, -1))

    def _run_pytorch(self, inputs):
        raise NotImplementedError
