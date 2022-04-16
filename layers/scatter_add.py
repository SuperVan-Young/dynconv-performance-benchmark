import numpy as np
import scipy.sparse as sparse
import tvm
from tvm import autotvm
import tvm.te as te
import tvm.testing
import torch
import torch.nn.functional as F

from .base import BaseLayerScheduler


@autotvm.template("dynconv/scatter_add")
def schedule_scatter_add(C, H, W, L, G):

    # define dataflow
    input = te.placeholder((1, C, H, W), dtype="float32", name="input")
    gathered = te.placeholder((L, C, G*G), dtype="float32", name="gathered")
    # mask is incremented and interpolated
    mask = te.placeholder((1, 1, H, W), dtype="int32", name="mask")

    def g(y, x):
        y_, x_ = y % G, x % G
        return y_ * G + x_

    output = te.compute((1, C, H, W),
                        lambda n, f, y, x: tvm.tir.if_then_else(
        mask[0, 0, y, x] > 0,
        input[n, f, y, x] + gathered[mask[0, 0, y, x] - 1, f, g(y, x)],
        input[n, f, y, x]),
        name="output"
    )

    s = te.create_schedule(output.op)

    # scheduling
    cfg = autotvm.get_config()

    n, f, y, x = s[output].op.axis    
    cfg.define_split("tile_n", n, num_outputs=2)  # no tn
    cfg.define_split("tile_f", f, num_outputs=4)
    cfg.define_split("tile_y", y, num_outputs=3)
    cfg.define_split("tile_x", x, num_outputs=3)

    # caching
    OL = s.cache_write(output, "local")

    # tile and bind spatial axes
    bn, ni = cfg["tile_n"].apply(s, output, n)
    bf, vf, tf, fi = cfg["tile_f"].apply(s, output, f)
    vy, ty, yi = cfg["tile_y"].apply(s, output, y)
    vx, tx, xi = cfg["tile_x"].apply(s, output, x)

    s[output].bind(bn, te.thread_axis("blockIdx.y"))
    s[output].bind(bf, te.thread_axis("blockIdx.x"))
    s[output].bind(vf, te.thread_axis("vthread"))
    s[output].bind(vy, te.thread_axis("vthread"))
    s[output].bind(vx, te.thread_axis("vthread"))
    s[output].bind(tf, te.thread_axis("threadIdx.z"))
    s[output].bind(ty, te.thread_axis("threadIdx.y"))
    s[output].bind(tx, te.thread_axis("threadIdx.x"))

    s[output].reorder(bn, bf, vf, vy, vx, tf, ty, tx, ni, fi, yi, xi)
    s[OL].compute_at(s[output], tx)

    return s, [input, gathered, mask, output]


class ScatterAddScheduler(BaseLayerScheduler):
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
        return "ScatterAddScheduler"

    @property
    def _task_name(self):
        return "dynconv/scatter_add"

    @property
    def _task_args(self):
        return [self.channel, self.width, self.width, self.sparselen, self.granularity]

    @property
    def _schedule(self):
        return schedule_scatter_add(*self._task_args)

    def _generate_sample(self):
        H, W = self.width, self.width
        C = self.channel
        L = self.sparselen
        G = self.granularity

        input = np.random.randn(1, C, H, W).astype("float32")
        gathered = np.random.randn(L, C, G*G).astype("float32")
        perm = np.arange(0, (H//G)*(W//G))
        np.random.shuffle(perm)
        rows = (perm[:L] // (W//G)).astype("int32")
        cols = (perm[:L] % (W//G)).astype("int32")
        mask = sparse.coo_matrix(
            (np.arange(0, L), (rows, cols)), shape=(H//G, W//G),  dtype="int32").toarray()
        mask_full = np.zeros((1, 1, H, W)).astype("int32")
        for i in range(H):
            for j in range(W):
                mask_full[:, :, i:i+1, j:j+1] = mask[i//G, j//G]
        return [input, gathered, mask_full]

    def _convert_sample(self, sample):
        C, H, W = self.channel, self.width, self.width
        sample = [tvm.nd.array(_, self.device) for _ in sample]
        output = tvm.nd.empty((1, C, H, W), "float32", self.device)
        sample.append(output)
        return sample

    def _run_tvm(self, inputs):
        inputs = self._convert_sample(inputs)
        self.func(*inputs)
        return inputs[-1].numpy()

    def _run_numpy(self, inputs):
        L, C, G, H, W = self.sparselen, self.channel, self.granularity, self.width, self.width
        input, gathered, mask = inputs
        gathered = gathered.reshape(L, C, G, G)
        output = input.copy()
        for i in range(H):
            for j in range(W):
                index = mask[0, 0, i, j]
                if index > 0:
                    output[:, :, i:i+1, j:j+1] = output[:, :, i:i+1, j:j+1] +  gathered[index-1:index, :, i%G:i%G+1, j%G:j%G+1]
        return output

    def _run_pytorch(self, inputs):
        raise NotImplementedError
