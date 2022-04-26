from .base import BaseBlockEvaluator
from functools import reduce
from layers import *
import os
from tqdm import tqdm

class TVMDynamicBlockEvaluator(BaseBlockEvaluator):
    """Offer more fine-grained control over the residual block in sparse mode.

    Conventional interfaces only work on dense operations, such as masker and
    conv1. We use new interfaces for gathered operations.

    Different from regnet's definitions, width means size of feature map here. 
    Since conv3x3_gathered cannot handle stride != 1,
    width stays consistent within the block

    """

    def __init__(self, channel, width, group, save_dir, n_trial=300):
        super().__init__()
        self.sparse_layers = {}

        self.channel = channel
        self.width = width
        self.group = group
        self.save_dir = save_dir
        self.n_trial = n_trial   # only support int now

    def setup(self):
        # Only set up dense part
        self.layers["maskconv1"] = ConvDenseScheduler(
            1, self.channel, self.width, 3, self.save_dir+"/maskconv1.json")
        self.layers["conv1"] = ConvDenseScheduler(
            self.channel, self.channel, self.width, 1, self.save_dir+"/conv1.json")
        self.layers["conv2"] = ConvDenseScheduler(
            self.channel, self.channel, self.width, 3, self.save_dir+"/conv2.json")
        for layer in self.layers.values():
            layer.n_trial = self.n_trial

    def run(self):
        raise NotImplementedError

    def setup_sparse(self, sparselen, granularity, test_gather=False, test_scatter=False, test_masker=False):
        assert sparselen * granularity * granularity < self.width ** 2
        assert self.width % granularity == 0

        sparse_layers = {}
        save_dir = self.save_dir + f"/l{sparselen}_m{granularity}"
        
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        sparse_layers["conv2"] = Conv3x3GatheredScheduler(
            self.channel, self.group, sparselen, granularity, save_dir+"/conv2.json"
        )
        sparse_layers["conv3"] = Conv1x1GatheredScheduler(
            self.channel, self.channel, sparselen, granularity, save_dir+"/conv3.json"
        )
        if test_gather:
            sparse_layers["gather"] = GatherScheduler(
                self.width, self.channel, sparselen, granularity, save_dir+"/gather.json"
            )
        if test_scatter:
            sparse_layers["scatter_add"] = ScatterAddScheduler(
                self.width, self.channel, sparselen, granularity, save_dir+"/scatter_add.json"
            )
        if test_masker:
            sparse_layers["maskconv2"] = ConvDenseScheduler(
                1, 1, self.width//granularity, 1, save_dir+"/maskconv2.json")
        
        for layer in sparse_layers.values():
            layer.n_trial = self.n_trial

        self.sparse_layers = sparse_layers

    def _select_layers(self, mode):
        layers = None
        if mode == "sparse":
            assert self.sparse_layers != {}, "run setup_sparse before autotuning!"
            layers = self.sparse_layers
        elif mode == "dense":
            assert self.layers != {}, "run setup before autotuning!"
            layers = self.layers
        else:
            raise NotImplementedError("invalid mode")
        return layers


    def autotune(self, mode="dense"):
        layers = self._select_layers(mode)

        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)
        bar = tqdm(layers.items())
        for name, layer in bar:
            bar.set_description(name)
            layer.autotune(refresh=True)
        print(f"All layers in {self} are autotuned.")

    def build(self, mode="dense"):
        layers = self._select_layers(mode)

        if not os.path.exists(self.save_dir):
            raise RuntimeError("Block not autotuned")
        for layer in layers.values():
            layer.build()

    def evaluate(self, mode="dense", verbose=False):
        layers = self._select_layers(mode)
        result = {k: v.evaluate() for k, v in layers.items()}
        if verbose:
            return result
        else:
            return reduce(lambda x, y: x+y, result.values())
    
    def __call__(self, mode="dense", sparselen=None, granularity=None, test_gather=False, test_scatter=False, test_masker=False):
        if mode == "dense":
            super().__call__()
        elif mode == "sparse":
            assert sparselen != None
            assert granularity != None
            self.setup_sparse(sparselen, granularity, test_gather, test_scatter, test_masker)
            self.autotune("sparse")
            self.build("sparse")
            return self.evaluate(mode="sparse", verbose=False)
        else:
            raise NotImplementedError