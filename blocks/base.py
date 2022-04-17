from abc import abstractmethod
from functools import reduce
from layers import *
import os
from tqdm import tqdm


class BaseBlockEvaluator():
    """Base class for evaluating residual block

    Example:
    setup -> autotune -> build -> evaluate/run
    """

    def __init__(self):
        self.layers = {}

    @abstractmethod
    def __repr__(self) -> str:
        return "BaseBlockEvaluator"

    @abstractmethod
    def setup(self):
        """setup layer schedulers according to hyperparameters
        """
        pass

    @abstractmethod
    def run(self, inputs):
        """Actually run the block on the given inputs

        Args:
            inputs (list): list of block inputs

        Returns:
            the output of the block, whose type depends on the implementation
        """
        return None

    def autotune(self):
        """Autotune all layers
        """
        assert self.layers != {}, "run setup before autotuning!"

        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)
        bar = tqdm(self.layers.items())
        for name, layer in bar:
            bar.set_description(name)
            layer.autotune()
        print(f"All layers in {self} are autotuned.")

    def build(self):
        """Build all layers runtime function

        Raises:
            RuntimeError: block not autotuned
        """
        if not os.path.exists(self.save_dir):
            raise RuntimeError("Block not autotuned")
        for layer in self.layers.values():
            layer.build()

    def evaluate(self, verbose=False):
        """Evaluate all layers inside the block

        Args:
            verbose (bool, optional): return a overall performance or 
                performance in details. Defaults to False.

        Returns:
            float or dict: depends on verbose
        """
        assert self.layers != {}, "Run 'build' before evaluation"
        result = {k: v.evaluate() for k, v in self.layers.items()}
        if verbose:
            return result
        else:
            return reduce(lambda x, y: x+y, result.values())


class TVMBlockEvaluator(BaseBlockEvaluator):
    """Use tvm implementations to evaluate residual block

    For dense version, the evaluator analyzes conv layers.

    x -> conv1x1_dense (BN, ReLU) ->  conv3x3_dense (BN, ReLU) -> conv1x1_dense (BN) -> (add) -> (ReLU) -> out
      \                                                                                  /
       --------------------------------------------------------------------------------->

    For sparse version, the evaluator analyzes performance of some crucial
    layers, which are not in marked with parentheses in the following graph.

    x -> conv3x3_dense (BN, ReLU) -> (pooling) -> conv1x1_dense (+bias)
     \                                                  \ 
      -> conv1x1_dense (BN, ReLU) -> gather -> conv3x3_gathered (BN, ReLU) -> conv1x1_gathered (BN) -> scatter_add -> (ReLU) -> out
      \                                                                                                   /
       -------------------------------------------------------------------------------------------------->

    TODO: support more operations
    """

    def __init__(self, mode, channel, bottleneck, width, group, sparselen, granularity, n_trial=100):
        """initialize TVMBlockEvaluator

        Args:
            n_trial (int or dict, optional): use int for global 'n_trial' setting,
                 or dict for more fine-grained settings. Defaults to 100.
        """
        super().__init__()
        assert width % granularity == 0
        assert channel % bottleneck == 0
        assert sparselen * granularity * granularity < width * width

        self.mode = mode
        self.channel = channel
        self.bottleneck = bottleneck
        self.width = width
        self.group = group
        self.sparselen = sparselen
        self.granularity = granularity
        self.save_dir = f"log/resblock_tvm_{mode}_c{channel}_b{bottleneck}_w{width}_g{group}_l{sparselen}_m{granularity}"

        self.n_trial = n_trial

    def __repr__(self) -> str:
        return "TVMBlockEvaluator"

    def setup(self):
        ########## Instantiate Layer Schedulers ##########
        if self.mode == "sparse":
            self.layers["maskconv1"] = ConvDenseScheduler(
                1, self.channel, self.width, 3, self.save_dir+"/maskconv1.json")
            self.layers["maskconv2"] = ConvDenseScheduler(
                1, 1, self.width//self.granularity, 1, self.save_dir+"/maskconv2.json")
            self.layers["conv1"] = ConvDenseScheduler(
                self.channel//self.bottleneck, self.channel, self.width, 1, self.save_dir+"/conv1.json")
            self.layers["gather"] = GatherScheduler(
                self.width, self.channel//self.bottleneck, self.sparselen, self.granularity, self.save_dir+"/gather.json")
            self.layers["conv2"] = Conv3x3GatheredScheduler(
                self.channel//self.bottleneck, self.group, self.sparselen, self.granularity, self.save_dir+"/conv2.json")
            self.layers["conv3"] = Conv1x1GatheredScheduler(
                self.channel, self.channel//self.bottleneck, self.sparselen, self.granularity, self.save_dir+"/conv3.json")
            self.layers["scatter_add"] = ScatterAddScheduler(
                self.width, self.channel, self.sparselen, self.granularity, self.save_dir+"/scatter_add.json")
        elif self.mode == "dense":
            self.layers["conv1"] = ConvDenseScheduler(
                self.channel//self.bottleneck, self.channel, self.width, 1, self.save_dir+"/conv1.json")
            # didn't impl padding in conv dense
            self.layers["conv2"] = ConvDenseScheduler(
                self.channel//self.bottleneck, self.channel//self.bottleneck, self.width+2, 3, self.save_dir+"/conv2.json")
            self.layers["conv3"] = ConvDenseScheduler(
                self.channel, self.channel//self.bottleneck, self.width, 1, self.save_dir+"/conv3.json")
        else:
            raise RuntimeError("Invalid Mode!")

        ########## Set autotuning parameters ##########
        if isinstance(self.n_trial, int):
            for layer in self.layers.values():
                layer.n_trial = self.n_trial
        elif isinstance(self.n_trial, dict):
            for name, layer in self.layers.values():
                if name in self.n_trial.keys():
                    layer.n_trial = self.n_trial[name]
                elif "default" in self.n_trial.keys():
                    layer.n_trial = self.n_trial["default"]

    def run(self, inputs):
        raise NotImplementedError
