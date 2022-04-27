from abc import abstractmethod
from warnings import warn
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

    def __call__(self):
        """Feed in parameters, give time evaluation, simple.

        Returns:
            float: result of evaluation
        """
        self.setup()
        self.autotune()
        self.build()
        return self.evaluate(verbose=False)