from abc import abstractmethod
import numpy as np
import torch
import tvm
from tvm import autotvm
import os
import logging


class BaseLayerScheduler():
    """Base class of layer schedulers
    """

    def __init__(self):
        self.task_name = None  # registered name of tvm task
        self.target = "cuda"  # default target is cuda
        self.device = tvm.device("cuda", 0)  # default device is GPU:0
        self.n_trial = 100  # number of trials in autotuning
        self.rtol = 1e-5  # acceptable accuracy degradation in checking
        self.eval_repeat = 100  # number to repeat in evaluation
        self.save_path = None  # where to save the autotuning result
        self.func = None  # currently best runtime function

    @abstractmethod
    def __repr__(self) -> str:
        return "BaseLayerScheduler"

    @autotvm.template("dynconv/baselayer")
    @abstractmethod
    @property
    def _schedule(self):
        """define operation's schedule and dataflow
        """
        pass

    @abstractmethod
    @property
    def task_args(self):
        """arguments that autotvm uses to create the corresponding task
        """

    @abstractmethod
    def generate_sample(self):
        """generate a random sample in numpy
        """
        return None

    @abstractmethod
    def _run_numpy(self, args):
        raise NotImplementedError

    @abstractmethod
    def _run_pytorch(self, args):
        raise NotImplementedError

    def autotune(self):
        """Autotune the given schedule and log the best one
        """
        # TODO: add logging

        task = autotvm.task.create(
            task_name=self.task_name, args=self.task_args, target=self.target
        )

        measure_option = autotvm.measure_option(
            builder=autotvm.LocalBuilder(),
            runner=autotvm.LocalRunner(repeat=3, min_repeat_ms=150, timeout=4),
        )

        tuner = autotvm.tuner.XGBTuner(task)
        tuner.tune(
            n_trial=self.n_trial,
            measure_option=measure_option,
            callbacks=[autotvm.callback.log_to_file(self.save_path)],
        )


    def build(self):
        """Build a runtime function on the best schedule
        """
        with autotvm.apply_history_best(self.save_path):
            with tvm.target.Target(self.target):
                s, arg_bufs = self._schedule()
                self.func = tvm.build(s, arg_bufs)

    def run(self, inputs, runtype="tvm"):
        """Run tvm runtime function or numpy/pytorch benchmark

        Args:
            inputs (list of numpy ndarrays): input arguments
            runtype (str, optional): type to run. Defaults to "tvm".

        Raises:
            ValueError: invalid runtype

        Returns:
            numpy ndarray: result of the op
        """
        result = None
        if runtype == "tvm":
            assert self.func != None, "You should a tvm runtime function first."
            inputs = [tvm.nd.array(arr, self.device) for arr in inputs]
            result = self.func(*inputs).numpy()
        elif runtype == "numpy":
            result = self._run_numpy(inputs)
        elif runtype == "pytorch":
            result = self._run_pytorch(inputs)
        else:
            raise ValueError(f"Invalid runtype in {self}.run!")

        assert type(result) == np.ndarray, "return type is not numpy array!s"
        return result

    def check(self, runtype="numpy"):
        """Check if the runtime function's result matches benchmark result.
        If nothing happens, the result is valid.

        Args:
            runtype (str, optional): benchmark runtype. Defaults to "numpy".
        """
        assert runtype != "tvm", "'tvm' shouldn't be benchmark runtype"
        sample = self.generate_sample()
        valid = self.run(sample, runtype)
        result = self.run(sample, "tvm")
        tvm.testing.assert_allclose(valid, result, rtol=self.rtol)

    def evaluate(self):
        """Evaluate runtime function's performance

        Returns:
            float: average running time (in seconds)
        """
        assert self.func != None, "You should a tvm runtime function first."
        sample = [tvm.nd.array(_, self.device) for _ in self.generate_sample()]
        evaluator = self.func.time_evaluator(
            self.func.entry_name, self.device, number=self.eval_repeat)
        return evaluator(*sample).mean
