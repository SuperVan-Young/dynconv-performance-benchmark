from abc import abstractmethod
import numpy as np
import tvm
from tvm import autotvm
import os


class BaseLayerScheduler():
    """Base class of layer schedulers
    """

    def __init__(self):
        self.target = "cuda"  # default target is cuda
        self.device = tvm.device("cuda", 0)  # default device is GPU:0
        self.n_trial = 100  # number of trials in autotuning
        self.rtol = 1e-5  # acceptable relative accuracy degradation
        self.atol = 1e-7  # acceptable absolute accuracy degradation
        self.eval_repeat = 100  # number to repeat in evaluation
        self.save_path = None  # where to save the autotuning result
        self.func = None  # currently best runtime function

    @abstractmethod
    def __repr__(self) -> str:
        return "BaseLayerScheduler"

    @property
    @abstractmethod
    def _schedule(self):
        """define operation's schedule and dataflow
        """
        pass

    @property
    @abstractmethod
    def _task_name(self):
        """autotvm uses this to find the template
        """

    @property
    @abstractmethod
    def _task_args(self):
        """autotvm uses these arguments to create a scheduling task
        """

    @abstractmethod
    def _generate_sample(self):
        """generate a random sample.

        Don't forget to EXPLICITLY point out the ndarray's dtype

        Returns:
            list: inner elements are all numpy.ndarray
        """
        return None
    
    @abstractmethod
    def _convert_sample(self, sample):
        """Convert sample so that runtime function could directly use it

        Args:
            sample (list): randomly generated sample in numpy ndarray format

        Returns:
            list: a list of both input and output tvm.nd.arrays that could
            be sent to tvm runtime function directly
        """
        return None

    @abstractmethod
    def _run_tvm(self, inputs):
        raise NotImplementedError

    @abstractmethod
    def _run_numpy(self, inputs):
        raise NotImplementedError

    @abstractmethod
    def _run_pytorch(self, inputs):
        raise NotImplementedError

    def autotune(self, refresh=False):
        """Autotune the given schedule and log the best one

        Args:
            refresh (bool, optional): refresh the log. Default to False.
        """

        # TODO: add logging
        assert self.save_path != None, "Give a saving path for autotuning!"

        if os.path.exists(self.save_path) and refresh:
            os.remove(self.save_path)

        task = autotvm.task.create(
            task_name=self._task_name, args=self._task_args, target=self.target
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

    def build(self, display=False):
        """Build a runtime function on the best schedule
        """
        with autotvm.apply_history_best(self.save_path):
            with tvm.target.Target(self.target):
                s, arg_bufs = self._schedule
                self.func = tvm.build(s, arg_bufs)
                if display:
                    print(tvm.lower(s, arg_bufs, simple_mode=True))

    def run(self, inputs, runtype="tvm"):
        """Run tvm runtime function or numpy/pytorch benchmark.

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
            assert self.func is not None, "You should a tvm runtime function first."
            result = self._run_tvm(inputs)
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
        sample = self._generate_sample()
        valid = self.run(sample, runtype)
        result = self.run(sample, "tvm")
        tvm.testing.assert_allclose(valid, result, rtol=self.rtol, atol=self.atol)

    def evaluate(self):
        """Evaluate runtime function's performance

        Returns:
            float: average running time (in seconds)
        """
        assert self.func is not None, "You should a tvm runtime function first."
        sample = self._generate_sample()
        sample = self._convert_sample(sample)
        evaluator = self.func.time_evaluator(
            self.func.entry_name, self.device, number=self.eval_repeat)
        return evaluator(*sample).mean
