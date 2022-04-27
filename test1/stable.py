from layers.base import BaseLayerScheduler
import matplotlib.pyplot as plt
import numpy as np
import os


def test_stability(scheduler, n_trial, repeat):
    """Test scheduler's stability.

    Args:
        scheduler (BaseLayerScheduler): a layer scheduler's instance
        n_trial (int): how many autotvm settings to try
        repeat (int): how many times to run autotuning
    
    Returns:
        list: means of every repeat
    """
    assert isinstance(scheduler, BaseLayerScheduler)
    assert isinstance(n_trial, int)
    assert isinstance(repeat, int)

    res = []
    scheduler.n_trial = n_trial
    for i in range(repeat):
        scheduler.autotune(refresh=True)
        scheduler.build()
        res.append(scheduler.evaluate())
    return res



def test_perf_curve(scheduler, n_trials, repeat=1, plot=False):
    """Test scheduler's performance given multiple number of trials

    Args:
        scheduler (BaseLayerScheduler): a layer scheduler's instance
        n_trials (list of ints): every item indicate how many autotvm settings to try in one experiment
        repeat (int): how many times to run autotuning
        plot (bool): whether to plot in this function

    Returns:
        list: mean of every n_trial
        list: standard deviation of every n_trial
    """
    assert isinstance(scheduler, BaseLayerScheduler)
    assert isinstance(n_trials, (list, tuple))
    assert isinstance(repeat, int)
    assert isinstance(plot, bool)

    means, stds = [], []
    for n_trial in n_trials:
        res = np.array(test_stability(scheduler, n_trial, repeat))
        means.append(res.mean())
        stds.append(res.std())

    if plot:
        r1 = list(map(lambda x: x[0]-x[1], zip(means, stds)))
        r2 = list(map(lambda x: x[0]+x[1], zip(means, stds)))
        plt.plot(n_trials, means)
        plt.fill_between(n_trials, r1, r2, alpha=0.2)

    return means, stds