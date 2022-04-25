# generate a valid argument given the search space
import random
import functools
from tabnanny import check
from textwrap import wrap


class SearchSpaceSampler():
    """Generate a valid setting of a residual block from the search space

    channel: 1 ~ 1024, divisible by 8
    bottleneck: 1, 2, 4
    group: 1, 2, 4, 8, 16, 32
    width: 112, 56, 28, 14

    sparsity: 0 ~ 1, depth of 0.01
    granularity: 1, 2, 4, 8
    """

    def checkwrap(func):
        """If func returns None, call again
        """
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            res = None
            while res == None:
                res = func(*args, **kwargs)
            return res

        return wrapper

    @checkwrap
    def get_block_setting(self):
        """Generate a valid block setting.

        Returns:
            dict: return a valid dict of block setting
        """
        channel = random.randrange(8, 1024, 8)
        bottleneck = random.choice([1, 2, 4])
        group = random.choice([1, 2, 4, 8, 16, 32])
        width = random.choice([112, 56, 28, 14])

        if (channel // bottleneck) % group != 0:
            return None

        return {
            "channel": channel,
            "bottleneck": bottleneck,
            "group": group,
            "width": width,
        }

    @checkwrap
    def get_sparsity_setting(self, block_setting):
        """Generate a valid sparsity setting w.r.t. block setting

        Args:
            block_setting (dict): previously block

        Returns:
            dict: return a valid dict of sparse setting
        """
        sparsity = random.random()
        granularity = random.choice([1, 2, 4, 8])

        if block_setting["width"] % granularity != 0:
            return None

        return {
            "sparselen": int(sparsity * ((block_setting["width"] // granularity) ** 2)),
            "granularity": granularity,
        }