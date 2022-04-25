import sys
sys.path.append("..")
import numpy as np
import os
from functools import reduce
import json

from layers import *
from blocks import *
from utils import *

# test tvm block evaluator on multiple points
logpath = "blockResult.json"

def run_tvm_block(**kwargs):
    tvm_block = TVMBlockEvaluator(**kwargs, n_trial=100) # 100 takes about 15min
    tvm_block.setup()
    tvm_block.autotune()
    tvm_block.build()
    return tvm_block.evaluate(verbose=True)

sampler = SearchSpaceSampler()
for i in range(100):
    block_setting = sampler.get_block_setting()
    for j in range(3):
        sparsity_setting = sampler.get_sparsity_setting(block_setting)

        res_sparse = run_tvm_block(mode="sparse", **block_setting, **sparsity_setting)
        res_dense = run_tvm_block(mode="dense", **block_setting, **sparsity_setting)

        sparse_total = reduce(lambda x, y: x+y, res_sparse.values())
        dense_total = reduce(lambda x, y: x+y, res_dense.values())

        res_sparse = {k: f"{v / sparse_total:.4%}" for k, v in res_sparse.items()}
        res_dense = {k: f"{v / dense_total:.4%}" for k, v in res_dense.items()}

        record = {
            "setting": [*block_setting.values(), *sparsity_setting.values()],
            "sparse_total": f"{sparse_total*1000:.4f}ms",
            "dense_total": f"{dense_total*1000:.4f}ms",
            "sparse_detail": res_sparse,
            "dense_detail": res_dense,
        }

        with open(logpath, "a+") as f:
            f.write(json.dumps(record) + "\n")
        