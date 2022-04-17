# dynconv-performance-benchmark
Performance evaluation on residual blocks in dynamic convolution networks.

## Requirements
- pytorch
- tvm

## Quick startup
```python
block = TVMBlockEvaluator(mode, channel, bottleneck, width, group, sparselen, granularity)
perf = block()
```