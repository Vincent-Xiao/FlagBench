# FlagBench：vllm大模型性能测试与profling指南

## 环境搭建
搭建vllm+Flaggems+FlagTree的环境，参考[FlagOS部署指南](FlagOS.md)，其中vllm版本根据自己需要测试的模型进行选择，本仓库使用0.16.0rc2版本进行测试，以支持Qwen3.5
```shell
pip install -U vllm 0.16.0rc2.dev447+g3bbb2046f --extra-index-url https://wheels.vllm.ai/nightly
```

## vllm补丁
在 Gems 模式下，需要修改 vLLM 的vllm/v1/worker/gpu_model_runner.py来启用 FlagGems 算子，参考当前目录下[gpu_model_runner.py](./gpu_model_runner.py)中的201到233行代码
```python
# vllm/vllm/v1/worker/gpu_model_runner.py
import os
if os.getenv("USE_FLAGOS") == "1":
    import flag_gems

    USE_GEMS_MODE = all

    FlagGemsList=["sort", "sort_stable", "layer_norm", "clamp_", "cos", "embedding", "exp", "exponential_", "full", "gather", "gelu", "index", "le", "lt", "lt_scalar", "masked_fill_", "max", "ones", "pow_scalar", "prod_dim", "rand_like", "reciprocal", "repeat", "scatter", "scatter_", "sin", "sub", "true_divide", "true_divide_", "uniform_", "where_scalar_self", "where_self_out", "zeros", "zeros_like"]


    if USE_GEMS_MODE == "all": ##开启所有FlagGems算子
        flag_gems.enable(record=True,
                         once=True,
                         path=f"./gems-config/gems-{USE_GEMS_MODE}.txt")
    elif USE_GEMS_MODE == 'list': ##使用include API开启自定义算子列表
        flag_gems.only_enable(record=True,
                              once=True,
                              path=f"./gems-config/gems-{USE_GEMS_MODE}.txt",
                              include=FlagGemsList)
    elif USE_GEMS_MODE == 0: ##使用unused API单独关闭部分算子，开启其他FlagGems算子
        flag_gems.enable(record=True,
                         once=True,
                         path=f"./gems-config/gems-{USE_GEMS_MODE}.txt",
                         unused=FlagGemsList)
    else: ##自定义模式，根据环境变量传入的算子名称或者算子列表，单独开启部分算子，其他算子保持默认
        keep_ops = [USE_GEMS_MODE] if isinstance(USE_GEMS_MODE, str) else USE_GEMS_MODE
        model_name = USE_GEMS_MODE if isinstance(USE_GEMS_MODE,str) else "custom"
        flag_gems.only_enable(record=True,
                              once=True,
                              path=f"./gems-config/gems-{model_name}-{keep_ops}.txt",
                              include=keep_ops)
```
将这部分代码添加到vllm安装目录下vllm/v1/worker/gpu_model_runner.py文件中对应位置

## vllm模型benchmark
### 前置修改
修改服务端脚本run-server-bench.sh中的模型路径,并行参数和served-model-name
```shell
BENCH_COMMAND="vllm serve /models/Qwen3.5-397B-A17B --tensor-parallel-size 8 --gpu_memory_utilization 0.9 \
     --trust-remote-code --max-num-batched-tokens 16384 --max-num-seqs 2048 \
     --served-model-name qwen --reasoning-parser qwen3 --port ${PORT}" 
```
以及客户端benchmark脚本benchmarks/benchmark_throughput_flagos.py中的模型路径,并行参数和served-model-name
```python
SERVED_MODEL_NAME = "/models/Qwen3.5-397B-A17B" if (USE_NSYS_PROFILER or USE_TORCH_PROFILER) else "cpmo"
TOKENIZER_PATH = "/models/Qwen3.5-397B-A17B"
TP_SIZE = 8
```
### 启动服务端
```shell
## cuda
./run-server-bench.sh cuda 0 all 4 False False False
## gems
./run-server-bench.sh gems 0 all 4 False False False
##使用说明
Usage: ./run-server-bench.sh [MODE] [DEVICE] [GEMS_MODE] [RUNS] [OPTIMIZED] [NSYS_PROFILE] [TORCH_PROFILE] [PLUGIN]

Arguments:
  MODE       : Execution mode. Options: 'cuda' (default) or 'gems'.
               - 'cuda': Sets USE_GEMS = False.
               - 'gems': Sets USE_GEMS = True.
  DEVICE     : CUDA device ID. Default is 4. Port will be 2345 + DEVICE.
  GEMS_MODE  : Gems mode configuration. Default is 34.
               Only used when MODE is 'gems'. Sets USE_GEMS_MODE.
  RUNS       : Number of benchmark runs. Default is 4.
  OPTIMIZED  : Whether to use optimized scenarios. Default is False.
  NSYS_PROFILE : Enable nsys profiling. Options: 'True' or 'False' (default).
  TORCH_PROFILE: Enable torch profiler in benchmark script. Options: 'True' or 'False' (default).
  PLUGIN     : Enable plugin mode (USE_FLAGGEMS=1). Options: 'True' or 'False' (default).
```
- 当使用gems模式时，`./gems-config/gems-{USE_GEMS_MODE}.txt`会记录开启的FlagGems算子列表，从而验证FlagGems成功执行
- 参数optimized为True时，指定benchmarks/benchmark_throughput_flagos.py中的场景（默认p4096d1024）测试

### 启动客户端
执行run-bench.sh调用benchmarks/benchmark_throughput_flagos.py进行性能测试
```shell
./run-bench.sh
```
### 结果分析
性能数据会保存在vllm-bench-log目录下，根据不同的模式和场景命名；性能测试完毕后可以执行`python processing/bench_stat.py`来统计性能数据，生成性能对比图表，保存markdown文件在reports目录下

## Profling
### 启动profling
在服务端脚本run-server-bench.sh中开启NSYS_PROFILE(推荐使用)或者TORCH_PROFILE参数来进行性能分析，生成的分析文件会保存在nsys-cuda或nsys-gems目录下
```shell
./run-server-bench.sh cuda 0 all 4 False True False
```
`NSYS_PROFILE`设为True时,不会单独启动vllm服务端，直接使用`vllm bench throughput`在optimized场景下测试

### profling分析
```shell
./export_nsys.sh # 导出nsys分析结果为sqlite文件
python processing/perf_analysis.py # 分析sqlite文件，生成性能分析报告
```
在reports目录下会生成性能分析报告，包含热点函数分析，算子分析，gems和cuda性能对比分析等各种报告

## 注意事项
1. FlagGems和FlagTree需要充分预热，建议在正式测试前先进行一次完整的benchmark，来预热FlagGems和FlagTree，以达到最佳性能；如果不预热，可能会因为编译开销导致性能不稳定；cuda最好也预热一次；
比如想单独benchmark optimized场景（默认p4096d1024），
```shell
./run-server-bench.sh  cuda 0 all 4 True False False #cuda预热
./run-server-bench.sh  gems 0 all 4 True False False #gems预热
./run-server-bench.sh  cuda 0 all 4 True False False #cuda正式测试
./run-server-bench.sh  gems 0 all 4 True False False #gems正式测试
./run-server-bench.sh  cuda 0 all 4 False True False #cuda nsys分析
./run-server-bench.sh  gems 0 all 4 False True False #gems nsys分析

```

2. 性能分析时，建议只分析optimized场景（默认p4096d1024），以解决测试时间；optimized=False时，测试场景较多，且每个场景的测试时间较长，可能会导致性能分析时间过长；如果想分析其他场景，可以在benchmarks/benchmark_throughput_flagos.py中修改SCENARIOS变量来指定想分析的场景
