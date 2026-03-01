#!/bin/bash

if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    echo "Usage: $0 [MODE] [DEVICE] [GEMS_MODE] [RUNS] [OPTIMIZED] [NSYS_PROFILE] [TORCH_PROFILE] [PLUGIN]"
    echo ""
    echo "Arguments:"
    echo "  MODE       : Execution mode. Options: 'cuda' (default) or 'gems'."
    echo "               - 'cuda': Sets USE_GEMS = False."
    echo "               - 'gems': Sets USE_GEMS = True."
    echo "  DEVICE     : CUDA device ID. Default is 4. Port will be 2345 + DEVICE."
    echo "  GEMS_MODE  : Gems mode configuration. Default is 34."
    echo "               Only used when MODE is 'gems'. Sets USE_GEMS_MODE."
    echo "  RUNS       : Number of benchmark runs. Default is 4."
    echo "  OPTIMIZED  : Whether to use optimized scenarios. Default is False."
    echo "  NSYS_PROFILE : Enable nsys profiling. Options: 'True' or 'False' (default)."
    echo "  TORCH_PROFILE: Enable torch profiler in benchmark script. Options: 'True' or 'False' (default)."
    echo "  PLUGIN     : Enable plugin mode (USE_FLAGGEMS=1). Options: 'True' or 'False' (default)."
    echo ""
    exit 0
fi

MODE=${1:-cuda}
DEVICE=${2:-4}
GEMS_MODE=${3:-34}
RUNS=${4:-2}
OPTIMIZED=${5:-False}
NSYS_PROFILE=${6:-False}
TORCH_PROFILE=${7:-False}
PLUGIN=${8:-False}
PORT=$((2345 + DEVICE))

# 根据模式确定配置值和输出文件名
# 使用函数简化 suffix 生成
build_suffix() {
    local flag=$1
    local suffix=$2
    [ "$flag" == "True" ] && echo "${suffix}" || echo ""
}

OPTI_SUFFIX=$(build_suffix "$OPTIMIZED" "_optimized")
NSYS_PROFILE_SUFFIX=$(build_suffix "$NSYS_PROFILE" "_nsys_profile")
TORCH_PROFILE_SUFFIX=$(build_suffix "$TORCH_PROFILE" "_torch_profile")
PLUGIN_SUFFIX=$(build_suffix "$PLUGIN" "_plugin")

if [ "$MODE" == "cuda" ]; then
    LOG_DIR_NAME="./vllm_bench${OPTI_SUFFIX}${NSYS_PROFILE_SUFFIX}${TORCH_PROFILE_SUFFIX}${PLUGIN_SUFFIX}_log/vllm_bench_${MODE}_logs"
    NSYS_NAME="nsys-${MODE}/report-${MODE}${PLUGIN_SUFFIX}"
    TORCH_NAME="torch-${MODE}/report-${MODE}${PLUGIN_SUFFIX}"
    SERVER_LOG_FILE="./vllm_bench_server_log/vllm_bench_${MODE}${OPTI_SUFFIX}${NSYS_PROFILE_SUFFIX}${TORCH_PROFILE_SUFFIX}${PLUGIN_SUFFIX}_server.log"
else
    LOG_DIR_NAME="./vllm_bench${OPTI_SUFFIX}${NSYS_PROFILE_SUFFIX}${TORCH_PROFILE_SUFFIX}${PLUGIN_SUFFIX}_log/vllm_bench_${MODE}_${GEMS_MODE}_logs"
    NSYS_NAME="nsys-${MODE}/report-${MODE}-${GEMS_MODE}${PLUGIN_SUFFIX}"
    TORCH_NAME="torch-${MODE}/report-${MODE}-${GEMS_MODE}${PLUGIN_SUFFIX}"
    SERVER_LOG_FILE="./vllm_bench_server_log/vllm_bench_${MODE}_${GEMS_MODE}${OPTI_SUFFIX}${NSYS_PROFILE_SUFFIX}${TORCH_PROFILE_SUFFIX}${PLUGIN_SUFFIX}_server.log"
fi

mkdir -p ./vllm_bench${OPTI_SUFFIX}${NSYS_PROFILE_SUFFIX}${TORCH_PROFILE_SUFFIX}${PLUGIN_SUFFIX}_log
mkdir -p ./vllm_bench_server_log
mkdir -p ./gems-config

# Update python scripts with the new LOG_DIR
BENCHMARK_SCRIPT="./benchmarks/benchmark_throughput_flagos.py"
if [ ! -f "$BENCHMARK_SCRIPT" ]; then
    echo "Error: ${BENCHMARK_SCRIPT} not found!"
    exit 1
fi

echo "Updating benchmark_throughput_flagos.py configurations..."
sed -i "s/^USE_NSYS_PROFILER[[:blank:]]*=.*/USE_NSYS_PROFILER = ${NSYS_PROFILE}/" "$BENCHMARK_SCRIPT"
sed -i 's/^USE_TORCH_PROFILER[[:blank:]]*=.*/USE_TORCH_PROFILER = '"${TORCH_PROFILE}"'/' "$BENCHMARK_SCRIPT"
sed -i "s|LOG_DIR = \".*\"|LOG_DIR = \"${LOG_DIR_NAME}\"|g" "$BENCHMARK_SCRIPT"
sed -i "s/^PORT = .*/PORT = ${PORT}/g" "$BENCHMARK_SCRIPT"
sed -i "s/^NUM_RUNS = .*/NUM_RUNS = ${RUNS}/g" "$BENCHMARK_SCRIPT"
sed -i "s/^USE_OPTIMIZED = .*/USE_OPTIMIZED = ${OPTIMIZED}/g" "$BENCHMARK_SCRIPT"

# Update TORCH_PROFILER_NAME if TORCH_PROFILE is True
if [ "$TORCH_PROFILE" == "True" ]; then
    sed -i "s|^TORCH_PROFILER_NAME[[:blank:]]*=.*|TORCH_PROFILER_NAME = \"${TORCH_NAME}.json\"|g" "$BENCHMARK_SCRIPT"
fi

# 收集环境变量到数组中
ENV_VARS=()
if [ "$MODE" == "cuda" ] && [ "$PLUGIN" == "True" ]; then
    ENV_VARS+=("USE_FLAGGEMS=0")
elif [ "$MODE" == "gems" ] && [ "$PLUGIN" == "False" ]; then
    ENV_VARS+=("USE_FLAGOS=1")
    # Update USE_GEMS_MODE in gpu_model_runner.py
    GPU_MODEL_RUNNER_FILE="/data/vllm/vllm/v1/worker/gpu_model_runner.py"
    if [ -f "$GPU_MODEL_RUNNER_FILE" ]; then
        echo "Updating USE_GEMS_MODE in ${GPU_MODEL_RUNNER_FILE} to ${GEMS_MODE}..."
        if [[ "$GEMS_MODE" =~ ^[0-9]+$ ]]; then
            sed -i "206s/^[[:space:]]*USE_GEMS_MODE[[:space:]]*=[[:space:]]*.*/    USE_GEMS_MODE = ${GEMS_MODE}/" "$GPU_MODEL_RUNNER_FILE"
        else
            sed -i "206s/^[[:space:]]*USE_GEMS_MODE[[:space:]]*=[[:space:]]*.*/    USE_GEMS_MODE = \"${GEMS_MODE}\"/" "$GPU_MODEL_RUNNER_FILE"
        fi
    else
        echo "Warning: ${GPU_MODEL_RUNNER_FILE} not found, skipping USE_GEMS_MODE update"
    fi

elif [ "$MODE" == "gems" ] && [ "$PLUGIN" == "True" ]; then
    ENV_VARS+=("USE_FLAGGEMS=1")
    ENV_VARS+=("FLAGGEMS_ENABLE_OPLIST_PATH=./gems-${GEMS_MODE}-plugin.txt")
    if [ "$GEMS_MODE" == "33" ]; then
        ENV_VARS+=("VLLM_FL_FLAGOS_WHITELIST=./plugin-gems-whitelist.txt")
    fi
fi

BENCH_COMMAND="vllm serve /models/Qwen3.5-397B-A17B --tensor-parallel-size 8 --gpu_memory_utilization 0.9 \
     --trust-remote-code --max-num-batched-tokens 16384 --max-num-seqs 2048 \
     --served-model-name cpmo --reasoning-parser qwen3 --port ${PORT}" 
PROFILE_COMMAND="python ./benchmarks/benchmark_throughput_flagos.py"

# 构建基础环境变量命令
BASE_ENV="TRITON_PRINT_AUTOTUNING=1"
if [ ${#ENV_VARS[@]} -gt 0 ]; then
    BASE_ENV="${BASE_ENV} env ${ENV_VARS[*]}"
fi

# 执行命令的函数
run_command() {
    local description=$1
    local cmd=$2
    local log_file=${3:-}
    
    echo "Starting ${description} for ${MODE}..."
    [ -n "$log_file" ] && echo "SERVER_LOG_FILE: ${log_file}"
    echo "Running command: ${BASE_ENV} ${cmd}"
    
    if [ -n "$log_file" ]; then
        # 简化输出处理，移除 awk 时间戳以避免缓冲问题
        eval "${BASE_ENV} ${cmd}" 2>&1 | tee "$log_file"
    else
        eval "${BASE_ENV} ${cmd}"
    fi
}

# 根据 profiling 模式执行不同的命令
if [ "$NSYS_PROFILE" == "True" ]; then
    mkdir -p $(dirname "${NSYS_NAME}")
    NSYS_CMD="nsys profile -t cuda,nvtx,osrt -o \"${NSYS_NAME}\" --capture-range=cudaProfilerApi --capture-range-end=stop-shutdown --force-overwrite true ${PROFILE_COMMAND}"
    run_command "nsys profile (Output: ${NSYS_NAME})" "$NSYS_CMD"
elif [ "$TORCH_PROFILE" == "True" ]; then
    mkdir -p $(dirname "${TORCH_NAME}")
    run_command "torch profile" "${PROFILE_COMMAND}"
else
    run_command "server without profiling" "${BENCH_COMMAND}" "${SERVER_LOG_FILE}"
fi
