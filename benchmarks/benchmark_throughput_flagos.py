#!/usr/bin/env python3
import subprocess
import os
from datetime import datetime
import torch
import contextlib

# ====== configs ======
HOST = "127.0.0.1"
PORT = 2345
ENDPOINT = "/v1/chat/completions"
BACKEND = "openai-chat"
USE_OPTIMIZED = True
NUM_RUNS = 2
USE_NSYS_PROFILER = False
USE_TORCH_PROFILER = False

PROFILER_WARMUP = NUM_RUNS-1
SERVED_MODEL_NAME = "/models/Qwen3.5-397B-A17B" if (USE_NSYS_PROFILER or USE_TORCH_PROFILER) else "cpmo"
TOKENIZER_PATH = "/models/Qwen3.5-397B-A17B"
TP_SIZE = 8
# scenarios (name, input_len, output_len, concurrency)
if USE_OPTIMIZED or USE_NSYS_PROFILER or USE_TORCH_PROFILER:
    SCENARIOS = [
        # from FlagScale
        # ("p128d128",     128,    128,   100),
        # ("p6144d128",    6144,   128,   100),
        # ("p128d6144",    128,    6144,  100),
        # ("p6144d6144",   6144,   6144,  100),
        # from FlagRelease
        # ("p4096d2048",   4096,   2048,  64),
        # from vendors
        # ("p6144d1024",   6144,  1024,   100),
        ("p4096d1024",   4096,  1024,   100),
        # ("p2048d1024",   2048,  1024,   100),
        # ("p1024d1024",   1024,  1024,   100),
    ]
else:
    SCENARIOS = [
        # from FlagScale
        ("p128d128",     128,    128,   100),
        ("p6144d128",    6144,   128,   100),
        ("p30720d128",   30720,  128,   100),
        ("p128d6144",    128,    6144,  100),
        ("p6144d6144",   6144,   6144,  100),
        # ("p30720d6144",  30720,  6144,  100),
        # from FlagRelease
        ("p4096d2048",   4096,   2048,  64),
        # # from vendors
        ("p6144d1024",   6144,  1024,   100),
        ("p4096d1024",   4096,  1024,   100),
        ("p2048d1024",   2048,  1024,   100),
        ("p1024d1024",   1024,  1024,   100),
    ]

LOG_DIR = "./vllm_bench_optimized_log/vllm_bench_gems_logs"
os.makedirs(LOG_DIR, exist_ok=True)
TORCH_PROFILER_NAME = "torch-cuda/report-cuda.json"
TORCH_PROFILER_DIR = os.path.abspath(os.path.dirname(TORCH_PROFILER_NAME))
# ====================

def run_benchmark(name, input_len, output_len, concurrency, run_id, torch_profile=False):
    num_prompts = concurrency
    if USE_NSYS_PROFILER or USE_TORCH_PROFILER:
        cmd = [
            # "vllm", "bench", "serve",
            "vllm",
            "bench",
            "throughput",
            # "--host", HOST,
            # "--port", str(PORT),
            # "--backend", BACKEND,
            "--tensor-parallel-size",
            str(TP_SIZE),
            "--model",
            SERVED_MODEL_NAME,
            "--tokenizer",
            TOKENIZER_PATH,
            "--dataset-name",
            "random",
            # "--endpoint", ENDPOINT,
            # "--ignore-eos",
            "--trust-remote-code",
            # "--random-input-len", str(input_len),
            "--input-len", str(input_len),
            "--output-len", str(output_len),
            "--num-prompts", str(num_prompts),
            # "--max-concurrency",str(concurrency)
        ]
        if torch_profile:
            cmd += [
                "--profile",
                "--profiler-config",
                f'{{"profiler": "torch","torch_profiler_dir":"{TORCH_PROFILER_DIR}", "torch_profiler_with_stack": true, "torch_profiler_with_flops": true, "torch_profiler_use_gzip": false, "torch_profiler_dump_cuda_time_total": true, "torch_profiler_record_shapes": true, "torch_profiler_with_memory": false, "ignore_frontend": false, "delay_iterations": 0, "max_iterations": 0}}',
            ]
    else:
        cmd = [
            "vllm", "bench", "serve",
            "--host", HOST,
            "--port", str(PORT),
            "--backend", BACKEND,
            "--model", SERVED_MODEL_NAME,
            "--tokenizer", TOKENIZER_PATH,
            "--dataset-name", "random",
            "--endpoint", ENDPOINT,
            "--ignore-eos",
            "--trust-remote-code",
            "--random-input-len", str(input_len),
            "--random-output-len", str(output_len),
            "--num-prompts", str(num_prompts),
            "--max-concurrency", str(concurrency)
        ]

    log_file = os.path.join(LOG_DIR, f"{name}_run{run_id}.log")
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 🚀 Starting scenario: {name} (Run {run_id})")
    print(f"    Input: {input_len}, Output: {output_len}, Concurrency: {concurrency}")
    print(f"    Logging to: {log_file}")
    print(f"    Command: {' '.join(cmd)}\n")

    with open(log_file, "w") as f:
        result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, text=True)

    status = "✅ Success" if result.returncode == 0 else "❌ Failed"
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {status}: {name} Run {run_id} (exit code: {result.returncode})\n")

def main():
    print(f"🧪 Starting vLLM benchmark suite for {len(SCENARIOS)} scenarios, each repeated {NUM_RUNS} times...\n")
    for name, inp, out, conc in SCENARIOS:
        for run_id in range(1, NUM_RUNS + 1):
            if run_id == PROFILER_WARMUP+1:
                if USE_NSYS_PROFILER:
                    print("Starting nsys profiling...")
                    torch.cuda.profiler.start()
                elif USE_TORCH_PROFILER:
                    print(f"Starting torch profiling in vllm subprocess (Output: {TORCH_PROFILER_NAME})...")
            # Use emit_nvtx to record aten ops with shapes
            if USE_NSYS_PROFILER and run_id == PROFILER_WARMUP + 1:
                ctx = torch.autograd.profiler.emit_nvtx(record_shapes=True)
                torch.cuda.nvtx.range_push(f"Model_Chat_Step_{run_id}")
            else:
                ctx = contextlib.nullcontext()
            try:
                with ctx:
                    if USE_TORCH_PROFILER and run_id == PROFILER_WARMUP + 1:
                        run_benchmark(name, inp, out, conc, run_id, torch_profile=True)
                    else:
                        run_benchmark(name, inp, out, conc, run_id)
            finally:
                # 确保 range_pop 一定会被执行
                if (USE_NSYS_PROFILER or USE_TORCH_PROFILER) and run_id == PROFILER_WARMUP + 1:
                    torch.cuda.nvtx.range_pop()
            if run_id == PROFILER_WARMUP + 1:
                if USE_NSYS_PROFILER:
                    torch.cuda.profiler.stop()
                    print("Stop nsys profiling...")
                elif USE_TORCH_PROFILER:
                    print(f"vllm subprocess profiling complete. Check {TORCH_PROFILER_NAME} for output.")

    print("🏁 All scenarios and runs completed. Logs saved in:", LOG_DIR)

if __name__ == "__main__":
    main()
