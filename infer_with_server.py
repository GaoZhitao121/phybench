import asyncio
import aiohttp
import json
import subprocess
import time
import os
import sys
import requests
from tqdm.asyncio import tqdm

# === 配置部分 ===
# 模型与硬件配置
MODEL_PATH = "/data/gaozhitao/PSP/models/psp_round_3"
SERVED_MODEL_NAME = "phybench-model"
GPU_DEVICES = "2,3"            # 对应 CUDA_VISIBLE_DEVICES
TENSOR_PARALLEL_SIZE = 2       # 如果用双卡跑一个模型设为2
PORT = 8002

# 数据与输出
INPUT_FILE = "/root/codespace/gaozhitao/PSP_bmk/phybench/dataset/PHYBench-questions_v1.json"
OUTPUT_FILE = "phybench_results_qwen2_5_7B_10_1000_ciritic_1126_round_3.jsonl"
CONCURRENCY_LIMIT = 50         

# API 地址 (本地)
BASE_URL = f"http://localhost:{PORT}"
API_URL = f"{BASE_URL}/v1/chat/completions"

# === 1. 服务器管理模块 ===

def start_vllm_server():
    """启动 vLLM API Server 子进程"""
    print(f"🚀 正在启动 vLLM 服务器 (Port: {PORT}, GPUs: {GPU_DEVICES})...")
    
    # 设置环境变量
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = GPU_DEVICES
    
    # 构建启动命令
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", MODEL_PATH,
        "--served-model-name", SERVED_MODEL_NAME,
        "--port", str(PORT),
        "--tensor-parallel-size", str(TENSOR_PARALLEL_SIZE),
        "--trust-remote-code",
        "--max-model-len", "4096",
        "--gpu-memory-utilization", "0.9"
    ]
    
    # 使用 Popen 启动后台进程
    process = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.DEVNULL, 
        stderr=subprocess.PIPE
    )
    return process

def wait_for_server(process, timeout=600): # 增加一点超时时间以防模型加载慢
    """轮询直到服务器准备就绪"""
    start_time = time.time()
    health_url = f"{BASE_URL}/health"
    print("⏳ 等待服务器就绪...")
    
    while True:
        # 检查子进程是否意外退出
        if process.poll() is not None:
            stdout, stderr = process.communicate()
            print(f"❌ 服务器启动失败！退出代码: {process.returncode}")
            if stderr:
                print(f"错误日志:\n{stderr.decode()}")
            raise RuntimeError("vLLM server failed to start.")

        # 尝试连接健康检查接口
        try:
            resp = requests.get(health_url, timeout=1)
            if resp.status_code == 200:
                print("✅ 服务器已就绪，开始推理！")
                return
        except requests.exceptions.RequestException:
            pass # 连接失败，继续等待

        if time.time() - start_time > timeout:
            process.terminate()
            raise TimeoutError("等待服务器启动超时。")
        
        time.sleep(5)


def build_prompt(content):
    return (
        f"Question: {content}\n\n"
        "Please solve the physics problem above step-by-step. "
        "At the very end, output the final symbolic expression in LaTeX format inside a boxed command, "
        "like \\boxed{expression}. Do not include the derivation inside the box."
    )

async def fetch_response(session, item, semaphore):
    async with semaphore:
        # 注意：这里假设 JSON 对象中的键名为 'content' 和 'answer'
        # 如果报错 KeyError，请检查 JSON 文件中的键名是否为 'Question', 'question' 等