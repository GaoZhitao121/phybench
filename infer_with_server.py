import asyncio
import aiohttp
import json
import subprocess
import time
import os
import sys
import requests
import argparse
from tqdm.asyncio import tqdm

# 解析命令行参数
def parse_args():
    parser = argparse.ArgumentParser(description="vLLM Inference Script")
    parser.add_argument("--model_path", type=str, required=True, help="模型在磁盘上的物理路径")
    parser.add_argument("--gpu", type=str, default="3", help="使用的 GPU 编号")
    parser.add_argument("--port", type=int, default=8003, help="vLLM 服务端口")
    parser.add_argument("--input_file", type=str, required=True, help="输入 JSON 文件路径")
    parser.add_argument("--tp", type=int, default=1, help="Tensor Parallel Size")
    parser.add_argument("--concurrency", type=int, default=50, help="并发请求数")
    return parser.parse_args()

def get_identifiers(model_path):
    """根据路径自动提取实验名和轮次名"""
    path = os.path.abspath(model_path).rstrip("/")
    round_name = os.path.basename(path) # psp_round_5
    exp_name = os.path.basename(os.path.dirname(os.path.dirname(path))) # Qwen2.5_...
    return exp_name, round_name

args = parse_args()
EXP_NAME, ROUND_NAME = get_identifiers(args.model_path)

# 自动生成的配置
SERVED_MODEL_NAME = f"{EXP_NAME}_{ROUND_NAME}"
GPU_DEVICES = args.gpu
PORT = args.port
MODEL_PATH = args.model_path
INPUT_FILE = args.input_file
# 输出文件名格式：phybench_results_实验名_轮次名.jsonl
OUTPUT_FILE = f"phybench_results_{EXP_NAME}_{ROUND_NAME}.jsonl"
BASE_URL = f"http://localhost:{PORT}"
API_URL = f"{BASE_URL}/v1/chat/completions"

# --- 以下保留原始服务器管理和推理逻辑 ---

def start_vllm_server():
    print(f"🚀 正在启动 vLLM 服务器 (Port: {PORT}, GPUs: {GPU_DEVICES})...")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = GPU_DEVICES
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", MODEL_PATH,
        "--served-model-name", SERVED_MODEL_NAME,
        "--port", str(PORT),
        "--tensor-parallel-size", str(args.tp),
        "--trust-remote-code",
        "--max-model-len", "4096",
        "--gpu-memory-utilization", "0.9"
    ]
    process = subprocess.Popen(cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    return process

def wait_for_server(process, timeout=600):
    start_time = time.time()
    health_url = f"{BASE_URL}/health"
    print(f"⏳ 等待服务器就绪: {SERVED_MODEL_NAME}...")
    while True:
        if process.poll() is not None:
            _, stderr = process.communicate()
            raise RuntimeError(f"vLLM server failed to start: {stderr.decode()}")
        try:
            if requests.get(health_url, timeout=1).status_code == 200:
                print("✅ 服务器已就绪，开始推理！")
                return
        except: pass
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
        prompt_content = build_prompt(item['content'])
        payload = {
            "model": SERVED_MODEL_NAME,
            "messages": [{"role": "user", "content": prompt_content}],
            "temperature": 0.0,
            "max_tokens": 2048
        }
        try:
            async with session.post(API_URL, json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    output_text = result['choices'][0]['message']['content']
                else: output_text = f"Error: {response.status}"
        except Exception as e: output_text = f"Exception: {str(e)}"
        return {"id": item.get('id'), "question": prompt_content, "ground_truth": item.get('answer'), "model_output": output_text}

async def run_inference():
    with open(INPUT_FILE, 'r', encoding='utf-8') as f: items = json.load(f)
    print(f"📦 加载数据: {len(items)} 条")
    semaphore = asyncio.Semaphore(args.concurrency)
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_response(session, item, semaphore) for item in items]
        results = []
        for f in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc=f"Inferencing {ROUND_NAME}"):
            results.append(await f)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for res in results: f.write(json.dumps(res, ensure_ascii=False) + '\n')
    print(f"💾 结果保存至: {OUTPUT_FILE}")

if __name__ == "__main__":
    server_process = None
    try:
        server_process = start_vllm_server()
        wait_for_server(server_process)
        asyncio.run(run_inference())
    finally:
        if server_process:
            server_process.terminate()
            server_process.wait()