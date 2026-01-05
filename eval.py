import json
import re
import sys
import numpy as np
import argparse

# 确保能找到 EED 文件夹
sys.path.append("./EED") 
from EED import EED

def extract_boxed_content(text):
    """
    【保持原始逻辑不变】
    提取 \boxed{...} 中的内容。
    """
    idx = text.rfind("\\boxed{")
    if idx == -1:
        return text 
    
    content = ""
    level = 0
    start_record = False
    
    for i in range(idx, len(text)):
        char = text[i]
        if text[i:i+7] == "\\boxed{":
            start_record = True
            level += 1
            continue
            
        if char == '{':
            level += 1
        elif char == '}':
            level -= 1
            if level == 0 and start_record:
                pass

    matches = re.findall(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", text)
    if matches:
        return matches[-1]
    return text

def evaluate(results_file):
    scores = []
    with open(results_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            gt = data['ground_truth']
            if not gt: continue
            pred_raw = data['model_output']
            pred = extract_boxed_content(pred_raw)
            try:
                res = EED(gt, pred)
                score = res[0]
            except: score = 0
            scores.append(score)

    mean_score = np.mean(scores) if scores else 0
    print(f"Result File: {results_file}")
    print(f"Total Samples: {len(scores)} | Average EED Score: {mean_score:.4f}")
    return mean_score

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="推理结果 jsonl 文件路径")
    args = parser.parse_args()
    evaluate(args.input)