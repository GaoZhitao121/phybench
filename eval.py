import json
import re
import sys
import numpy as np

# 确保能找到 EED 文件夹
sys.path.append("./EED") 
from EED import EED

def extract_boxed_content(text):
    """
    提取 \boxed{...} 中的内容。
    PHYBench 推荐模型输出 boxed，这是最稳健的提取方式。
    """
    # 匹配最内层的 boxed，或者最后一个 boxed
    # 这是一个简化的正则，处理简单的嵌套需更复杂逻辑，但对大多数模型输出足够
    idx = text.rfind("\\boxed{")
    if idx == -1:
        return text # 如果没有 boxed，返回全文让 EED 尝试处理（通常分很低）
    
    content = ""
    level = 0
    start_record = False
    
    for i in range(idx, len(text)):
        char = text[i]
        if text[i:i+7] == "\\boxed{":
            start_record = True
            level += 1
            # 跳过 \boxed{ 也就是跳过6个字符，循环会自动+1
            # 这里简单处理，直接从大括号开始计数
            continue
            
        if char == '{':
            level += 1
        elif char == '}':
            level -= 1
            if level == 0 and start_record:
                # 找到匹配的结束括号
                # 注意：这里需要精确截取 \boxed{ 之后的内容
                # 简单起见，建议用正则或专门的括号匹配库
                # 这里为了演示仅做简单说明，建议直接把整个 text 传给 latex_pre_process 
                # 如果 latex_pre_process 处理能力够强的话。
                # 但查看你的 latex_pre_process.py，它有 extract_last_equal_content 
                # 和 remove_command('\\boxed')，所以直接传 boxed 块比较好。
                pass

    # 更简单的正则提取最后一个 boxed
    matches = re.findall(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", text)
    if matches:
        return matches[-1]
    return text

def evaluate():
    results_file = "phybench_results_qwen2_5_7B_10_1000_ciritic_1126_round_2.jsonl"
    scores = []
    
    with open(results_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            
            gt = data['ground_truth'] # 例如: "$$\\gamma = \\frac{2}{5}$$"
            if gt == "":
                continue
            pred_raw = data['model_output']
            
            # 提取模型预测的关键部分
            pred = extract_boxed_content(pred_raw)
            
            try:
                # EED 返回 (score, rel_dist, tree_size, dist)
                # 我们只需要 score
                res = EED(gt, pred)
                score = res[0]
            except Exception as e:
                print(f"Error computing EED for ID {data['id']}: {e}")
                score = 0
            
            scores.append(score)
            
            # 打印部分结果用于检查
            # print(f"ID: {data['id']} | Score: {score}")
            # print(f"GT: {gt}")
            # print(f"Pred: {pred}")
            # print("-" * 20)

    print(f"Total Samples: {len(scores)}")
    print(f"Average EED Score: {np.mean(scores):.2f}")

if __name__ == "__main__":
    evaluate()