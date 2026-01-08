import json
import os
import sys

def load_and_split_dataset(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
    except FileNotFoundError:
        print(f"错误: 数据文件 '{file_path}' 未找到。")
        sys.exit(1)
    
    classification_data = []
    reasoning_dpo_data = []
    
    for entry in dataset:
        if 'content' in entry and 'label' in entry:
            classification_data.append(entry)
        
        if (entry.get('reasoning_text') and entry.get('contentneg1') and 
            entry.get('contentneg2') and entry.get('contentneg3')):
            reasoning_dpo_data.append(entry)
            
    print(f"从 {os.path.basename(file_path)} 加载完成: {len(classification_data)}条用于分类, {len(reasoning_dpo_data)}条用于DPO-Reasoning。")
    return classification_data, reasoning_dpo_data
