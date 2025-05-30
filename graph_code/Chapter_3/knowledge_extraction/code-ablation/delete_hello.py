import json

def process_json_file(filepath):
    # 读取JSON文件
    with open(filepath, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    # 处理数据
    filtered_data = []
    for item in data:
        # 检查summarypre字段是否存在
        if 'summary_pre' in item:
            summarypre = item['summary_pre']
            # 检查是否满足删除条件
            if '(' not in summarypre or summarypre.startswith('你好'):
                # 如果满足条件，跳过该条目
                continue
        
        # 如果不满足条件，保留该条目
        filtered_data.append(item)
    
    # 将处理后的数据写回原文件
    with open(filepath, 'w', encoding='utf-8') as file:
        json.dump(filtered_data, file, ensure_ascii=False, indent=4)

# 使用示例
filepath = '/home/ducaili/PythonProjects/ChatGLM2-6B-main/A_paper0521/data-major/major_computer_pre.json'  # 替换为您的JSON文件路径
process_json_file(filepath)
