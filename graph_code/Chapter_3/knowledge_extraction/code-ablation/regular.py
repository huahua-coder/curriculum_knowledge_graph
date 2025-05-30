import json
import re

def process_json(input_file, output_file):
    # 读取输入JSON文件
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 处理每个字典项
    for item in data:
        # 处理summary_pre字段
        if 'summary_pre' in item and item['summary_pre']:
            # 使用正则表达式提取括号内的内容
            match = re.search(r'[(（](.*?)[)）]', item['summary_pre'])
            if match:
                # 如果找到括号，替换为括号内的内容
                item['summary_pre'] = match.group(1)
            else:
                # 如果没有括号，取前5个字符
                item['summary_pre'] = item['summary_pre'][:5]
    
    # 将处理后的数据写入新JSON文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# 使用示例
input_json = '/home/ducaili/PythonProjects/ChatGLM2-6B-main/A_paper0521/data-major/major_computer_pre.json'  # 输入文件路径
output_json = '/home/ducaili/PythonProjects/ChatGLM2-6B-main/A_paper0521/data-major/major_computer_pre_02.json'  # 输出文件路径
process_json(input_json, output_json)
