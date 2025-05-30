import json

# 读取三个JSON文件
with open('/home/ducaili/PythonProjects/ChatGLM2-6B-main/A_paper0521/code-ablation/data/data-com/output_01_com.json', 'r', encoding='utf-8') as f1:
    data_01 = json.load(f1)

with open('/home/ducaili/PythonProjects/ChatGLM2-6B-main/A_paper0521/code-ablation/data/data-com/output_02_com.json', 'r', encoding='utf-8') as f2:
    data_02 = json.load(f2)

with open('/home/ducaili/PythonProjects/ChatGLM2-6B-main/A_paper0521/code-ablation/data/data-com/output_03_com.json', 'r', encoding='utf-8') as f3:
    data_03 = json.load(f3)

# 合并数据
combined_data = []
for item_01, item_02, item_03 in zip(data_01, data_02, data_03):
    # 确保content和summary相同
    assert item_01['content'] == item_02['content'] == item_03['content']
    assert item_01['summary'] == item_02['summary'] == item_03['summary']

    # 合并字段
    combined_item = {
        'content': item_01['content'],
        'summary': item_01['summary'],
        'output_01': item_01['output_01'],
        'output_02': item_02['output_01'],
        'output_03': item_03['output_01']
    }
    combined_data.append(combined_item)

# 写入新文件
with open('/home/ducaili/PythonProjects/ChatGLM2-6B-main/A_paper0521/code-ablation/data/data-com/output_all_computer.json', 'w', encoding='utf-8') as f_out:
    json.dump(combined_data, f_out, ensure_ascii=False, indent=4)

