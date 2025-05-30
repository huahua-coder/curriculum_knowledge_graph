import pandas as pd
import json

# 读取Excel文件
excel_file = '/home/ducaili/PythonProjects/ChatGLM2-6B-main/A_paper0521/data-edu/train.xlsx'  # 替换为你的Excel文件路径
df = pd.read_excel(excel_file)

# 检查列名是否正确
if 'point' not in df.columns or 'point_text' not in df.columns:
    raise ValueError("Excel文件必须包含'point'和'point_text'列")

# 转换数据格式
data = []
for index, row in df.iterrows():
    item = {
        "summary": str(row['point']),  # 将point列的值作为summary
        "content": str(row['point_text'])  # 将point_text列的值作为content
    }
    data.append(item)

# 将数据写入JSON文件
json_file = '/home/ducaili/PythonProjects/ChatGLM2-6B-main/A_paper0521/data-edu/train.json'  # 输出的JSON文件路径
with open(json_file, 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)

print(f"数据已成功写入 {json_file}")
