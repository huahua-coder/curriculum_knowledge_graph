from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
# 添加根目录
import sys
import json


def predict(text,his):
    response,history = model.chat(tokenizer,f"{text}",history=his,temperature = 0.01)
    return response

def process_json(input_file, output_file,his):
    # 读取输入的JSON文件
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    pre_data = []
    # 处理每个条目
    for item in data:
        content = item["content"]
        summary = item["summary"]
        # 调用predict函数获取预测结果
        summary_pre = predict(item['content'], his)
        pre_data.append({
        "content": content,
        "summary": summary,
        "summary_pre":summary_pre
        })
    # 将处理后的数据写入新的JSON文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(pre_data, f, ensure_ascii=False, indent=2)


# 代码调用
tokenizer = AutoTokenizer.from_pretrained("/home/ducaili/PythonProjects/ChatGLM2-6B-main/model", trust_remote_code=True)
model = AutoModel.from_pretrained("/home/ducaili/PythonProjects/ChatGLM2-6B-main/model", trust_remote_code=True).cuda('cuda:2')
model = model.eval()
response, history = model.chat(tokenizer, "你好", history=[])

# 添加范例
query_1 ='首先，环境是人发展的外部条件。环境泛指个体生存与其中并影响个体发展 的外部世界。分为自然环境和社会环境两大类。对人的发展起主要影响作用的是 社会环境，关系到一个人的身心能否得到发展或发展到什么程度。'
answer_1='环境在人的发展中的作用'
history.append((query_1,answer_1))

query_2 ='首先，个体的能动性是在人的活动中产生和表现出来的。人作为活动的主体 能动地认识和改造客观世界，其目的是为了解决生活和发展的需要。并在整个活 动过程中表现出主观能动性。'
answer_2 ='个体能动性在人的发展中的作用'
history.append((query_2,answer_2))

query_3 ='首先，教育从本质上来说，是一种有目的地培养人的社会活动。'
answer_3 ='教育的本质'
history.append((query_3,answer_3))


input_file = '/home/ducaili/PythonProjects/ChatGLM2-6B-main/A_paper0521/data-edu/test.json'
output_file = '/home/ducaili/PythonProjects/ChatGLM2-6B-main/A_paper0521/code-ablation/data/data-edu/pre-education-glm2-few-shot.json'
process_json(input_file,output_file,history)
print(".............")



