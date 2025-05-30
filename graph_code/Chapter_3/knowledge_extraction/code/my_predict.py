import json
from transformers import AutoModel,AutoConfig,AutoTokenizer
import torch
import os
import pandas as pd

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
# 加载P-Tuning的checkpoint
model_path = "/home/ducaili/PythonProjects/ChatGLM2-6B-main/model"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
config = AutoConfig.from_pretrained(model_path, trust_remote_code=True, pre_seq_len=128)
model = AutoModel.from_pretrained(model_path, config=config, trust_remote_code=True)
CHECKPOINT_PATH = '/home/ducaili/PythonProjects/ChatGLM2-6B-main/A_paper0521/model-edu/checkpoint-250'
prefix_state_dict = torch.load(os.path.join(CHECKPOINT_PATH, "pytorch_model.bin"))
new_prefix_state_dict = {}
for k, v in prefix_state_dict.items():
    if k.startswith("transformer.prefix_encoder."):
        new_prefix_state_dict[k[len("transformer.prefix_encoder."):]] = v
model.transformer.prefix_encoder.load_state_dict(new_prefix_state_dict)

# 使用模型
model = model.quantize(4)
model = model.to(device)
model = model.eval()

# 预测函数
# 封装成预测函数
def predict(text,his):
    response,history = model.chat(tokenizer,f"{text}",history=his,temperature = 0.01)
    return response

# 加载test.json文件
with open("/home/ducaili/PythonProjects/ChatGLM2-6B-main/A_paper0521/data-edu/test.json", "r", encoding="utf-8") as f:
    test_data = json.load(f)

response, history = model.chat(tokenizer, "你好", history=[])

# 进行预测并保存结果
pre_data = []
for item in test_data:
    content = item["content"]
    summary = item["summary"]
    summary_pre = predict(content,history)
    pre_data.append({
        "content": content,
        "summary": summary,
        "summary_pre":summary_pre
    })

# 保存预测结果到pre.json文件
with open("/home/ducaili/PythonProjects/ChatGLM2-6B-main/A_paper0521/data-edu/pre.json", "w", encoding="utf-8") as f:
    json.dump(pre_data, f, ensure_ascii=False, indent=4)

print("预测完成，结果已保存到pre.json文件。")

