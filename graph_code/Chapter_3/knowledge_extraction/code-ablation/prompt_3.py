from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
# 添加根目录
import sys
import json

prompt_text = """
段落：XXXXXX  
请给出段落中的不同观点的关联性"""

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
        content_re = prompt_text.replace('XXXXXX',item['content'])
        # 调用predict函数获取预测结果
        summary_pre = predict(content_re, his)
        print(summary)
        pre_data.append({
        "content": content,
        "summary": summary,
        "output_01":summary_pre
        })
    # 将处理后的数据写入新的JSON文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(pre_data, f, ensure_ascii=False, indent=2)


# 代码调用
tokenizer = AutoTokenizer.from_pretrained("/home/ducaili/PythonProjects/ChatGLM2-6B-main/model", trust_remote_code=True)
model = AutoModel.from_pretrained("/home/ducaili/PythonProjects/ChatGLM2-6B-main/model", trust_remote_code=True).cuda('cuda:2')
model = model.eval()
response, history = model.chat(tokenizer, "你好", history=[])

# # 添加范例
# query_1 ='UDP用户数据报首部中检验和的计算方法有些特殊。在计算检验和时，要在UDP  用户 数据报之前增加12个字节的伪首部。所谓“伪首部”是因为这种伪首部并不是UDP 用户数 据报真正的首部。只是在计算检验和时，临时添加在UDP用户数据报前面，得到一个临时的UDP用户数据报。检验和就是按照这个临时的UDP用户数据报来计算的。伪首部既不向 下传送也不向上递交，而仅仅是为了计算检验和。'
# answer_1='这段段落讲述了 UDP 用户数据报首部中检验和的计算方法的特殊性。在计算检验和时，需要在 UDP 用户数据报之前增加 12 个字节的伪首部。这个伪首部并不是 UDP 用户数据报真正的首部，而仅仅是为了计算检验和而临时添加在 UDP 用户数据报前面。检验和是按照这个临时的 UDP 用户数据报来计算的，而伪首部并不向下游传送也不向上游递交，仅仅是为了计算检验和。'
# history.append((query_1,answer_1))

# query_2 ='TCP是面向连接的运输层协议。这就是说，应用程序在使用TCP 协议之前，必须先 建立TCP 连接。在传送数据完毕后，必须释放已经建立的TCP 连接。也就是说，应用进程 之间的通信好像在\"打电话\":通话前要先拨号建立连接，通话结束后要挂机释放连接。'
# answer_2 ='这段话主要介绍了TCP协议的特点，包括面向连接、传输层协议、应用程序建立TCP连接和释放连接等。'
# history.append((query_2,answer_2))

# query_3 ='首先，教育从本质上来说，是一种有目的地培养人的社会活动。'
# answer_3 ='教育的本质'
# history.append((query_3,answer_3))


input_file = '/home/ducaili/PythonProjects/ChatGLM2-6B-main/A_paper0521/data-edu/test.json'
output_file = '/home/ducaili/PythonProjects/ChatGLM2-6B-main/A_paper0521/code-ablation/data/data-edu/output_03_edu.json'
process_json(input_file,output_file,history)
print(".............")



