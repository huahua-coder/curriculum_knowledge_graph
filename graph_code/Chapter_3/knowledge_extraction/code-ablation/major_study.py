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
        if summary_pre.startswith("你好"):
            summary_pre_02 = predict(item['content'], his)
            pre_data.append({
            "content": content,
            "summary": summary,
            "summary_pre":summary_pre_02
            })
            
            
        else:
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
model = AutoModel.from_pretrained("/home/ducaili/PythonProjects/ChatGLM2-6B-main/model", trust_remote_code=True).cuda('cuda:1')
model = model.eval()
response, history = model.chat(tokenizer, "你好", history=[])

# # 添加范例
# query_1 ='其次，个体的能动性是人发展的内在动力。人不仅是社会历史活动的主体， 也是自身发展的主体。个体能动性不仅影响个体对环境的选择，也会影响个体对 环境的加工。。'
# answer_1='这段段落的主要观点是(个体能动性在人的发展中的作用)'
# history.append((query_1,answer_1))

# query_2 ='知识的能力价值。知识是行为操作与心理操作认识的结晶。学生学习知识， 不仅要掌握知识的内容，还要把握知识的形式；不仅要掌握对事物的认识，还要 掌握操作事物的方法和能力。'
# answer_2 ='这段段落的主要观点是(知识的能力价值)'
# history.append((query_2,answer_2))

# query_3 ='广义的文化是指人类在社会生活实践过程中创造的精神财富和物质财富的 总和，包括物质文化、精神文化、制度文化。'
# answer_3 ='这段段落的主要观点是(广义的文化的含义)'
# history.append((query_3,answer_3))

# query_4 ='教育的社 会变迁功能包括：教育的经济功能、政治功能、生态功能、文化功能'
# answer_4 ='这段段落的主要观点是(教育的社会变迁功能的组成)'
# history.append((query_4,answer_4))

# 添加范例
query_1 ='TCP是面向连接的运输层协议。这就是说，应用程序在使用TCP 协议之前，必须先 建立TCP 连接。在传送数据完毕后，必须释放已经建立的TCP 连接。也就是说，应用进程 之间的通信好像在\"打电话\":通话前要先拨号建立连接，通话结束后要挂机释放连接。'
answer_1='这段段落的主要观点是(TCP的主要特点)'
history.append((query_1,answer_1))

query_2 ='TCP连接的端点叫做套接字(socket)或插口。根据RFC  793 的定义：端口号拼接到(contatenated with) IP 地址即 构成了套接字'
answer_2 ='这段段落的主要观点是(套接字的概念)'
history.append((query_2,answer_2))

query_3 ='只要超过了一段时间仍然没有收到确认，就认为刚才发送的分组丢失了， 因而重传前面发送过的分组。这就叫做超时重传'
answer_3 ='这段段落的主要观点是(超时重传)'
history.append((query_3,answer_3))



input_file = '/home/ducaili/PythonProjects/ChatGLM2-6B-main/A_paper0521/data-major/test_computer.json'
output_file = '/home/ducaili/PythonProjects/ChatGLM2-6B-main/A_paper0521/data-major/major_computer_pre.json'
process_json(input_file,output_file,history)
print(".............")



