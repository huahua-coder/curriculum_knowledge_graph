from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
# 添加根目录
import sys
sys.path.append('/home/ducaili/PythonProjects/ChatGLM2-6B-main')



# 模型调用
tokenizer = AutoTokenizer.from_pretrained("/home/ducaili/PythonProjects/ChatGLM2-6B-main/model", trust_remote_code=True)
model = AutoModel.from_pretrained("/home/ducaili/PythonProjects/ChatGLM2-6B-main/model", trust_remote_code=True).cuda('cuda:2')
model = model.eval()
response, history = model.chat(tokenizer, "你好", history=[])
prompt_text = """
段落：UDP用户数据报首部中检验和的计算方法有些特殊。在计算检验和时，要在UDP  用户 数据报之前增加12个字节的伪首部。所谓“伪首部”是因为这种伪首部并不是UDP 用户数 据报真正的首部。只是在计算检验和时，临时添加在UDP用户数据报前面，得到一个临时的UDP用户数据报。检验和就是按照这个临时的UDP用户数据报来计算的。伪首部既不向 下传送也不向上递交，而仅仅是为了计算检验和。
请给出段落的核心观点。
"""
response, history = model.chat(tokenizer, prompt_text, history=history)
print(prompt_text)
print(response)

