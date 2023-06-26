import os
import platform
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

#特点:
#1.自动支持cpu及gpu模式
#2.使用gpu时使用half模式载入，减少一半显存
#3.使用gpu时多显卡模式自动分布载入
#4.暂不支持 聊天上下文功能
#5.暂不支持 打字输出效果 (所以答案太长时会卡死，可以调整MAX_TOKENS来暂时解决)
#作者: lanny.yang.sh@gmail.com 个人兴趣开发者/杨,有问题也可以邮我

def auto_configure_device_map(num_gpus: int):
    num_trans_layers = 32
    per_gpu_layers = num_trans_layers / num_gpus
    device_map = {'model.embed_tokens': 0,
    'model.norm': num_gpus-1, 'lm_head': num_gpus-1}
    for i in range(num_trans_layers):
        device_map[f'model.layers.{i}'] = int(i//per_gpu_layers)
    return device_map


def build_prompt(history):
    prompt = hello_string
    for query, response in history:
        prompt += f"\n\n用户： {query}"
        prompt += f"\n回复： {response}"
    return prompt

#MODEL_NAME = "../baichuan-7B-model"

MODEL_NAME = "baichuan-inc/baichuan-7B"

NUM_GPUS = torch.cuda.device_count() if torch.cuda.is_available() else None
MAX_TOKENS = 512
device_map = auto_configure_device_map(NUM_GPUS) if NUM_GPUS>0 else None
device = torch.device("cuda") if NUM_GPUS>0 else torch.device("cpu")
device_dtype = torch.half if NUM_GPUS>0 else torch.float

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, trust_remote_code=True, device_map=device_map, torch_dtype=device_dtype)
model = model.eval()

os_name = platform.system()
clear_command = 'cls' if os_name == 'Windows' else 'clear'
hello_string = "欢迎使用 BaiChuan-7B 模型，输入内容即可进行对话，clear 清空对话历史，stop/exit/quit 终止程序"



history = []
print(hello_string)

while True:

    query = input("\n用户： ")

    if query.strip() in ["stop", "stop()", "exit", "exit()", "quit", "quit()", "q", "q()"]:
        break
    if query.strip() in ["clear", "clear()", "cls", "cls()"]:
        history = []
        os.system(clear_command)
        print(hello_string)
        continue

    inputs = tokenizer(query, return_tensors='pt')
    inputs.input_ids = inputs.input_ids.to(device)
    inputs.attention_mask = inputs.attention_mask.to(device)
    pred = model.generate(inputs=inputs.input_ids, attention_mask=inputs.attention_mask, max_new_tokens=MAX_TOKENS, repetition_penalty=1.1)
    response = tokenizer.decode(pred.cpu().tolist()[0])
    response = response[len(query)+response.find(query):]
    if response[-4:] == "</s>": response = response[:-4]

    history += [(query, response)]
    print(f"\n回复： {response}")

    os.system(clear_command)
    print(build_prompt(history), flush=True)


