from zhipuai import ZhipuAI
from openai import OpenAI
client_zp = ZhipuAI(api_key="xxx")
client_openai = OpenAI(api_key="xxx")
client_zp = ZhipuAI(api_key="xxx")
client_openai = OpenAI(api_key="xxx")


model_path = {
    "qwen2-0.5b" : "/home/kai/workspace/modelfiles/qwen2-0.5b",
    "qwen2-1.5b" : "/home/kai/workspace/modelfiles/qwen2-1.5b",
    "qwen2-7b" : "/home/kai/workspace/modelfiles/qwen2-7b",
    "baichuan2-13b" : "/home/kai/workspace/modelfiles/baichuan2-13b",
    "gpt-4o-mini":"",
    "gpt-4o-2024-08-06":""
}

oneline_models = ["gpt-4o-mini", "gpt-4o-2024-08-06"]
loacl_models = ["qwen2-0.5b", "qwen2-1.5b", "qwen2-7b", "baichuan2-13b"]

batch_file_templet = {"custom_id": None, # will be replaced
                "method": "POST", 
                "url": "/v1/chat/completions", 
                "body": {
                    "model": None, # will be replaced 
                    "messages": None, # will be replaced
                    "max_tokens": 1000}}

mode = "online" # local
dataset = "cail"# criminal




