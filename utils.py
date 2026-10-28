import json, re, time
import copy
from pkuseg import pkuseg
import torch
from rouge_chinese import Rouge
from tqdm.auto import tqdm
from settings import model_path, client_openai
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
device = "cuda:0"

def exact_and_fuzzy_match(label_list, response, label_num=10):
    # 解析zero shot的输出
    label_list = sorted(label_list, key=lambda x:len(x),reverse=True)
    exact_match_ans = []
    # 精确匹配
    copied_res = copy.copy(response)
    for charge in label_list:
        if charge in copied_res: 
            exact_match_ans.append(charge)
            copied_res = re.sub(f"{charge}", "", copied_res)
    # 精确匹配为空，利用字符召回率
    fuzzy_match_ans = []
    scores = []
    seg = pkuseg()
    rouge = Rouge()
    for charge in label_list:
        ref = " ".join([item for item in seg.cut(charge) if item not in ["、"]])
        hyp = " ".join(seg.cut(response))
        rouge_score = rouge.get_scores(hyp, ref)
        scores.append(rouge_score[0]['rouge-1']['f'])
    if sum(scores) == 0.0:
        return exact_match_ans, fuzzy_match_ans
    scores = torch.tensor(scores, dtype=torch.float32)
    idxs = torch.topk(scores, label_num)[1].tolist()
    for i in idxs:
        if label_list[i] not in exact_match_ans:
            fuzzy_match_ans.append(label_list[i])
    # exact_match_ans, fuzzy_match_ans交集为空集
    return exact_match_ans, fuzzy_match_ans

def parse_answer(zero_shot_res_path):
    with open("/home/kai/workspace/longtail/dataset/origin/charge_labels.txt", "r", encoding="utf-8") as fi:
        charge_labels = json.load(fi)
    output_path = re.sub(r".json","_with_parsed_answer.json",zero_shot_res_path)
    fo = open(output_path, "w", encoding="utf-8")
    with open(zero_shot_res_path, "r", encoding="utf-8") as fi:
        data = fi.readlines()
        process_bar = tqdm(range(len(data)))
        for line in data:
            case = json.loads(line)
            exact_match_ans, exact_and_fuzzy_match_answer = exact_and_fuzzy_match(charge_labels, case["response"], label_num=10)
            case.pop("answer")
            case["exact_match_answer"] = exact_match_ans
            case["exact_and_fuzzy_match_answer"] = exact_and_fuzzy_match_answer
            fo.write(json.dumps(case, ensure_ascii=False)+"\n")
            process_bar.update(1)
    fo.close()

def get_label_mapping(input_path):
    # 构造predicted label到true label的映射表(长尾知识)
    predLabel2trueLabel = {}
    trueLabel2predLabel = {}
    with open(input_path, "r", encoding="utf-8") as fi:
        for line in fi.readlines():
            case = json.loads(line)
            true_label = case["charge"]
            trueLabel2predLabel.setdefault(true_label,{})
            predicted_label = copy.copy(case["exact_match_answer"])
            if len(predicted_label)==0:
                predicted_label.extend(case["exact_and_fuzzy_match_answer"][:1])
            for p in predicted_label:
                trueLabel2predLabel[true_label].setdefault(p,0)
                trueLabel2predLabel[true_label][p]+=1
                predLabel2trueLabel.setdefault(p, {})
                predLabel2trueLabel[p].setdefault(true_label, 0)
                predLabel2trueLabel[p][true_label]+=1
    return predLabel2trueLabel, trueLabel2predLabel

def add_random_label(queries, label_num):
    with open("/home/kai/workspace/longtail/dataset/origin/charge_labels.txt", "r", encoding="utf-8") as fi:
        charge_labels = json.load(fi)
    for q in queries:
        charge_labels.remove(q)
    queries.extend(random.sample(charge_labels, k=label_num-len(queries)))
    return queries

def add_long_tail_label(querys, pl2tl, label_num):
    res = copy.copy(querys)
    if len(res)>=label_num:
        return res
    for q in querys:
        label_dict = pl2tl.get(q, {})
        label_dict.pop(q, None)
        if len(label_dict)==0:
            continue
        sorted_labels = sorted(label_dict.items(), key=lambda x:x[1], reverse=True)
        for item in sorted_labels:
            if len(res)>=label_num:
                break
            if item[0] not in res:
                res.append(item[0])
    return res

def add_long_tail_and_sim_label(querys, pl2tl, similar_crime, label_num):
    res = copy.copy(querys)
    if len(res)>=label_num:
        return res
    for q in querys:
        label_dict = pl2tl.get(q, {})
        label_dict.pop(q, None)
        if len(label_dict)==0:
            continue
        sorted_labels = sorted(label_dict.items(), key=lambda x:x[1], reverse=True)
        for item in sorted_labels:
            if len(res)>=label_num:
                break
            if item[0] not in res:
                res.append(item[0])
    if len(res)<label_num:
        for q in querys:
            for crime in similar_crime[q]:
                if len(res)>=label_num:
                    break
                if crime not in res:
                    res.append(crime)
    return res

def load_model(model_name):
    assert model_name in model_path
    path = model_path[model_name]
    if model_name.startswith("qwen"):
        model = AutoModelForCausalLM.from_pretrained(path,torch_dtype="auto",device_map="auto")
        tokenizer =  AutoTokenizer.from_pretrained(path)
        return model, tokenizer
    if model_name.startswith("baichuan"):
        tokenizer = AutoTokenizer.from_pretrained(path,revision="v2.0",use_fast=False,trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(path,revision="v2.0",device_map="auto",torch_dtype=torch.bfloat16,trust_remote_code=True)
    return model, tokenizer

def llm_infer(model_name, model, tokenizer, prompt):
    if model_name.startswith("baichuan"): 
        model.generation_config = GenerationConfig.from_pretrained(model_path[model_name], revision="v2.0")
        messages = [{"role": "user", "content": prompt}]
        response = model.chat(tokenizer, messages)
    if model_name.startswith("qwen"):
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(device)
        attention_mask = torch.ones(model_inputs.input_ids.shape,dtype=torch.long,device=device)
        generated_ids = model.generate(
            model_inputs.input_ids,
            attention_mask=attention_mask,
            max_new_tokens=512,
            pad_token_id=tokenizer.eos_token_id
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response

def load_corpus(path):
    data = []
    with open(path, "r", encoding="utf-8") as fi:
        for line in fi.readlines():
            data.append(json.loads(line))
    return data

def _parse_gpt_results(bf_result_path, ans_file_path, output_path, response_key):
    """
    Args:
        result_path (_type_): 模型推理结果文件
        source_path (_type_): 带结果选项的文件
        output_path (_type_): 输出文件
    """
    source_data = {}
    with open(ans_file_path, "r", encoding="utf-8") as fi:
        for line in fi.readlines():
            case = json.loads(line)
            source_data[case["id"]]=case
    with open(bf_result_path, "r", encoding="utf-8") as fi:
        for line in fi.readlines():
            case = json.loads(line)
            response = case['response']['body']['choices'][0]['message']['content']
            source_data[int(case["custom_id"])][response_key]=response
    with open(output_path, "w", encoding="utf-8") as fo:
        for _, case in source_data.items():
            fo.write(json.dumps(case, ensure_ascii=False)+'\n')

def invoke_gpt(bf_path, ans_file_path, output_path, response_key):
    print(f"\nload batch input file：{bf_path}")
    batch_input_file = client_openai.files.create(
        file=open(bf_path, "rb"),
        purpose="batch")
    print("\nupload batch input file to server and get batch metadata:")
    batch_info = client_openai.batches.create(
        input_file_id=batch_input_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"description": "nightly eval job"})
    print(batch_info)
    print(f"\nbatch status is {batch_info.status}...")
    while 1:
        time.sleep(60*5)# 每十分钟查询一次处理结果
        check = client_openai.batches.retrieve(batch_info.id)
        print(f"\nbatch status is {check.status}...")
        if check.status == 'completed':
            break
    bf_res_path = re.sub(r".jsonl","_result.jsonl",bf_path)
    file_response = client_openai.files.content(check.output_file_id)
    with open(bf_res_path, "w", encoding="utf-8") as fo:
        fo.write(file_response.text)
    _parse_gpt_results(bf_res_path, ans_file_path, output_path, response_key)

def concel_batches():
    batches = client_openai.batches.list(limit=10)
    for b in batches.data:
        id = b.id
        if b.status != "completed":
            client_openai.batches.cancel(id)
            print(f"{id} is concelled")

            
if __name__=="__main__":
    # parse_answer("/home/kai/workspace/longtail/outputs/gpt-4o-mini/testset_zero_shot.json")
    # pl2tl, tl2pl = get_label_mapping("outputs/gpt-4o-2024-08-06/trainset_for_llm_zero_shot_gpt4_with_parsed_answer.json")
    concel_batches()