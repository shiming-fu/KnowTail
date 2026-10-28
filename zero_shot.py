import sys, re
import utils
import json
from tqdm.auto import tqdm


def zero_shot(model_name, ds):
    assert ds in ['trainset', 'testset'], f"{ds} 不在范围内"
    if ds == "testset":
        path = "/home/kai/workspace/longtail/dataset/origin/testset_for_llm.json"
    if ds == "trainset":
        path = "/home/kai/workspace/longtail/dataset/trainset_for_llm.json"
        
    print(f"---------------Zero-Shot {model_name} Exp-------------------")
    print(f"load {ds}...")
    data = utils.load_corpus(path)
    with open("/home/kai/workspace/longtail/dataset/origin/charge_labels.txt", "r", encoding="utf-8") as fi:
        charge_labels = json.load(fi)
    print(f"load model {model_name}")
    model, tokenzier = utils.load_model(model_name)
    
    process_bar = tqdm(range(len(data)))
    count = 0
    with open(f"/home/kai/workspace/longtail/outputs/{model_name}/{ds}_zero_shot.json", "a+", encoding="utf-8") as fo:
        for case in data:
            fact = re.sub(r'[\n\r]', '', case['fact'][:2048])
            prompt = f"请你模拟法官依据下面事实给出罪名，只需要给出罪名的名称，将答案格式为[罪名]。请你严格按照这个格式回答。\n事实：{fact}"
            response = utils.llm_infer(model_name, model, tokenzier, prompt)
            case["response"] = response
            exact_match_answer, exact_and_fuzzy_match_answer = utils.exact_and_fuzzy_match(charge_labels, response, label_num=10)
            case["exact_match_answer"] = exact_match_answer
            case["exact_and_fuzzy_match_answer"] = exact_and_fuzzy_match_answer
            fo.write(json.dumps(case, ensure_ascii=False)+"\n")
            process_bar.update(1)
            count+=1
            if count%500 == 0:
                fo.flush()
    
if __name__=="__main__":
    ds = "testset"
    model_names = ["gpt-4o-mini", "gpt-4o-2024-08-06"]
    # model_names = ["qwen2-0.5b", "qwen2-1.5b", "qwen2-7b", "baichuan2-13b"]#
    for mn in model_names:
        zero_shot(mn, ds)