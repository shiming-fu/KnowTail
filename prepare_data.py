import sys
sys.path.append("/home/kai/workspace/longtail")
import json
import random
import re
from settings import model_path
from utils import get_label_mapping, add_long_tail_label, add_random_label
from tqdm.auto import tqdm

def get_crime2cases(input_path):
    with open(input_path, 'r', encoding="utf-8") as fi:
        crime2cases = {}
        for line in fi.readlines():
            case = json.loads(line)
            for c in case['meta']["accusation"]:
                c = re.sub(r"[\[\]]", "", c)
                crime2cases.setdefault(c, [])
                crime2cases[c].append(case['fact'])
    return crime2cases

def get_demo_by_crime(candidate_labels, crime2cases):
    res = ""
    candidate_facts = [random.sample(crime2cases[crime], k=1)[0] for crime in candidate_labels]
    for fact, crime in zip(candidate_facts, candidate_labels):
        res += "事实：{}\n答案：{}\n".format(re.sub("[\n\r]","",fact)[:512], crime)
    return res

def for_rag(bf_path, source_path):
    batch_files = []
    source_files = []
    if "rag_1" in bf_path:
        output_path = f"/home/kai/workspace/longtail/dataset/rag/rag_1.json"
    if "rag_2" in bf_path:
        output_path = f"/home/kai/workspace/longtail/dataset/rag/rag_2.json"
    if "rag_3" in bf_path:
        output_path = f"/home/kai/workspace/longtail/dataset/rag/rag_3.json"
    if "rag_4" in bf_path:
        output_path = f"/home/kai/workspace/longtail/dataset/rag/rag_4.json"
    fo = open(output_path, "w", encoding="utf-8")
    with open(bf_path, "r", encoding="utf-8") as fi:
        for line in fi.readlines():
            case = json.loads(line)
            batch_files.append(case["body"]["messages"][0]["content"])
    with open(source_path, "r", encoding="utf-8") as fi:
        for line in fi.readlines():
            case = json.loads(line)
            source_files.append(case)
    for prompt, case in zip(batch_files, source_files):
         case["prompt"] = prompt
         fo.write(json.dumps(case, ensure_ascii=False)+"\n")
    fo.close()

def for_cot(source_path, num=1):
    demostrations = []
    instruction = "请你模拟法官根据下面事实一步步推理相关罪名。\n"
    with open("/home/kai/workspace/longtail/dataset/origin/cot.json", "r", encoding="utf-8") as fi:
        for line in fi.readlines():
            case = json.loads(line)
            fact = "事实："+re.sub("[\n\r]","",case["fact"])[:1024]+"\n"
            ans = "罪名："+re.sub("[\n\r]","",case["annotation"][0][1])[:256]+"\n"
            demostrations.append(fact+ans)
    fo = open(f"/home/kai/workspace/longtail/dataset/few_cot/{num}_cot.json", "w", encoding="utf-8")
    with open(source_path, "r", encoding="utf-8") as fi:
        for line in fi.readlines():
            case = json.loads(line)
            fact = "事实："+re.sub(r"[\n\r]","",case["fact"])[:1024]+"\n"
            ans = "罪名："
            if num==0:
                case["prompt"] = instruction+fact+ans
            else:
                demos = random.sample(demostrations, k=num)
                demos_str = "".join(demos)
                case["prompt"] = instruction+demos_str+fact+ans
            fo.write(json.dumps(case, ensure_ascii=False)+"\n")

def for_options_random(model_name, option_num):
    source_path = f"/home/kai/workspace/longtail/outputs/{model_name}/testset_zero_shot.json"
    prompt = "你是一个法官，请根据犯罪事实，从选项中选择合适的罪名，只用给出选项。若答案不在选项中请预测一个罪名。 \n犯罪事实：{} \n选项：{}\n"
    option_idxs = ["A", "B", "C", "D", "E","F", "G", "H"]
    fo = open(f"/home/kai/workspace/longtail/dataset/with_options_random/{model_name}_{option_num}_options_random.json", "w", encoding="utf-8")
    with open(source_path, "r", encoding="utf-8") as fi:
        data = fi.readlines()
        process_bar = tqdm(range(len(data)))
        for line in data:
            case = json.loads(line)
            fact = re.sub('[\n\r]','',case["fact"])[:2048]
            # 利用长尾知识获取候选标签
            if len(case["exact_match_answer"])!=0:
                answers = add_random_label(case["exact_match_answer"][:1],  label_num=option_num)
            else:
                answers = add_random_label(case["exact_and_fuzzy_match_answer"][:1], label_num=option_num)
            # 构造prompt
            options = []
            gold_ans = ""
            for idx, crime in zip(option_idxs, answers):
                op = idx+"."+crime+"罪"
                if crime == case["charge"]: # 正确答案
                    gold_ans = op
                options.append(op)
            case["prompt"] = prompt.format(fact, "\n".join(options))+"答案："
            case["options"] = options
            case["gold_option"] = gold_ans
            fo.write(json.dumps(case, ensure_ascii=False)+"\n")
            process_bar.update(1)
    fo.close()

def for_lt_options(model_name, option_num, ds):
    source_path = f"outputs/{model_name}/zero_shot/testset_{ds}.json"
    prompt = "你是一个法官，请根据犯罪事实，从选项中选择合适的罪名，只用给出选项。若答案不在选项中请预测一个罪名。 \n犯罪事实：{} \n选项：{}\n"
    option_idxs = ["A", "B", "C", "D", "E","F", "G", "H"]
    knowledge_path = f"knowledge/trainset_{model_name}_zero_shot.json"
    pl2tl, _ = get_label_mapping(knowledge_path)
    fo = open(f"/home/kai/workspace/longtail/dataset/with_lt_options/{model_name}_{option_num}_options.json", "w", encoding="utf-8")
    with open(source_path, "r", encoding="utf-8") as fi:
        data = fi.readlines()
        process_bar = tqdm(range(len(data)))
        for line in data:
            case = json.loads(line)
            fact = re.sub('[\n\r]','',case["fact"])[:2048]
            # 利用长尾知识获取候选标签
            if len(case["exact_match_answer"])!=0:
                label = []
                label.extend(case["exact_match_answer"])
                label.extend(case["exact_and_fuzzy_match_answer"][:1])
                answers = add_long_tail_label(label, pl2tl, label_num=option_num)
            else:
                answers = add_long_tail_label(case["exact_and_fuzzy_match_answer"][:2], pl2tl, label_num=option_num)
            # 构造prompt
            options = []
            gold_ans = ""
            for idx, crime in zip(option_idxs, answers):
                op = idx+"."+crime+"罪"
                if crime == case["charge"]: # 正确答案
                    gold_ans = op
                options.append(op)
            case["prompt"] = prompt.format(fact, "\n".join(options))+"答案："
            case["options"] = options
            case["gold_option"] = gold_ans
            fo.write(json.dumps(case, ensure_ascii=False)+"\n")
            process_bar.update(1)
    fo.close()

def for_lt_options_desc(model_name, option_num):
    source_path = f"/home/kai/workspace/longtail/outputs/{model_name}/testset_zero_shot.json"
    prompt = "你是一个法官，请根据犯罪事实，从选项中选择合适的罪名，只用给出选项。若答案不在选项中请预测一个罪名。 \n犯罪事实：{} \n选项：{}\n"
    option_idxs = ["A", "B", "C", "D", "E","F", "G", "H"]
    knowledge_path = f"/home/kai/workspace/longtail/knowledge/trainset_{model_name}_zero_shot.json"
    pl2tl, _ = get_label_mapping(knowledge_path)
    with open("/home/kai/workspace/longtail/knowledge/crime2desc.json", "r", encoding="utf-8") as fi:
        crime2desc = json.load(fi)
    fo = open(f"/home/kai/workspace/longtail/dataset/with_lt_options_desc/{model_name}_{option_num}_options.json", "w", encoding="utf-8")
    with open(source_path, "r", encoding="utf-8") as fi:
        data = fi.readlines()
        process_bar = tqdm(range(len(data)))
        for line in data:
            case = json.loads(line)
            fact = re.sub('[\n\r]','',case["fact"])[:2048]
            # 利用长尾知识获取候选标签
            if len(case["exact_match_answer"])!=0:
                label = []
                label.extend(case["exact_match_answer"])
                label.extend(case["exact_and_fuzzy_match_answer"][:1])
                answers = add_long_tail_label(label, pl2tl, label_num=option_num)
            else:
                answers = add_long_tail_label(case["exact_and_fuzzy_match_answer"][:2], pl2tl, label_num=option_num)
            # 构造prompt
            options = []
            gold_ans = ""
            crime_kwg = ""
            for idx, crime in zip(option_idxs, answers):
                op = idx+"."+crime+"罪"
                crime_kwg += op+crime2desc[crime]
                if crime == case["charge"]: # 正确答案
                    gold_ans = op
                options.append(op)
            case["prompt"] = prompt.format(fact, "\n".join(options))+crime_kwg+"因此答案选："
            case["options"] = options
            case["gold_option"] = gold_ans
            fo.write(json.dumps(case, ensure_ascii=False)+"\n")
            process_bar.update(1)
    fo.close()

def for_lt_options_case(model_name, option_num):
    source_path = f"/home/kai/workspace/longtail/outputs/{model_name}/testset_zero_shot.json"
    prompt = "你是一个法官，请根据犯罪事实，从选项中选择合适的罪名，只用给出选项。若答案不在选项中请预测一个罪名。 \n犯罪事实：{} \n选项：{}\n"
    option_idxs = ["A", "B", "C", "D", "E","F", "G", "H"]
    knowledge_path = f"/home/kai/workspace/longtail/knowledge/trainset_{model_name}_zero_shot.json"
    pl2tl, _ = get_label_mapping(knowledge_path)
    crime2cases = get_crime2cases("/home/kai/workspace/longtail/dataset/origin/rag_corpus.json")
    fo = open(f"/home/kai/workspace/longtail/dataset/with_lt_options_case/{model_name}_{option_num}_options.json", "w", encoding="utf-8")
    with open(source_path, "r", encoding="utf-8") as fi:
        data = fi.readlines()
        process_bar = tqdm(range(len(data)))
        for line in data:
            case = json.loads(line)
            fact = re.sub('[\n\r]','',case["fact"])[:2048]
            # 利用长尾知识获取候选标签
            if len(case["exact_match_answer"])!=0:
                label = []
                label.extend(case["exact_match_answer"])
                label.extend(case["exact_and_fuzzy_match_answer"][:1])
                answers = add_long_tail_label(label, pl2tl, label_num=option_num)
            else:
                answers = add_long_tail_label(case["exact_and_fuzzy_match_answer"][:2], pl2tl, label_num=option_num)
            demostrations = get_demo_by_crime(answers, crime2cases)
            # 构造prompt
            options = []
            gold_ans = ""
            for idx, crime in zip(option_idxs, answers):
                op = idx+"."+crime+"罪"
                if crime == case["charge"]: # 正确答案
                    gold_ans = op
                options.append(op)
            case["prompt"] = demostrations+"\n"+prompt.format(fact, "\n".join(options))+"答案："
            case["options"] = options
            case["gold_option"] = gold_ans
            fo.write(json.dumps(case, ensure_ascii=False)+"\n")
            process_bar.update(1)
    fo.close()

def for_lt_options_desc_case(model_name, option_num):
    source_path = f"/home/kai/workspace/longtail/outputs/{model_name}/testset_zero_shot.json"
    prompt = "你是一个法官，请根据犯罪事实，从选项中选择合适的罪名，只用给出选项。若答案不在选项中请预测一个罪名。 \n犯罪事实：{} \n选项：{}\n"
    option_idxs = ["A", "B", "C", "D", "E","F", "G", "H"]
    knowledge_path = f"/home/kai/workspace/longtail/knowledge/trainset_{model_name}_zero_shot.json"
    pl2tl, _ = get_label_mapping(knowledge_path)
    with open("/home/kai/workspace/longtail/knowledge/crime2desc.json", "r", encoding="utf-8") as fi:
        crime2desc = json.load(fi)
    crime2cases = get_crime2cases("/home/kai/workspace/longtail/dataset/origin/rag_corpus.json")
    fo = open(f"/home/kai/workspace/longtail/dataset/with_lt_options_desc_case/{model_name}_{option_num}_options.json", "w", encoding="utf-8")
    with open(source_path, "r", encoding="utf-8") as fi:
        data = fi.readlines()
        process_bar = tqdm(range(len(data)))
        for line in data:
            case = json.loads(line)
            fact = re.sub('[\n\r]','',case["fact"])[:2048]
            # 利用长尾知识获取候选标签
            if len(case["exact_match_answer"])!=0:
                label = []
                label.extend(case["exact_match_answer"])
                label.extend(case["exact_and_fuzzy_match_answer"][:1])
                answers = add_long_tail_label(label, pl2tl, label_num=option_num)
            else:
                answers = add_long_tail_label(case["exact_and_fuzzy_match_answer"][:2], pl2tl, label_num=option_num)
            demostrations = get_demo_by_crime(answers, crime2cases)
            # 构造prompt
            options = []
            gold_ans = ""
            crime_kwg = ""
            for idx, crime in zip(option_idxs, answers):
                op = idx+"."+crime+"罪"
                crime_kwg += op+crime2desc[crime]
                if crime == case["charge"]: # 正确答案
                    gold_ans = op
                options.append(op)
            case["prompt"] = demostrations+"\n"+prompt.format(fact, "\n".join(options))+crime_kwg+"因此答案选："
            case["options"] = options
            case["gold_option"] = gold_ans
            fo.write(json.dumps(case, ensure_ascii=False)+"\n")
            process_bar.update(1)
    fo.close()

if __name__=="__main__":
    for name in model_path.keys():
        for op_num in [4,5,6,7,8]:
            print(f"model_name: {name} | option_num:{op_num}")
            for_lt_options(name, op_num, ds="small")
            for_lt_options_case(name, op_num, ds="small")
            for_lt_options_desc(name, op_num, ds="small")
            for_lt_options_desc_case(name, op_num, ds="small")