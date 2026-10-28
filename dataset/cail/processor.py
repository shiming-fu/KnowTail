import json
import random
import re
from rank_bm25 import BM25Okapi
import torch
import pkuseg
from tqdm.auto import tqdm

def load_corpus(path):
    data = []
    with open(path, "r", encoding="utf-8") as fi:
        for line in fi.readlines():
            data.append(json.loads(line))
    return data

def recall_of_bm25IR(path="dataset/cail/testset_large_with_bm25_sim_cases.json", top_k=10):
    total_instance = 0
    recalled_label = 0
    with open(path, "r", encoding="utf-8") as fi:
        for line in fi.readlines():
            item = json.loads(line)
            ground_truth = item["charge"]
            retrieved_charges = [case['meta']['accusation'][0] for case in item['similar_cases'][:top_k]]
            total_instance+=1
            if ground_truth in retrieved_charges:
                recalled_label+=1
    print(f"recall: {round(recalled_label/total_instance, 4)}")
            
def stat_charge2num(input_path):
    charge2num = {}
    with open(input_path, "r", encoding="utf-8") as fi:
        for line in fi.readlines():
            case = json.loads(line)
            charge2num.setdefault(case["charge"], 0)
            charge2num[case["charge"]]+=1
    sorted_items = sorted(charge2num.items(), key=lambda x:x[1])
    print(f"length:{len(sorted_items)}")
    print(sorted_items)
                
def select_sub_dataset(input_paths, output_path, num=10):
    with open("dataset/cail/charge_labels.txt","r", encoding="utf-8") as fi:
        labels = json.load(fi)
    fo = open(output_path, "w", encoding="utf-8")
    charge2case = {}
    idx = 0
    for path in input_paths:
        with open(path, "r", encoding="utf-8") as fi:
            for line in fi.readlines():
                case = json.loads(line)
                charges = case["meta"]["accusation"]
                if len(charges)>1:
                    continue
                if re.sub(r"[\[\]]", "", charges[0]) not in labels:
                    continue
                charge2case.setdefault(charges[0], [])
                charge2case[charges[0]].append({"id":idx, "charge":charges[0], "fact":case["fact"]})
                idx+=1
    if len(charge2case) == len(labels):
        for charge, cases in charge2case.items():
            random.shuffle(cases)
            if len(cases)<num:
                num = len(cases)
            for case in random.sample(cases, k=num):
                fo.write(json.dumps(case, ensure_ascii=False)+"\n")
        fo.close()
    else:
        print("label 不足")

def get_similar_cases(input_path,num=10):
    output_path = re.sub(".json","with_similar_cases.json", input_path)
    seg = pkuseg.pkuseg()
    print("load bm25 engine...")
    corpus = load_corpus("dataset/cail/rag_corpus.json")
    bm25_eng = BM25Okapi(load_corpus("dataset/cail/rag_corpus.txt"))
    print("processing start...")
    fo = open(output_path, "w", encoding="utf-8")
    with open(input_path, "r", encoding="utf-8") as fi:
        lines = fi.readlines()
        process_bar = tqdm(range(len(lines)))
        for line in lines:
            case = json.loads(line)
            fact = re.sub(r"[\n\r]","",case["fact"][:1000])
            query = seg.cut(fact)
            scores = torch.tensor(bm25_eng.get_scores(query), dtype=torch.float32)
            top_k_idx = torch.topk(scores, k=num)[1].tolist()
            similar_cases = [corpus[i] for i in top_k_idx]
            case["similar_cases"] = similar_cases
            fo.write(json.dumps(case, ensure_ascii=False)+"\n")
            process_bar.update(1)
    print("processing end...")
    fo.close()
    
          
if __name__=="__main__":
    # select_sub_dataset(input_paths=["D:\\dataset_hub\\CAIL2018_ALL_DATA\\final_all_data\\large\\test.json"], 
    #                   output_path="dataset/cail/test_large.json", 
    #                   num=10)
    # get_similar_cases("dataset/cail/testset_small.json")
    recall_of_bm25IR(top_k=6)