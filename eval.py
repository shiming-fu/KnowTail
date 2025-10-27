import json
import os
import re
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from utils import add_long_tail_label, get_label_mapping, add_long_tail_and_sim_label

# 评估长尾知识的召回率

def eval_zero_shot(test_file_path):
    count=0
    count_recall = 0
    count_acc = 0
    with open(test_file_path, "r", encoding="utf-8") as fi:
        for line in fi.readlines():
            case = json.loads(line)
            ground_truth = case.get("charge")
            exact_match_answer = case["exact_match_answer"]
            exact_and_fuzzy_match_answer = case["exact_and_fuzzy_match_answer"][:1]
            if len(exact_match_answer)==1 and exact_match_answer[0] == ground_truth:
                count_acc+=1
                count_recall+=1
            else:
                all_ans = []
                if len(exact_match_answer)>1:
                    all_ans = exact_match_answer
                else:
                    all_ans = exact_and_fuzzy_match_answer
                for answer in all_ans:
                    if answer == ground_truth:
                        count_recall+=1
                        break
            count+=1
    print(f"Recall:{count_recall/count} | Accuracy: {count_acc/count}")

def eval_zero_shot_with_long_tail_label(knowledge_path, test_file_path, label_num):
    count=0
    count_recall = 0
    pl2tl, _ = get_label_mapping(knowledge_path)
    with open(test_file_path, "r", encoding="utf-8") as fi:
        for line in fi.readlines():
            case = json.loads(line)
            ground_truth = case.get("charge")
            if len(case["exact_match_answer"])!=0:
                label = []
                label.extend(case["exact_match_answer"])
                label.extend(case["exact_and_fuzzy_match_answer"][:1])
                answers = add_long_tail_label(label, pl2tl, label_num=label_num)
            else:
                answers = add_long_tail_label(case["exact_and_fuzzy_match_answer"][:2], pl2tl, label_num=label_num)
            if ground_truth in answers:
                count_recall+=1
            count+=1
    print(f"Recall:{count_recall/count}")

def eval_zero_shot_with_long_tail_and_sim_label(knowledge_path, test_file_path, label_num):
    count=0
    count_recall = 0
    pl2tl, tl2pl = get_label_mapping(knowledge_path)
    with open("./longtail/knowledge/similar_crimes.json","r", encoding="utf-8") as fi:
        sim_crimes = json.load(fi)
    with open(test_file_path, "r", encoding="utf-8") as fi:
        for line in fi.readlines():
            case = json.loads(line)
            ground_truth = case.get("charge")
            if len(case["exact_match_answer"])!=0:
                label = []
                label.extend(case["exact_match_answer"])
                label.extend(case["exact_and_fuzzy_match_answer"][:1])
                answers = add_long_tail_and_sim_label(label, pl2tl, sim_crimes, label_num=label_num)
            else:
                answers = add_long_tail_and_sim_label(case["exact_and_fuzzy_match_answer"][:2], pl2tl, sim_crimes, label_num=label_num)
            if ground_truth in answers:
                count_recall+=1
            count+=1
    print(f"Recall:{count_recall/count}")    
    
def conduct_eval_1():
    model_name = "baichuan2-13b"
    knowledge_path = f"./longtail/knowledge/trainset_{model_name}_zero_shot.json"
    test_file_path = f"./outputs/{model_name}/testset_zero_shot.json"
    eval_zero_shot(test_file_path=test_file_path)
    eval_zero_shot_with_long_tail_label(knowledge_path=knowledge_path, 
                                        test_file_path=test_file_path, 
                                        label_num=5)
    eval_zero_shot_with_long_tail_and_sim_label(knowledge_path=knowledge_path, 
                                        test_file_path=test_file_path, 
                                        label_num=5)


# 评估baseline和ours methods

with open("dataset/origin/charge_labels.txt", "r", encoding="utf-8") as fi:
    charge_labels = json.load(fi)
head_labels = charge_labels[-15:]
tail_labels = charge_labels[:-15]
base_path = f"outputs"
responsekey = {
    "zero_shot":"response",
    "zero_shot_cot":"zero_cot_response",
    "few_shot_cot":"few_cot_response",
    "options_random":"options_random_response",
    "options":"lt_options_response",
    "options_case":"lt_options_case_response", 
    "options_desc":"lt_options_desc_response",
    "options_desc_case":"lt_options_desc_case_response",
    "rag":"rag_response"
}

def get_perded_labels_wo_options(response, true_label):
    pred = [0]*len(charge_labels)
    truth = [0]*len(charge_labels)
    truth[charge_labels.index(true_label)]=1
    pred_labels = []
    sorted_charge_labels = sorted(charge_labels, key=lambda x:len(x), reverse=True)
    for c in sorted_charge_labels:
        if c in response:
            pred_labels.append(c)
            response = re.sub(c, "", response)
    for p in pred_labels:
        pred[charge_labels.index(p)]=1
    return pred, truth

def print_scores(model_name, 
                 method, 
                 file_name,  
                 preds,  
                 truths):
    # All
    acc_all = round(accuracy_score(y_pred=preds[0], y_true=truths[0])*100,2)
    prec_all = round(precision_score(y_pred=preds[0], y_true=truths[0], average='samples')*100,2)
    rec_all = round(recall_score(y_pred=preds[0], y_true=truths[0], average='samples')*100,2)
    f1_all = round(f1_score(y_pred=preds[0], y_true=truths[0], average='samples')*100,2)
    # Head
    acc_head = round(accuracy_score(y_pred=preds[1], y_true=truths[1])*100,2)
    prec_head = round(precision_score(y_pred=preds[1], y_true=truths[1], average='samples')*100,2)
    rec_head = round(recall_score(y_pred=preds[1], y_true=truths[1], average='samples')*100,2)
    f1_head = round(f1_score(y_pred=preds[1], y_true=truths[1], average='samples')*100,2)
    # Tail
    acc_tail = round(accuracy_score(y_pred=preds[2], y_true=truths[2])*100,2)
    prec_tail = round(precision_score(y_pred=preds[2], y_true=truths[2], average='samples')*100,2)
    rec_tail = round(recall_score(y_pred=preds[2], y_true=truths[2], average='samples')*100,2)
    f1_tail = round(f1_score(y_pred=preds[2], y_true=truths[2], average='samples')*100,2)
    print(f"\n=====> [{model_name} | {method} | {file_name}] <=====\n")
    print(f"[Head] Acc: {acc_head} | Prec: {prec_head} | Rec: {rec_head} | F1: {f1_head}")
    print(f"[Tail] Acc: {acc_tail} | Prec: {prec_tail} | Rec: {rec_tail} | F1: {f1_tail}")
    print(f"[ALL ] Acc: {acc_all} | Prec: {prec_all} | Rec: {rec_all} | F1: {f1_all}")
    print(",".join([str(acc_head), str(prec_head),str(rec_head),str(f1_head),str(acc_tail), str(prec_tail),str(rec_tail),str(f1_tail),str(acc_all), str(prec_all), str(rec_all), str(f1_all)]))
    # print(",".join([])+"")
    # print(",".join([])+"")
    
    
def eval_cot_rag(model_name, method):
    # zero/few/rag shot CoT
    assert model_name in ["qwen2-0.5b", "qwen2-1.5b", "qwen2-7b", "baichuan2-13b", "gpt-4o-mini", "gpt-4o-2024-08-06"], "模型名称错误"   
    assert method in ["zero_shot", "zero_shot_cot", "few_shot_cot","rag"], "method 参数错误"
    path_ = os.path.join(base_path, model_name, method)
    for file_name in os.listdir(path_):
        preds_all, ground_truths_all = [], []
        preds_head, ground_truths_head = [], []
        preds_tail, ground_truths_tail = [], []
        input_path = os.path.join(path_, file_name)
        with open(input_path, "r", encoding="utf-8") as fi:
            for line in fi.readlines():
                case = json.loads(line)
                response = case[responsekey[method]]
                pred, truth = get_perded_labels_wo_options(response,case["charge"])
                preds_all.append(pred)
                ground_truths_all.append(truth)
                if case["charge"] in head_labels:
                    preds_head.append(pred)
                    ground_truths_head.append(truth)
                elif case["charge"] in tail_labels:
                    preds_tail.append(pred)
                    ground_truths_tail.append(truth)
        preds = [preds_all, preds_head, preds_tail]
        ground_truths = [ground_truths_all, ground_truths_head, ground_truths_tail]
        print_scores(model_name, method, file_name, preds, ground_truths)


def get_perded_labels_w_options(response, charge, options):
    index2charge = dict([tuple(item.split(".")) for item in options])
    sorted_charge_labels = sorted(charge_labels, key=lambda x:len(x), reverse=True)
    pred = [0]*len(charge_labels)
    truth = [0]*len(charge_labels)
    truth[charge_labels.index(charge)]=1
    # get predicted charges or indexs
    pred_labels = []
    pred_indexs = []
    for c in sorted_charge_labels:
        if c in response:
            pred_labels.append(c)
            response = re.sub(c, "", response)
    for i in ["A", "B", "C", "D", "E", "F", "G"]:
        if i in response:
            pred_indexs.append(i)
    # get predictions
    for pi in pred_indexs:
        if pi in index2charge:
            pred[charge_labels.index(index2charge[pi][:-1])]=1
    for pl in pred_labels:
        pred[charge_labels.index(pl)]=1
    
    return pred, truth

def eval_option(model_name, method):
    assert model_name in ["qwen2-0.5b", "qwen2-1.5b", "qwen2-7b", "baichuan2-13b", "gpt-4o-mini", "gpt-4o-2024-08-06"], "模型名称错误"   
    assert method in ["options", "options_random", "options_case", "options_desc", "options_desc_case"], f"{method} 参数错误"
    path_ = os.path.join(base_path, model_name, method)
    for file_name in os.listdir(path_):
        preds_all, ground_truths_all = [], []
        preds_head, ground_truths_head = [], []
        preds_tail, ground_truths_tail = [], []
        input_path = os.path.join(path_, file_name)
        with open(input_path, "r", encoding="utf-8") as fi:
            for line in fi.readlines():
                case = json.loads(line)
                response = case[responsekey[method]]
                pred, truth = get_perded_labels_w_options(response,case["charge"], options=case["options"])
                preds_all.append(pred)
                ground_truths_all.append(truth)
                if case["charge"] in head_labels:
                    preds_head.append(pred)
                    ground_truths_head.append(truth)
                elif case["charge"] in tail_labels:
                    preds_tail.append(pred)
                    ground_truths_tail.append(truth)
        preds = [preds_all, preds_head, preds_tail]
        ground_truths = [ground_truths_all, ground_truths_head, ground_truths_tail]                
        print_scores(model_name, method, file_name, preds, ground_truths)
            
                    
def conduct_eval_2():
    for mn in ["gpt-4o-2024-08-06"]: #  "qwen2-0.5b", "qwen2-1.5b", "qwen2-7b","baichuan2-13b",
        eval_cot_rag(mn, "few_shot_cot")
    # model_names = ["qwen2-0.5b", "qwen2-1.5b", "qwen2-7b", "baichuan2-13b"]
    # model_names = ["gpt-4o-2024-08-06"] # ,"gpt-4o-2024-08-06"
    # for mn in model_names: #
    #     eval_option(mn, "options_desc_case")

if __name__=="__main__":
    conduct_eval_1()
62.0,69.8,78.67,72.56,33.0,40.45,50.0,43.23,35.36,42.85,52.34,45.62