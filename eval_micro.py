import json
import os
import re
from collections import defaultdict
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from utils import add_long_tail_label, get_label_mapping, add_long_tail_and_sim_label

# 评估长尾知识的召回率

def eval_zero_shot(test_file_path):
    count = 0
    count_recall = 0
    count_acc = 0
    charge_metrics = defaultdict(lambda: {"true_positives": 0, "false_positives": 0, "false_negatives": 0})
    
    with open(test_file_path, "r", encoding="utf-8") as fi:
        for line in fi.readlines():
            case = json.loads(line)
            ground_truth = case.get("charge")
            exact_match_answer = case["exact_match_answer"]
            exact_and_fuzzy_match_answer = case["exact_and_fuzzy_match_answer"][:1]
            
            if len(exact_match_answer) == 1 and exact_match_answer[0] == ground_truth:
                count_acc += 1
                count_recall += 1
                charge_metrics[ground_truth]["true_positives"] += 1
            else:
                all_ans = []
                if len(exact_match_answer) > 1:
                    all_ans = exact_match_answer
                else:
                    all_ans = exact_and_fuzzy_match_answer
                
                found = False
                for answer in all_ans:
                    if answer == ground_truth:
                        count_recall += 1
                        charge_metrics[ground_truth]["true_positives"] += 1
                        found = True
                        break
                
                if not found:
                    charge_metrics[ground_truth]["false_negatives"] += 1
                
                # For each incorrect prediction, mark false positives for the predicted charge
                for answer in all_ans:
                    if answer != ground_truth:
                        charge_metrics[answer]["false_positives"] += 1
            
            count += 1
    
    # Calculate recall and accuracy
    print(f"Recall: {count_recall / count} | Accuracy: {count_acc / count}")
    
    # Calculate precision, recall, and F1-score for each charge
    for charge, metrics in charge_metrics.items():
        tp = metrics["true_positives"]
        fp = metrics["false_positives"]
        fn = metrics["false_negatives"]
        precision = tp / (tp + fp) if tp + fp > 0 else 0
        recall = tp / (tp + fn) if tp + fn > 0 else 0
        f1 = (2 * precision * recall) / (precision + recall) if precision + recall > 0 else 0
        print(f"Charge: {charge}, Precision: {precision:.2f}, Recall: {recall:.2f}, F1: {f1:.2f}")


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
    with open("./knowledge/similar_crimes.json","r", encoding="utf-8") as fi:
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
    knowledge_path = f"./knowledge/trainset_{model_name}_zero_shot.json"
    test_file_path = f"./outputs/{model_name}/zero_shot/testset_zero_shot.json"
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

import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def print_scores_by_charge(model_name, method, file_name, preds, truths, charges):
    json_file_name = f"./micro_accuracy/{model_name}_{method}_{file_name.split('.')[0]}_metrics.json"
    print(model_name)
    # 打开文件并写入模式设为追加（'a'），这样可以一行一行地写入
    with open(json_file_name, 'a', encoding='utf-8') as f:

        # 遍历每个charge，计算相应的指标
        for charge in charge_labels:
            preds_for_charge = []
            truths_for_charge = []

            # 获取针对该charge的预测和真实值
            for i in range(len(charges)):
                if charges[i] == charge:
                    preds_for_charge.append(preds[i])
                    truths_for_charge.append(truths[i])

            if preds_for_charge and truths_for_charge:
                acc = round(accuracy_score(y_pred=preds_for_charge, y_true=truths_for_charge) * 100, 2)
                prec = round(precision_score(y_pred=preds_for_charge, y_true=truths_for_charge, average='samples', zero_division=0) * 100, 2)
                rec = round(recall_score(y_pred=preds_for_charge, y_true=truths_for_charge, average='samples') * 100, 2)
                f1 = round(f1_score(y_pred=preds_for_charge, y_true=truths_for_charge, average='samples') * 100, 2)

                # 准备要写入的字典数据
                charge_metrics = {
                    "charge": charge,
                    "accuracy": acc,
                    "precision": prec,
                    "recall": rec,
                    "f1_score": f1
                }

                # 将当前 charge 的数据写入文件，每次写入一行
                f.write(json.dumps(charge_metrics, ensure_ascii=False) + "\n")

                # 打印该 charge 的指标
                #print(f"[Charge: {charge}] Acc: {charge_metrics['accuracy']} | Prec: {charge_metrics['precision']} | Rec: {charge_metrics['recall']} | F1: {charge_metrics['f1_score']}")
               
                if(charge_metrics['accuracy']<=20):
                    print(f"'{charge}',")

    
    
def eval_cot_rag(model_name, method):
    assert model_name in ["qwen2-0.5b", "qwen2-1.5b", "qwen2-7b", "baichuan2-13b", "gpt-4o-mini", "gpt-4o-2024-08-06"], "模型名称错误"   
    assert method in ["zero_shot", "zero_shot_cot", "few_shot_cot","rag"], "method 参数错误"
    path_ = os.path.join(base_path, model_name, method)
    for file_name in os.listdir(path_):
        preds_all, ground_truths_all = [], []
        charges_all = []  # 新增的charge列表
        input_path = os.path.join(path_, file_name)
        with open(input_path, "r", encoding="utf-8") as fi:
            for line in fi.readlines():
                case = json.loads(line)
                response = case[responsekey[method]]
                pred, truth = get_perded_labels_wo_options(response, case["charge"])
                preds_all.append(pred)
                ground_truths_all.append(truth)
                charges_all.append(case["charge"])  # 记录charge
                
        preds = preds_all
        ground_truths = ground_truths_all
        print_scores_by_charge(model_name, method, file_name, preds, ground_truths, charges_all)



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
def calculate_avg_accuracy(model_name, method, file_name, low_accuracy_charges, preds, truths, charges):
    total_accuracy = 0
    count = 0

    # 遍历 low_accuracy_charges，计算它们在指定方法下的准确率
    for charge in low_accuracy_charges:
        preds_for_charge = []
        truths_for_charge = []

        # 获取针对该 charge 的预测和真实值
        for i in range(len(charges)):
            if charges[i] == charge:
                preds_for_charge.append(preds[i])
                truths_for_charge.append(truths[i])

        if preds_for_charge and truths_for_charge:
            acc = round(accuracy_score(y_pred=preds_for_charge, y_true=truths_for_charge) * 100, 2)
            total_accuracy += acc
            count += 1

    # 计算平均准确率
    avg_accuracy = total_accuracy / count if count > 0 else 0
    print(f"Model: {model_name} | Method: {method} | Low Accuracy Charges Avg Accuracy: {avg_accuracy}")
    return avg_accuracy
low_accuracy_charges = ['组织、领导、参加黑社会性质组织',
'窝藏、转移、隐瞒毒品、毒赃',
'强迫他人吸毒',
'劫持船只、汽车',
'金融凭证诈骗',
'提供侵入、非法控制计算机信息系统程序、工具',
'过失损坏武器装备、军事设施、军事通信',
'伪造、倒卖伪造的有价票证',
'徇私舞弊不移交刑事案件',
'非法获取国家秘密',
'收买被拐卖的妇女、儿童',
'过失投放危险物质',
'盗窃、侮辱尸体',
'生产、销售伪劣农药、兽药、化肥、种子',
'招收公务员、学生徇私舞弊',
'过失损坏广播电视设施、公用电信设施',
'非法生产、买卖警用装备',
'伪造、变造、买卖武装部队公文、证件、印章',
'窝藏、转移、收购、销售赃物',
'聚众冲击国家机关',
'非法制造、买卖、运输、储存危险物质',
'窃取、收买、非法提供信用卡信息',
'破坏监管秩序',
'帮助毁灭、伪造证据',
'走私武器、弹药',
'非法制造、销售非法制造的注册商标标识',
'利用影响力受贿',
'非法携带枪支、弹药、管制刀具、危险物品危及公共安全',
'隐匿、故意销毁会计凭证、会计帐簿、财务会计报告',
'非法制造、出售非法制造的发票',
'非法生产、销售间谍专用器材',
'引诱、教唆、欺骗他人吸毒',
'动植物检疫徇私舞弊',
'走私国家禁止进出口的货物、物品',
'走私珍贵动物、珍贵动物制品',
'侵占',
'虚报注册资本',
'帮助犯罪分子逃避处罚',
'对单位行贿',
'妨害作证',
'拐骗儿童',
'非法出售发票',
'编造、故意传播虚假恐怖信息',
'过失以危险方法危害公共安全',
'非法进行节育手术',
'介绍贿赂',
'单位受贿',
'制作、复制、出版、贩卖、传播淫秽物品牟利',
'聚众扰乱公共场所秩序、交通秩序',
'伪造、变造金融票证',
'非法收购、运输盗伐、滥伐的林木',
'非法买卖制毒物品',
'出售、购买、运输假币',
'冒充军人招摇撞骗',
'伪造、变造居民身份证',
'非法获取公民个人信息',
'破坏易燃易爆设备',
'对非国家工作人员行贿',
'聚众扰乱社会秩序',
'保险诈骗',
'盗掘古文化遗址、古墓葬',
'持有、使用假币',
'非法猎捕、杀害珍贵、濒危野生动物',
'重大劳动安全事故',
'非法处置查封、扣押、冻结的财产',
'强制猥亵、侮辱妇女',
'非法收购、运输、加工、出售国家重点保护植物、国家重点保护植物制品',
'持有伪造的发票',
'虚开发票',
'拐卖妇女、儿童',
'破坏广播电视设施、公用电信设施',
'妨害信用卡管理',
'走私普通货物、物品',
'单位行贿',
'伪造公司、企业、事业单位、人民团体印章',
'窝藏、包庇',
'非国家工作人员受贿',
'骗取贷款、票据承兑、金融票证',
'伪造、变造、买卖国家机关公文、证件、印章',
'非法制造、买卖、运输、邮寄、储存枪支、弹药、爆炸物',
'行贿',
'虚开增值税专用发票、用于骗取出口退税、抵扣税款发票',
'掩饰、隐瞒犯罪所得、犯罪所得收益',
'非法持有、私藏枪支、弹药',
'走私、贩卖、运输、制造毒品'



]

def eval_option(model_name, method):
    assert model_name in ["qwen2-0.5b", "qwen2-1.5b", "qwen2-7b", "baichuan2-13b", "gpt-4o-mini", "gpt-4o-2024-08-06"], "模型名称错误"   
    assert method in ["options", "options_random", "options_case", "options_desc", "options_desc_case"], f"{method} 参数错误"
    path_ = os.path.join(base_path, model_name, method)
    
    for file_name in os.listdir(path_):
        preds_all, truths_all = [], []
        preds_head, truths_head = [], []
        preds_tail, truths_tail = [], []
        charges_all = []  # 存储每个case的charge
        
        input_path = os.path.join(path_, file_name)
        with open(input_path, "r", encoding="utf-8") as fi:
            for line in fi.readlines():
                case = json.loads(line)
                response = case[responsekey[method]]
                pred, truth = get_perded_labels_w_options(response, case["charge"], options=case["options"])
                
                preds_all.append(pred)
                truths_all.append(truth)
                charges_all.append(case["charge"])  # 添加charge信息
                

        # 传递charge信息用于每个罪名的评估
        preds = preds_all
        truths = truths_all
        charges = charges_all
        calculate_avg_accuracy(model_name, method, file_name, low_accuracy_charges, preds, truths, charges)
            


def conduct_eval_2():
    for mn in ["qwen2-1.5b","gpt-4o-2024-08-06","baichuan2-13b","qwen2-7b"]: #  "qwen2-0.5b",,"baichuan2-13b" , "qwen2-7b","baichuan2-13b","gpt-4o-2024-08-06"
        for method in ["options_desc_case"]:
            eval_option(mn, method)
        # eval_option(mn,"options_desc_case")
    # model_names = ["qwen2-0.5b", "qwen2-1.5b", "qwen2-7b", "baichuan2-13b"]
    # model_names = ["gpt-4o-2024-08-06"] # ,"gpt-4o-2024-08-06"
    # for mn in model_names: #
    #     eval_option(mn, "options_desc_case")

if __name__=="__main__":
    conduct_eval_2()
62.0,69.8,78.67,72.56,33.0,40.45,50.0,43.23,35.36,42.85,52.34,45.62