from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json, re
import numpy as np
import pandas as pd
import tqdm
from tqdm import tqdm
from collections import defaultdict
from transformers.generation.utils import GenerationConfig
device = "cuda" # the device to load the model onto

model_path = {
    "qwen2-0.5b":"/home/kai/workspace/modelfiles/qwen2-0.5b",
    "qwen2-1.5b":"/home/kai/workspace/modelfiles/qwen2-1.5b",
    "qwen2-7b":"/home/kai/workspace/modelfiles/qwen2-7b",
    "baichuan2-13b":"/home/kai/workspace/modelfiles/baichuan2-13b"
}

lt_kwg_path = {
    "qwen2-0.5b":"/home/kai/workspace/LongTail/konwledge/answers_xx_100_qwen-0.5b.json",
    "qwen2-1.5b":"/home/kai/workspace/LongTail/konwledge/answers_xx_100_qwen-1.5b.json",
    "qwen2-7b":"/home/kai/workspace/LongTail/konwledge/answers_xx_100_qwen-7b.json",
    "baichuan2-13b":"/home/kai/workspace/LongTail/konwledge/answers_xx_100_baichuan-13b.json",
    "glm-4":"./longtail_knowledge/glm-4_trainset.json",
    "gpt-4":"./longtail_knowledge/trainset_for_llm_zero_shot_gpt4.json",
    "gpt-4-mini":"./longtail_knowledge/gpt-4-mini_trainset.json",
}
charge_labels_sorted = ['非法收购、运输、加工、出售国家重点保护植物、国家重点保护植物制品', '非法携带枪支、弹药、管制刀具、危险物品危及公共安全', '非法制造、买卖、运输、邮寄、储存枪支、弹药、爆炸物', '虚开增值税专用发票、用于骗取出口退税、抵扣税款发票', '隐匿、故意销毁会计凭证、会计帐簿、财务会计报告', '提供侵入、非法控制计算机信息系统程序、工具', '伪造、变造、买卖武装部队公文、证件、印章', '制作、复制、出版、贩卖、传播淫秽物品牟利', '伪造、变造、买卖国家机关公文、证件、印章', '伪造公司、企业、事业单位、人民团体印章', '过失损坏武器装备、军事设施、军事通信', '生产、销售伪劣农药、兽药、化肥、种子', '非法制造、销售非法制造的注册商标标识', '过失损坏广播电视设施、公用电信设施', '非法制造、买卖、运输、储存危险物质', '非法猎捕、杀害珍贵、濒危野生动物', '掩饰、隐瞒犯罪所得、犯罪所得收益', '组织、领导、参加黑社会性质组织', '窃取、收买、非法提供信用卡信息', '走私国家禁止进出口的货物、物品', '聚众扰乱公共场所秩序、交通秩序', '非法收购、运输盗伐、滥伐的林木', '非法处置查封、扣押、冻结的财产', '破坏广播电视设施、公用电信设施', '非法采伐、毁坏国家重点保护植物', '生产、销售不符合安全标准的食品', '非法制造、出售非法制造的发票', '骗取贷款、票据承兑、金融票证', '窝藏、转移、隐瞒毒品、毒赃', '窝藏、转移、收购、销售赃物', '非法生产、销售间谍专用器材', '走私珍贵动物、珍贵动物制品', '编造、故意传播虚假恐怖信息', '过失以危险方法危害公共安全', '走私、贩卖、运输、制造毒品', '伪造、倒卖伪造的有价票证', '招收公务员、学生徇私舞弊', '引诱、教唆、欺骗他人吸毒', '非法转让、倒卖土地使用权', '生产、销售有毒、有害食品', '非法持有、私藏枪支、弹药', '徇私舞弊不移交刑事案件', '收买被拐卖的妇女、儿童', '非法生产、买卖警用装备', '盗掘古文化遗址、古墓葬', '扰乱无线电通讯管理秩序', '以危险方法危害公共安全', '销售假冒注册商标的商品', '帮助犯罪分子逃避处罚', '出售、购买、运输假币', '伪造、变造居民身份证', '非法获取公民个人信息', '对非国家工作人员行贿', '帮助毁灭、伪造证据', '破坏计算机信息系统', '动植物检疫徇私舞弊', '伪造、变造金融票证', '强制猥亵、侮辱妇女', '走私普通货物、物品', '生产、销售伪劣产品', '组织、领导传销活动', '非国家工作人员受贿', '非法种植毒品原植物', '拒不执行判决、裁定', '非法获取国家秘密', '过失投放危险物质', '聚众冲击国家机关', '非法进行节育手术', '非法买卖制毒物品', '冒充军人招摇撞骗', '破坏易燃易爆设备', '聚众扰乱社会秩序', '重大劳动安全事故', '拒不支付劳动报酬', '非法吸收公众存款', '劫持船只、汽车', '倒卖车票、船票', '盗窃、侮辱尸体', '走私武器、弹药', '利用影响力受贿', '持有、使用假币', '持有伪造的发票', '拐卖妇女、儿童', '非法捕捞水产品', '妨害信用卡管理', '非法占用农用地', '生产、销售假药', '强迫他人吸毒', '打击报复证人', '虐待被监管人', '破坏交通工具', '金融凭证诈骗', '挪用特定款物', '传授犯罪方法', '破坏交通设施', '危险物品肇事', '破坏监管秩序', '非法组织卖血', '私分国有资产', '虚报注册资本', '非法出售发票', '违法发放贷款', '投放危险物质', '破坏电力设备', '破坏生产经营', '过失致人重伤', '非法侵入住宅', '重大责任事故', '假冒注册商标', '过失致人死亡', '故意毁坏财物', '非法持有毒品', '容留他人吸毒', '对单位行贿', '侵犯著作权', '窝藏、包庇', '信用卡诈骗', '高利转贷', '倒卖文物', '聚众哄抢', '伪造货币', '强迫劳动', '走私废物', '传播性病', '徇私枉法', '妨害作证', '拐骗儿童', '串通投标', '介绍贿赂', '单位受贿', '诬告陷害', '强迫交易', '票据诈骗', '贷款诈骗', '保险诈骗', '集资诈骗', '虚开发票', '非法采矿', '猥亵儿童', '招摇撞骗', '滥用职权', '单位行贿', '非法狩猎', '玩忽职守', '污染环境', '挪用公款', '非法行医', '非法经营', '盗伐林木', '挪用资金', '聚众斗殴', '敲诈勒索', '合同诈骗', '故意杀人', '职务侵占', '非法拘禁', '滥伐林木', '开设赌场', '妨害公务', '寻衅滋事', '交通肇事', '故意伤害', '危险驾驶', '走私', '洗钱', '诽谤', '虐待', '侮辱', '脱逃', '侵占', '伪证', '遗弃', '逃税', '爆炸', '绑架', '重婚', '放火', '行贿', '强奸', '失火', '贪污', '赌博', '抢夺', '受贿', '抢劫', '诈骗', '盗窃']
class CrimeDataProcessor:
    def __init__(self, answer_data, query, n, option_list):
        self.answer_data = answer_data
        self.query = query
        self.n = n
        self.option_list = option_list
        self.count()
        self.get_top_n_counts()
        
    def count(self):
        indict = [f"{crime}" for crime in self.option_list]
        count_matrix = {answer: {refr: 0 for refr in indict} for answer in indict}

        for item in self.answer_data:
            refr = item['charge'].replace("罪名:", "").replace("<eoa>", "").strip()
            answer = item['response'].replace("罪名:", "").replace("<eoa>", "").strip()

            if refr in indict and answer in indict:
                pattern = rf'(?<!\d){re.escape(answer)}'
                if re.search(pattern, answer):
                    count_matrix[answer][refr] += 1

        sampled_count = defaultdict(dict)

        for refr in indict:
            counts = np.array([count_matrix[answer][refr] for answer in indict])
            max_count = np.max(counts)
            exp_counts = np.exp(counts - max_count)
            total_sum = np.sum(exp_counts)

            for answer in indict:
                exp_count = np.exp(count_matrix[answer][refr] - max_count)
                sampled_count[answer][refr] = exp_count / total_sum
                sampled_count[answer][answer] = 0

        self.count_matrix = sampled_count
    

    def get_top_n_counts(self):
        if self.count_matrix is None:
            raise ValueError("Count matrix has not been calculated. Please run count() first.")

        top_counts = {}

        for answer in self.count_matrix:
            sorted_items = sorted(self.count_matrix[answer].items(), key=lambda x: x[1], reverse=True)
            top_counts[answer] = [item[0] for item in sorted_items[:self.n]]

        self.top_counts = top_counts

    def process_data(self):
        result = []
        if self.top_counts is not None:
            assert self.query in self.option_list #, f"answer:{answer}, item:{item}"
            for i in range(self.n):
                result.append(str(self.top_counts[str(self.query)][i]))

        return result

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
        generated_ids = model.generate(
            model_inputs.input_ids,
            max_new_tokens=512
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response

def load_corpus(path):
    data = []
    with open(path,"r",encoding="utf-8") as f:
        for line in f.readlines():
            data.append(json.loads(line))
    return data

def load_json(path):
    data = []
    with open(path,"r",encoding="utf-8") as f:
        data = json.load(f)
    return data

def get_count_matrix(data,charge_labels):

    indict = [f"{crime}" for crime in charge_labels]
    indict_sorted = [f"{crime}" for crime in charge_labels_sorted]
    length = len(indict)
    
    count_matrix = pd.DataFrame(index=indict,columns=indict).fillna(0)
    for predict_label, true_label_dict in data.items():
        for true_label, count in true_label_dict.items():
            if predict_label == "走私":
                count=0
            count_matrix.at[predict_label,true_label] += count

    # for item in data:
    #     refr = item.get("charge") or str(item.get("refr")).replace("罪名:","").replace("罪名：","")
    #     response = item.get('response', '').strip()
    #     answer_fallback = item.get('answer', '').strip()

    #     if response:
    #         answer = response.replace("罪名:", "").replace("<eoa>", "").replace("A.", "").replace("B.", "").replace("C.", "").replace("D.", "").strip()
    #     else:
    #         answer = answer_fallback.replace("罪名:", "").replace("<eoa>", "").replace("A.", "").replace("B.", "").replace("C.", "").replace("D.", "").strip()

    #     for pattern in indict_sorted:
    #         if pattern == "走私":
    #             continue
    #         if re.search(pattern, answer) and answer != refr and refr in charge_labels:
    #             #print(f"Match found: pattern '{pattern}' in answer '{answer}'")
    #             answer = pattern
    #             count_matrix.at[answer,refr] += 1
    #             continue
    for crime in indict_sorted:
        count_matrix.at[crime,crime] =0
    return count_matrix




if __name__=="__main__":
    pass