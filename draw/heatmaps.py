# encoding:utf-8
import sys
sys.path.append("D:/longtail-cail-1008")
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.font_manager import FontProperties
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from utils import get_label_mapping
import plot_utils as utils
import numpy as np
import pandas as pd
import sys
from matplotlib.patches import ConnectionPatch
pd.set_option('future.no_silent_downcasting', True)

model_path = {
    "GPT-4o":"knowledge/trainset_gpt-4o-2024-08-06_zero_shot.json",
    "GPT-4o-mini":"knowledge/trainset_gpt-4o-mini_zero_shot.json",
    "GPT-4o-mini-rag":"",
    "GPT-4o-mini-ours":"",
    "GPT-4-mini-ours-with-desc-4-choices":"",
    "Qwen2-0.5B":"knowledge/trainset_qwen2-0.5b_zero_shot.json",
    "Qwen2-1.5B":"knowledge/trainset_qwen2-1.5b_zero_shot.json",
    "Qwen2-7B":"knowledge/trainset_qwen2-7b_zero_shot.json",
    "Baichuan2-13B":"knowledge/trainset_baichuan2-13b_zero_shot.json",
}

charge_labels = ["组织、领导、参加黑社会性质组织", "窝藏、转移、隐瞒毒品、毒赃", "走私", "强迫他人吸毒", "高利转贷", "打击报复证人", "倒卖文物", "劫持船只、汽车", "虐待被监管人", "聚众哄抢", "破坏交通工具", "金融凭证诈骗", "洗钱", "倒卖车票、船票", "伪造货币", "诽谤", "提供侵入、非法控制计算机信息系统程序、工具", "过失损坏武器装备、军事设施、军事通信", "伪造、倒卖伪造的有价票证", "挪用特定款物", "强迫劳动", "徇私舞弊不移交刑事案件", "传授犯罪方法", "虐待", "非法获取国家秘密", "收买被拐卖的妇女、儿童", "过失投放危险物质", "走私废物", "盗窃、侮辱尸体", "生产、销售伪劣农药、兽药、化肥、种子", "招收公务员、学生徇私舞弊", "过失损坏广播电视设施、公用电信设施", "非法生产、买卖警用装备", "伪造、变造、买卖武装部队公文、证件、印章", "窝藏、转移、收购、销售赃物", "聚众冲击国家机关", "破坏交通设施", "危险物品肇事", "非法制造、买卖、运输、储存危险物质", "窃取、收买、非法提供信用卡信息", "破坏监管秩序", "帮助毁灭、伪造证据", "非法组织卖血", "走私武器、弹药", "非法制造、销售非法制造的注册商标标识", "利用影响力受贿", "传播性病", "侮辱", "非法携带枪支、弹药、管制刀具、危险物品危及公共安全", "隐匿、故意销毁会计凭证、会计帐簿、财务会计报告", "私分国有资产", "非法制造、出售非法制造的发票", "非法生产、销售间谍专用器材", "破坏计算机信息系统", "引诱、教唆、欺骗他人吸毒", "动植物检疫徇私舞弊", "走私国家禁止进出口的货物、物品", "脱逃", "徇私枉法", "走私珍贵动物、珍贵动物制品", "侵占", "虚报注册资本", "帮助犯罪分子逃避处罚", "伪证", "遗弃", "对单位行贿", "妨害作证", "拐骗儿童", "非法出售发票", "串通投标", "编造、故意传播虚假恐怖信息", "过失以危险方法危害公共安全", "逃税", "非法进行节育手术", "介绍贿赂", "单位受贿", "制作、复制、出版、贩卖、传播淫秽物品牟利", "聚众扰乱公共场所秩序、交通秩序", "诬告陷害", "伪造、变造金融票证", "非法收购、运输盗伐、滥伐的林木", "强迫交易", "非法买卖制毒物品", "出售、购买、运输假币", "票据诈骗", "贷款诈骗", "冒充军人招摇撞骗", "伪造、变造居民身份证", "违法发放贷款", "非法获取公民个人信息", "破坏易燃易爆设备", "对非国家工作人员行贿", "聚众扰乱社会秩序", "保险诈骗", "爆炸", "盗掘古文化遗址、古墓葬", "非法转让、倒卖土地使用权", "持有、使用假币", "非法猎捕、杀害珍贵、濒危野生动物", "投放危险物质", "侵犯著作权", "重大劳动安全事故", "非法处置查封、扣押、冻结的财产", "强制猥亵、侮辱妇女", "扰乱无线电通讯管理秩序", "非法收购、运输、加工、出售国家重点保护植物、国家重点保护植物制品", "破坏电力设备", "绑架", "持有伪造的发票", "重婚", "集资诈骗", "虚开发票", "拐卖妇女、儿童", "破坏广播电视设施、公用电信设施", "破坏生产经营", "非法采矿", "猥亵儿童", "非法捕捞水产品", "妨害信用卡管理", "以危险方法危害公共安全", "走私普通货物、物品", "招摇撞骗", "过失致人重伤", "生产、销售伪劣产品", "滥用职权", "组织、领导传销活动", "单位行贿", "非法侵入住宅", "非法狩猎", "伪造公司、企业、事业单位、人民团体印章", "非法采伐、毁坏国家重点保护植物", "窝藏、包庇", "非国家工作人员受贿", "骗取贷款、票据承兑、金融票证", "玩忽职守", "重大责任事故", "假冒注册商标", "销售假冒注册商标的商品", "污染环境", "非法种植毒品原植物", "拒不支付劳动报酬", "伪造、变造、买卖国家机关公文、证件、印章", "挪用公款", "非法行医", "非法经营", "盗伐林木", "挪用资金", "生产、销售不符合安全标准的食品", "非法制造、买卖、运输、邮寄、储存枪支、弹药、爆炸物", "放火", "行贿", "拒不执行判决、裁定", "聚众斗殴", "非法吸收公众存款", "虚开增值税专用发票、用于骗取出口退税、抵扣税款发票", "强奸", "生产、销售有毒、有害食品", "非法占用农用地", "失火", "过失致人死亡", "敲诈勒索", "贪污", "合同诈骗", "生产、销售假药", "故意杀人", "职务侵占", "非法拘禁", "赌博", "抢夺", "故意毁坏财物", "受贿", "滥伐林木", "掩饰、隐瞒犯罪所得、犯罪所得收益", "开设赌场", "非法持有毒品", "非法持有、私藏枪支、弹药", "妨害公务", "抢劫", "信用卡诈骗", "寻衅滋事", "诈骗", "容留他人吸毒", "走私、贩卖、运输、制造毒品", "交通肇事", "故意伤害", "危险驾驶", "盗窃"]
charge_labels_sorted = ['非法收购、运输、加工、出售国家重点保护植物、国家重点保护植物制品', '非法携带枪支、弹药、管制刀具、危险物品危及公共安全', '非法制造、买卖、运输、邮寄、储存枪支、弹药、爆炸物', '虚开增值税专用发票、用于骗取出口退税、抵扣税款发票', '隐匿、故意销毁会计凭证、会计帐簿、财务会计报告', '提供侵入、非法控制计算机信息系统程序、工具', '伪造、变造、买卖武装部队公文、证件、印章', '制作、复制、出版、贩卖、传播淫秽物品牟利', '伪造、变造、买卖国家机关公文、证件、印章', '伪造公司、企业、事业单位、人民团体印章', '过失损坏武器装备、军事设施、军事通信', '生产、销售伪劣农药、兽药、化肥、种子', '非法制造、销售非法制造的注册商标标识', '过失损坏广播电视设施、公用电信设施', '非法制造、买卖、运输、储存危险物质', '非法猎捕、杀害珍贵、濒危野生动物', '掩饰、隐瞒犯罪所得、犯罪所得收益', '组织、领导、参加黑社会性质组织', '窃取、收买、非法提供信用卡信息', '走私国家禁止进出口的货物、物品', '聚众扰乱公共场所秩序、交通秩序', '非法收购、运输盗伐、滥伐的林木', '非法处置查封、扣押、冻结的财产', '破坏广播电视设施、公用电信设施', '非法采伐、毁坏国家重点保护植物', '生产、销售不符合安全标准的食品', '非法制造、出售非法制造的发票', '骗取贷款、票据承兑、金融票证', '窝藏、转移、隐瞒毒品、毒赃', '窝藏、转移、收购、销售赃物', '非法生产、销售间谍专用器材', '走私珍贵动物、珍贵动物制品', '编造、故意传播虚假恐怖信息', '过失以危险方法危害公共安全', '走私、贩卖、运输、制造毒品', '伪造、倒卖伪造的有价票证', '招收公务员、学生徇私舞弊', '引诱、教唆、欺骗他人吸毒', '非法转让、倒卖土地使用权', '生产、销售有毒、有害食品', '非法持有、私藏枪支、弹药', '徇私舞弊不移交刑事案件', '收买被拐卖的妇女、儿童', '非法生产、买卖警用装备', '盗掘古文化遗址、古墓葬', '扰乱无线电通讯管理秩序', '以危险方法危害公共安全', '销售假冒注册商标的商品', '帮助犯罪分子逃避处罚', '出售、购买、运输假币', '伪造、变造居民身份证', '非法获取公民个人信息', '对非国家工作人员行贿', '帮助毁灭、伪造证据', '破坏计算机信息系统', '动植物检疫徇私舞弊', '伪造、变造金融票证', '强制猥亵、侮辱妇女', '走私普通货物、物品', '生产、销售伪劣产品', '组织、领导传销活动', '非国家工作人员受贿', '非法种植毒品原植物', '拒不执行判决、裁定', '非法获取国家秘密', '过失投放危险物质', '聚众冲击国家机关', '非法进行节育手术', '非法买卖制毒物品', '冒充军人招摇撞骗', '破坏易燃易爆设备', '聚众扰乱社会秩序', '重大劳动安全事故', '拒不支付劳动报酬', '非法吸收公众存款', '劫持船只、汽车', '倒卖车票、船票', '盗窃、侮辱尸体', '走私武器、弹药', '利用影响力受贿', '持有、使用假币', '持有伪造的发票', '拐卖妇女、儿童', '非法捕捞水产品', '妨害信用卡管理', '非法占用农用地', '生产、销售假药', '强迫他人吸毒', '打击报复证人', '虐待被监管人', '破坏交通工具', '金融凭证诈骗', '挪用特定款物', '传授犯罪方法', '破坏交通设施', '危险物品肇事', '破坏监管秩序', '非法组织卖血', '私分国有资产', '虚报注册资本', '非法出售发票', '违法发放贷款', '投放危险物质', '破坏电力设备', '破坏生产经营', '过失致人重伤', '非法侵入住宅', '重大责任事故', '假冒注册商标', '过失致人死亡', '故意毁坏财物', '非法持有毒品', '容留他人吸毒', '对单位行贿', '侵犯著作权', '窝藏、包庇', '信用卡诈骗', '高利转贷', '倒卖文物', '聚众哄抢', '伪造货币', '强迫劳动', '走私废物', '传播性病', '徇私枉法', '妨害作证', '拐骗儿童', '串通投标', '介绍贿赂', '单位受贿', '诬告陷害', '强迫交易', '票据诈骗', '贷款诈骗', '保险诈骗', '集资诈骗', '虚开发票', '非法采矿', '猥亵儿童', '招摇撞骗', '滥用职权', '单位行贿', '非法狩猎', '玩忽职守', '污染环境', '挪用公款', '非法行医', '非法经营', '盗伐林木', '挪用资金', '聚众斗殴', '敲诈勒索', '合同诈骗', '故意杀人', '职务侵占', '非法拘禁', '滥伐林木', '开设赌场', '妨害公务', '寻衅滋事', '交通肇事', '故意伤害', '危险驾驶', '走私', '洗钱', '诽谤', '虐待', '侮辱', '脱逃', '侵占', '伪证', '遗弃', '逃税', '爆炸', '绑架', '重婚', '放火', '行贿', '强奸', '失火', '贪污', '赌博', '抢夺', '受贿', '抢劫', '诈骗', '盗窃']
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('max_colwidth',100)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['SimSun']

# 创建一个FontProperties对象
font_roman = FontProperties(family='Times New Roman', style='normal', weight='regular', size='medium')

def plot_multiple_heatmaps(model_names,data):
    fig, ((ax1, ax2,ax3),(ax4,ax5,ax6)) = plt.subplots(2, 3)  
    axes = [ax1, ax2,ax3,ax4,ax5,ax6] 
    # fig, ax1 = plt.subplots(1, 1)  
    # axes = [ax1] 
    
    for i, model in enumerate(model_names):
        count_matrix = utils.get_count_matrix(data[i], charge_labels)
        mask = np.ones_like(count_matrix, dtype=bool)
        label_to_index = {label: idx for idx, label in enumerate(charge_labels)}
        ax = axes[i]

        for label in charge_labels:
            idx = label_to_index[label]
            mask[:] = True
            mask[idx, idx] = False
        count_matrix = count_matrix.astype(int)
        sns.heatmap(count_matrix, annot=False,  mask=mask,cmap='Reds', fmt='.2f', cbar=True, ax=ax,vmin=0, vmax=count_matrix.max().max())

        # for label in charge_labels:
        #     idx = label_to_index[label]
        #     ax.add_patch(plt.Rectangle((idx, idx), 1, 1, fill=True, color='lightgreen'))

        # 绘制圆形表示值的大小
        for i in range(count_matrix.shape[0]):
            for j in range(count_matrix.shape[1]):
                value = count_matrix.iat[i, j]
                if value>0:
                    normed_value = value / count_matrix.values.max()
                    color = ax.collections[0].get_cmap()(normed_value)
                    radius = 5 * (value / count_matrix.max().max())
                    circle = plt.Circle((j + 0.5, i + 0.5), radius=radius, color=color, fill=True, alpha=0.5)
                    ax.add_patch(circle)
                    
        ax.invert_xaxis()
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=8)
        step = 20  
        start_label = 0
        end_label = 180

        num_labels = (end_label - start_label) // step + 1

        xticks = np.arange(-5, len(charge_labels), step)
        xticklabels = np.arange(end_label + step, start_label,-step)[:len(xticks)]
        ax.set_xticks([])
        ax.set_xticklabels(xticklabels,fontsize = 8)

        yticks = np.arange(-5, len(charge_labels), step)
        yticklabels = np.arange( end_label + step,start_label,- step)[:len(yticks)]
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_visible(True)
        #ax.axhline(y=len(charge_labels)-29, color='dimgray', linestyle='--',  label=f'80% Highlight')
        ax.text(len(charge_labels)-17, len(charge_labels)+2, 'Head', font = font_roman,fontsize=16, ha='center',fontdict={ 'weight':'bold'})
        ax.text(len(charge_labels)-50, len(charge_labels)+2, 'Tail',font = font_roman, fontsize=16, ha='center',fontdict={ 'weight':'bold'})
        ax.text(len(charge_labels)+2, len(charge_labels)-16, 'Head', font = font_roman,fontsize=16, ha='center',fontdict={ 'weight':'bold'})
        ax.text(len(charge_labels)+2, len(charge_labels)-50, 'Tail',font = font_roman, fontsize=16, ha='center',fontdict={ 'weight':'bold'})
        ax.set_yticks([])
        ax.set_yticklabels(yticklabels,fontsize = 8)
        ax.set_title(f'{model}',font = font_roman,fontsize = "large")
        ax.set_xlabel('True label',font = font_roman,fontsize = 16)
        ax.set_ylabel('Predicted label',font = font_roman,fontsize = 16)
        
    plt.show()
    #plt.savefig(f'./zero-shot_vs_ours_with_desc.png', bbox_inches='tight',pad_inches = 1)



if __name__ == '__main__':
    model_names = ["Qwen2-0.5B","Qwen2-1.5B","Qwen2-7B","Baichuan2-13B",'GPT-4o-mini', 'GPT-4o'] 
    datas = []
    for mn in model_names:
        pl2tl, _ = get_label_mapping(model_path[mn])
        datas.append(pl2tl)
    plot_multiple_heatmaps(model_names, datas)

