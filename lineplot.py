import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import json

dataset = {
    'gpt-4o-rag':'./micro_accuracy/gpt-4o-2024-08-06_rag_testset_rag_4_metrics.json',
    'gpt-4o-cot':'./micro_accuracy/gpt-4o-2024-08-06_few_shot_cot_testset_3_shot_cot_metrics.json',
    #'gpt-4o-zero-shot':'./micro_accuracy/gpt-4o-2024-08-06_zero_shot_testset_zero_shot_metrics.json',
    'qwen2-1.5b-rag':'./micro_accuracy/qwen2-1.5b_rag_testset_rag_4_metrics.json',
    'qwen2-1.5b-cot':'./micro_accuracy/qwen2-1.5b_few_shot_cot_testset_3_shot_cot_metrics.json',
    #'qwen2-1.5b-zero-shot':'./micro_accuracy/qwen2-1.5b_zero_shot_testset_zero_shot_metrics.json',
}

# Sample data for plotting
def load_metrics():
    data = {}
    
    # Add placeholders for each model's metrics
    for crime in ["洗钱", "虐待"]:
            data[crime] = {
                'model' :['gpt-4o-rag'] * 15 + ['gpt-4o-cot'] * 15 + ['qwen2-1.5b-rag'] * 15 + ['qwen2-1.5b-cot'] * 15
            }
            for model_name in dataset.keys():
                data[crime][f'{model_name}_acc'] = []
                data[crime][f'{model_name}_prec'] = []
                data[crime][f'{model_name}_rec'] = []
                data[crime][f'{model_name}_f1'] = []
    
    # Load data from each file and fill in the metrics
    for model_name, file_path in dataset.items():
        with open(file_path, 'r',encoding='utf-8') as f:
            json_data = [json.loads(line) for line in f]  # Load each line of JSON

            # Extract different metrics depending on the model
            for metric in json_data:
                if metric.get('charge') in ["洗钱","虐待"]:

                    
                    data[metric.get('charge')][f'{model_name}_prec'].insert(0, metric.get('precision'))
                    data[metric.get('charge')][f'{model_name}_rec'].insert(0, metric.get('recall'))
               
    return data
def barplot():
    data = load_metrics()
    
    df = pd.DataFrame(data)
    sns.set(style='whitegrid')
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))  # 1x2 grid for 2 metrics
    palette = ['#1f77b4', '#ff7f0e']  # Precision 和 Recall 的颜色
    # 左边子图 - Money Laundering
    sns.barplot(data=df.filter(like='洗钱'), ax=ax[0], palette=palette, ci=None,dodge=False,hue = "model")
    ax[0].set_title('Money Laundering')
    ax[0].set_ylabel('Score (%)')
    ax[0].set_xticklabels(df.filter(like='洗钱').columns.str.replace('洗钱', '').str.replace('_prec', ''), rotation=45)

    # 在柱子上显示具体数字
    for p in ax[0].patches:
        ax[0].annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                       ha='center', va='bottom')

    ax[0].spines['right'].set_visible(False)
    ax[0].spines['top'].set_visible(False)
    ax[0].grid(True)

    # 右边子图 - Abuse
    sns.barplot(data=df.filter(like='虐待'), ax=ax[1], palette=palette, ci=None, dodge=False,hue = "model")
    ax[1].set_title('Abuse')
    ax[1].set_ylabel('Score (%)')
    ax[1].set_xticklabels(['cot','rag'])

    # 在柱子上显示具体数字
    for p in ax[1].patches:
        ax[1].annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                       ha='center', va='bottom')

    ax[1].spines['right'].set_visible(False)
    ax[1].spines['top'].set_visible(False)
    ax[1].grid(True)

    # 处理图例
    handles = [plt.Rectangle((0, 0), 1, 1, color=palette[0]), plt.Rectangle((0, 0), 1, 1, color=palette[1])]
    labels = ['Precision', 'Recall']
    fig.legend(handles, labels, loc='lower center', ncol=2, fontsize='small', frameon=False)

    plt.tight_layout()  # 调整布局
    plt.show()







if __name__ == '__main__':
    barplot()
