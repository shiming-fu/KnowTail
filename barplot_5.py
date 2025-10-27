import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# 数据
data = {
    "Model": ["qwen-rag", "qwen-cot", "gpt-4o-rag", "gpt-4o-cot"],
    "Riotous Robbery": [38.89, 19.26, 52.59, 48.15],
    "Destruction of Means of Transportation": [42.86, 38.1, 42.86, 13.61],
    "Abuse of Persons Under Custody": [33.33, 14.81, 35.93, 29.26],
    "Trafficking in Cultural Relics": [25, 17.5, 70.83, 77.78], 
    "Usury Transfer of Loans": [40, 13.33, 66.67, 60]
}

# 转换为DataFrame
df = pd.DataFrame(data)

# 重塑数据框以便于seaborn绘图
df_melted = df.melt(id_vars='Model', var_name='Charge', value_name='F1-score')

# 添加Type列来区分COT和RAG
df_melted['Type'] = df_melted['Model'].apply(lambda x: 'COT' if 'cot' in x else 'RAG')
df_melted['Group'] = df_melted['Model'].apply(lambda x: 'Qwen' if 'qwen' in x else 'GPT')

sns.set(style='darkgrid')

# 设置子图为2行1列
fig, axes = plt.subplots(1, 2, figsize=(12, 14))

# 绘制Qwen相关的柱状图
sns.barplot(x='Charge', y='F1-score', hue='Type',
            data=df_melted[df_melted['Group'] == 'Qwen'],
            palette='Set2', ax=axes[0])
axes[0].set_title('QWen2-1.5B', fontsize="x-large")
axes[0].set_ylabel('F1-score (%)')
axes[0].set_xlabel('')
axes[0].set_xticklabels([1,2,3,4,5],  fontsize='large')
axes[0].spines['right'].set_visible(False)
axes[0].spines['top'].set_visible(False)
axes[0].legend( loc='upper right')

# 添加Qwen数据标签
for p in axes[0].patches:
    height = p.get_height()
    if height > 0:
        axes[0].annotate(f'{height:.2f}', 
                         (p.get_x() + p.get_width() / 2., height), 
                         ha='center', va='bottom')

# 绘制GPT相关的柱状图
sns.barplot(x='Charge', y='F1-score', hue='Type',
            data=df_melted[df_melted['Group'] == 'GPT'],
            palette='Set2', ax=axes[1])
axes[1].set_title('GPT-4o', fontsize="x-large")
axes[1].set_ylabel('F1-score (%)')
axes[1].set_xlabel('')
axes[1].set_xticklabels([1,2,3,4,5],  fontsize='large')
axes[1].spines['right'].set_visible(False)
axes[1].spines['top'].set_visible(False)
axes[1].legend_.remove()
axes[1].legend( loc='upper right')
# 添加GPT数据标签
for p in axes[1].patches:
    height = p.get_height()
    if height > 0:
        axes[1].annotate(f'{height:.2f}', 
                         (p.get_x() + p.get_width() / 2., height), 
                         ha='center', va='bottom')


plt.show()
