import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# 数据
data = {
    "Model": ["qwen-rag", "qwen-cot", "gpt-4o-rag", "gpt-4o-cot"],
    "Money Laundering ": [40.62, 17.5, 19.79, 5.0],
    "Retaliation Against Witnesses ": [10.0, 10.0, 13.33, 6.67],
}

# 转换为DataFrame
df = pd.DataFrame(data)

# 重塑数据框以便于seaborn绘图
df_melted = df.melt(id_vars='Model', var_name='Charge', value_name='F1-score')

# 添加Type列来区分COT和RAG
df_melted['Type'] = df_melted['Model'].apply(lambda x: 'COT' if 'cot' in x else 'RAG')

sns.set(style="darkgrid")
# 设置子图
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# 绘制QWEn的柱状图
sns.barplot(x='Charge', y='F1-score', hue='Type', 
            data=df_melted[df_melted['Model'].str.contains('qwen')], 
            palette='Set2', ax=axes[0])
axes[0].set_title('QWen2-1.5B', fontsize="x-large")
axes[0].set_ylabel('F1-score (%)')
axes[0].set_xlabel('')
axes[0].legend(loc='upper right')
axes[0].set_xticklabels(axes[0].get_xticklabels(), fontsize='large')
axes[0].spines['right'].set_visible(False)
axes[0].spines['top'].set_visible(False)
axes[0].spines['left'].set_visible(False)

sns.barplot(x='Charge', y='F1-score', hue='Type', 
            data=df_melted[df_melted['Model'].str.contains('gpt')], 
            palette='Set2', ax=axes[1])
axes[1].set_title('GPT-4o', fontsize="x-large")
axes[1].set_ylabel('F1-score (%)')
axes[1].set_xlabel('')
axes[1].legend( loc='upper right')
axes[1].set_xticklabels(axes[1].get_xticklabels(), fontsize='large')
axes[1].spines['right'].set_visible(False)
axes[1].spines['top'].set_visible(False)
axes[1].spines['left'].set_visible(False)

for ax in axes:
    for p in ax.patches:
        height = p.get_height()
        if height>0:
            ax.annotate(f'{height:.2f}', (p.get_x() + p.get_width() / 2., height), ha='center', va='bottom')

plt.show()
