import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

op_QWen2_7B = [72.83, 73.4, 74.34, 74.69, 75.15]
op_desc_QWen2_7B = [73.03, 73.45, 73.97, 74.17, 74.65]
random_op_QWen2_7B = [47.23, 47.94, 47.8, 47.77, 47.06]

op_BaiChuan2_13B = [64.43, 65.21, 65.69, 66.39, 66.23]
op_desc_BaiChuan2_13B = [64.75, 65.96, 66.47, 66.35, 66.93]
random_op_BaiChuan2_13B = [47.02, 46.8, 47.02, 47.13, 46.73]

op_GPT4o_mini = [60.75, 61.89, 63.23, 63.29, 64.4]
op_desc_GPT4o_mini = [62.51, 64.32, 65.35, 66.51, 67.26]
random_op_GPT4o_mini = [30, 30.9, 30.2, 30.1, 29.3]

op_GPT4o = [74.62, 75.4, 75.71, 76.18, 75.99]
op_desc_GPT4o = [76.72, 77.68, 78.19, 78.49, 78.39]
random_op_GPT4o = [50.25, 51, 50.1, 50.5, 50.7]




# Data
x = np.array([4, 5, 6, 7, 8])

data = {
    '': list(x) * 3 * 4,
    'F1-score': random_op_QWen2_7B+op_QWen2_7B + op_desc_QWen2_7B  + 
              random_op_BaiChuan2_13B+ op_BaiChuan2_13B + op_desc_BaiChuan2_13B  + 
              random_op_GPT4o_mini + op_GPT4o_mini + op_desc_GPT4o_mini +
              random_op_GPT4o+op_GPT4o + op_desc_GPT4o ,
    'model': ['QWen2-7B'] * 15 + ['BaiChuan2-13B'] * 15 + ['GPT-4o-mini'] * 15 + ['GPT-4o'] * 15,
    'type': ( ['Random options'] * 5 + ['Self-reflection options'] * 5 + ['Self-reflection options+Legal rule'] * 5) * 4
}

df = pd.DataFrame(data)

# Set the theme for the plot
sns.set(style="whitegrid")

# Create a 2x2 grid of plots
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# Create list of model names for iteration
models = ['QWen2-7B', 'BaiChuan2-13B', 'GPT-4o-mini', 'GPT-4o']
axes = axs.flatten()

# Plot each model in its corresponding subplot
palette = [(0.5529411764705883, 0.6274509803921569, 0.796078431372549),
           (0.9882352941176471, 0.5529411764705883, 0.3843137254901961),
           (0.4, 0.7607843137254902, 0.6470588235294118)]
print(sns.color_palette('Set2'))
for i, model in enumerate(models):
    bar_plot = sns.barplot(data=df[df['model'] == model], x='', y='F1-score', hue='type', palette=palette, ax=axes[i], dodge=True)
    axes[i].set_ylabel("F1-scores (%)",fontsize='x-large',y = 0.4)
    axes[i].set_title(model, fontsize='xx-large',y=0.98)
    axes[i].legend_.remove()
    axes[i].tick_params(axis='y', labelsize='small')
    
    if model == 'GPT-4o-mini':
        axes[i].set_ylim(20, None)
        axes[i].spines['top'].set_visible(False)
        axes[i].spines['right'].set_visible(False)
        axes[i].spines['left'].set_visible(False)
        y_ticks = axes[i].get_yticks()
        new_y_ticks = y_ticks[:-2]
        new_y_ticks = new_y_ticks.astype(int)
        axes[i].set_yticks(new_y_ticks)
        axes[i].set_yticklabels(new_y_ticks, fontsize='large') 
        x_tick_labels = [f"{val}" for val in x]
        axes[i].set_xticklabels(x_tick_labels,fontsize = 'large')
        axes[i].set_xlabel("Number of options",fontsize = 'x-large')
        axes[i].yaxis.grid(True)
        
        
        # 转换为整数
    elif model == "QWen2-7B":
        axes[i].set_ylim(40, None)
        axes[i].spines['top'].set_visible(False)
        axes[i].spines['right'].set_visible(False)
        axes[i].spines['left'].set_visible(False)
       # 获取当前 y 轴范围并生成新的 y_ticks
        y_min, y_max = axes[i].get_ylim()
        new_y_ticks = np.arange(np.ceil(y_min / 10) * 10, np.floor(y_max / 10) * 10 + 1, 10)
        new_y_ticks = new_y_ticks.astype(int)
        axes[i].set_yticks(new_y_ticks)
        axes[i].set_yticklabels(new_y_ticks, fontsize='large') 
        x_tick_labels = [f"{val}" for val in x]
        axes[i].set_xticklabels(x_tick_labels,fontsize = 'large')
        axes[i].set_xlabel("Number of options",fontsize = 'x-large')
        axes[i].yaxis.grid(True)
    elif model == "GPT-4o":
            axes[i].set_ylim(40, None)
            axes[i].spines['top'].set_visible(False)
            axes[i].spines['right'].set_visible(False)
            axes[i].spines['left'].set_visible(False)
        # 获取当前 y 轴范围并生成新的 y_ticks
            y_min, y_max = axes[i].get_ylim()
            new_y_ticks = np.arange(np.ceil(y_min / 10) * 10, np.floor(y_max / 10) * 10 + 1, 10)
            new_y_ticks = new_y_ticks[:-1]
            new_y_ticks = new_y_ticks.astype(int)
            axes[i].set_yticks(new_y_ticks)
            axes[i].set_yticklabels(new_y_ticks, fontsize='medium') 
            x_tick_labels = [f"{val}" for val in x]
            axes[i].set_xticklabels(x_tick_labels,fontsize = 'large')
            axes[i].set_xlabel("Number of options",fontsize = 'x-large')
            axes[i].yaxis.grid(True)
    else:
        axes[i].set_ylim(40, None)
        axes[i].spines['top'].set_visible(False)
        axes[i].spines['right'].set_visible(False)
        axes[i].spines['left'].set_visible(False)
       # 获取当前 y 轴范围并生成新的 y_ticks
        y_ticks = axes[i].get_yticks()
        new_y_ticks = y_ticks[:-2]
        axes[i].set_yticks(new_y_ticks)
        new_y_ticks = new_y_ticks.astype(int)
        axes[i].set_yticks(new_y_ticks)
        axes[i].set_yticklabels(new_y_ticks, fontsize='medium') 
        x_tick_labels = [f"{val}" for val in x]
        axes[i].set_xticklabels(x_tick_labels,fontsize = 'large')
        axes[i].set_xlabel("Number of options",fontsize = 'x-large')
        axes[i].yaxis.grid(True)
    # for p in bar_plot.patches:
    #     axes[i].annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2., p.get_height()),
    #                      ha='center', va='bottom', fontsize='xx-small', color='black')

# Place the legend at the bottom of the entire plot
handles, labels = axes[0].get_legend_handles_labels()  # Get legend information
fig.legend(handles, labels, loc='lower center', ncol=3, fontsize='x-large', frameon=False)

plt.show()



