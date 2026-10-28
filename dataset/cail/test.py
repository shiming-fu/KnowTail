import pandas as pd
 
# 假设df是一个pandas DataFrame
df = pd.DataFrame({'A': [1, 2, 3, 4], 'B': [5, 6, 7, 8]})
 
# 假设我们想要选择列'A'中值为3的行
boolean_index = df['A'] == 3
selected_row = df[boolean_index]
print(selected_row)