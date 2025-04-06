import pandas as pd

# 读取 Excel 文件
df = pd.read_excel("./data/IMPRESS/cluster/factor_main_variables_ml_0.7.xlsx")
df = pd.read_excel("./data/IMPRESS/cluster/factor_main_variables_ml_0.8.xlsx")
df = pd.read_excel("./data/IMPRESS/cluster/factor_main_variables_ml_0.9.xlsx")

# 把整张表拉成一维（所有值摊开），然后去除缺失值（NaN）
all_values = df.values.flatten()
all_values = pd.Series(all_values).dropna()

# 找出重复的值
duplicated_values = all_values[all_values.duplicated()].unique()

# 输出
if len(duplicated_values) > 0:
    print("表格中存在重复的值：")
    print(duplicated_values)
else:
    print("整张表格中所有值都是唯一的，没有重复。")
