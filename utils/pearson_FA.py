import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from factor_analyzer.factor_analyzer import FactorAnalyzer

# -------------------------------
# 步骤1：读取数据并进行基于 Pearson 相关性的特征筛选
# -------------------------------

# 读取数据（请根据实际路径调整文件路径）
df = pd.read_excel("./data/IMPRESS/random_group_one_TNBC_v2.xlsx")

# 获取特征列，排除 ID 和 pCR 列
feature_columns = [col for col in df.columns if col not in ["ID", "pCR"]]

# 删除非数值列
non_numeric_columns = df[feature_columns].select_dtypes(exclude=["number"]).columns.tolist()
df_numeric = df.drop(columns=non_numeric_columns)

# 计算 Pearson 相关性矩阵，并取出与 pCR 的相关系数（排除 pCR 自身和 ID 列）
correlation_matrix = df_numeric.corr(method="pearson")
correlation_with_pCR = correlation_matrix["pCR"].drop(["pCR", "ID"])

# 设置相关性阈值
threshold = 0.25
selected_features = correlation_with_pCR[correlation_with_pCR.abs() > threshold]
selected_features_list = selected_features.index.tolist()

# 输出筛选结果
sorted_selected_features = selected_features.sort_values(ascending=False)
print(f"阈值: {threshold}, 特征数: {len(sorted_selected_features)}")
print(sorted_selected_features)

# 使用筛选后的特征构建因子分析的数据集
# 注意：如果 pCR 或其他非特征变量需要保留，可根据实际需求调整
data = df_numeric[selected_features_list]

# -------------------------------
# 步骤2：因子分析
# -------------------------------

# 选择数值列（此处 data 已为数值型数据）
data_numeric = data.copy()

# 标准化数据
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_numeric)

# 绘制碎石图，观察各因子的特征值（使用全部因子，不进行旋转）
fa_full = FactorAnalyzer(n_factors=data_numeric.shape[1], rotation=None)
fa_full.fit(data_numeric)  # 注意：因子分析通常基于相关矩阵
ev, _ = fa_full.get_eigenvalues()

# 可视化设置
plt.rcParams['font.sans-serif'] = ['Hei']  # 显示中文
plt.rcParams['axes.unicode_minus'] = False   # 正常显示负号
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(ev) + 1), ev, marker="o", linestyle="-")
plt.axhline(y=1, color="r", linestyle="--", label="Kaiser Criterion")
plt.xlabel("因子数量")
plt.ylabel("特征值")
plt.title("碎石图 - 因子分析")
plt.legend()
plt.grid(True)
plt.show()

# 选择特征值大于1的因子数量
n_factors = sum(ev > 1)
print(f"建议提取的因子数量: {n_factors}")

# 使用 FactorAnalyzer 进行因子分析，指定斜交旋转（Promax）和最大似然法（ml）
fa = FactorAnalyzer(n_factors=n_factors, rotation="promax", method="ml")
fa.fit(data_numeric)
data_fa = fa.transform(data_numeric)  # 每个样本在各因子上的得分（新的合成指标）

# 将因子得分转换为 DataFrame，并保留原始样本索引
factor_scores = pd.DataFrame(data_fa, columns=[f"Factor {i+1} Score" for i in range(n_factors)])
factor_scores.index = data.index  # 保留原始数据的索引
print("因子得分：")
print(factor_scores.head())

# 保存因子得分到 Excel
factor_scores.to_excel("./data/IMPRESS/cluster/factor_scores_ml_pearson.xlsx", sheet_name="Factor Scores", index=True)
print("因子得分已保存到 './data/IMPRESS/cluster/factor_scores_ml_pearson.xlsx'")

# 计算因子载荷矩阵（使用斜交旋转后的载荷）
factor_loadings = pd.DataFrame(fa.loadings_, index=data_numeric.columns,
                               columns=[f"Factor {i+1}" for i in range(n_factors)])

# 保存因子载荷矩阵到 Excel
factor_loadings.to_excel("./data/IMPRESS/cluster/result_with_clusters_ml_pearson.xlsx", sheet_name="Factor Loadings", index=True)
print("因子载荷矩阵已保存到 './data/IMPRESS/cluster/result_with_clusters_ml_pearson.xlsx'")

# 绘制因子载荷矩阵的热图（不显示具体数值）
plt.figure(figsize=(10, 8))
sns.heatmap(factor_loadings, annot=False, cmap="coolwarm", center=0, linewidths=0.5)
plt.xlabel("因子")
plt.ylabel("特征")
plt.title("因子载荷矩阵热图 - 斜交旋转（ml）")
plt.show()
