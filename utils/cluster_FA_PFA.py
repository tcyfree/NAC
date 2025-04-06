import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from factor_analyzer.factor_analyzer import FactorAnalyzer

# 1. 读取数据
data = pd.read_excel("./data/IMPRESS/select_by_pearson_columns_data_TNBC.xlsx")

# 2. 选择数值列
data_numeric = data.select_dtypes(include=[np.number])

# 3. 标准化数据
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_numeric)

# 4. 绘制碎石图，观察各因子的特征值（使用全部因子，不进行旋转）
fa_full = FactorAnalyzer(n_factors=data_numeric.shape[1], rotation=None)
fa_full.fit(data_numeric)  # 注意：因子分析一般基于相关矩阵
ev, _ = fa_full.get_eigenvalues()

# 可视化设置
plt.rcParams['font.sans-serif'] = ['Hei']  # 显示中文
plt.rcParams['axes.unicode_minus'] = False   # 显示负号
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(ev) + 1), ev, marker="o", linestyle="-")
plt.axhline(y=1, color="r", linestyle="--", label="Kaiser Criterion")
plt.xlabel("Number of Factors")
plt.ylabel("Eigenvalue")
plt.title("Scree Plot (碎石图) - Factor Analysis")
plt.legend()
plt.grid(True)
plt.show()

# 5. 选择特征值大于1的因子数量
n_factors = sum(ev > 1)
print(f"建议提取的因子数量: {n_factors}")

# 6. 使用 FactorAnalyzer 进行因子分析，指定斜交旋转（Promax）和PFA（主因子分析）
fa = FactorAnalyzer(n_factors=n_factors, rotation="promax", method="principal")
fa.fit(data_numeric)
data_fa = fa.transform(data_numeric)  # 每个样本在各因子上的得分（新的合成指标）

# 7. 将因子得分转换为 DataFrame，并保留原始样本索引
factor_scores = pd.DataFrame(data_fa, columns=[f"Factor {i+1} Score" for i in range(n_factors)])
factor_scores.index = data.index
print("因子得分：")
print(factor_scores.head())

# 保存因子得分到 Excel
factor_scores.to_excel("./data/IMPRESS/cluster/factor_scores_pfa.xlsx", sheet_name="Factor Scores", index=True)
print("因子得分已保存到 './data/IMPRESS/cluster/factor_scores_pfa.xlsx'")

# 8. 计算因子载荷矩阵（使用斜交旋转后的载荷）
factor_loadings = pd.DataFrame(fa.loadings_, index=data_numeric.columns,
                               columns=[f"Factor {i+1}" for i in range(n_factors)])
print("因子载荷矩阵：")
print(factor_loadings)

# 保存因子载荷矩阵到 Excel
factor_loadings.to_excel("./data/IMPRESS/cluster/result_with_clusters_pfa.xlsx", sheet_name="Factor Loadings", index=True)
print("因子载荷矩阵已保存到 './data/IMPRESS/cluster/result_with_clusters_pfa.xlsx'")

# 9. 绘制因子载荷矩阵的热图（不显示具体数值）
plt.figure(figsize=(10, 8))
sns.heatmap(factor_loadings, annot=False, cmap="coolwarm", center=0, linewidths=0.5)
plt.xlabel("Factors")
plt.ylabel("Features")
plt.title("Factor Loadings Heatmap - Oblique Rotation (Promax) [PFA]")
plt.show()
