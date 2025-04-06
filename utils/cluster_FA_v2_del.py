import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import FactorAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity, calculate_kmo, FactorAnalyzer
import seaborn as sns

# 1. 读取数据
data = pd.read_excel("./data/IMPRESS/select_by_pearson_columns_data_TNBC.xlsx")

# 排除指定列
columns_to_drop = [
    "NuclearTexture.SumAverage.Mean.StromalSuperclass.Mean",
    "NuclearTexture.Mag.Skewness.TILsSuperclass.Mean",
    "NuclearTexture.Mag.Skewness.StromalSuperclass.Mean",
    "NuclearStaining.Skewness.StromalSuperclass.Mean",
    "NuclearStaining.Min.StromalSuperclass.Std",
    "NuclearStaining.Median.StromalSuperclass.Mean",
    "NuclearStaining.Mean.StromalSuperclass.Mean",
    "NuclearStaining.Max.StromalSuperclass.Mean",
    "NuclearStaining.Kurtosis.StromalSuperclass.Mean",
    "NoOfNuclei.UnknownOrAmbiguousCell",
    "ComplexNuclearShape.WeightedHuMoments1.StromalSuperclass.Mean"
]
data = data.drop(columns=columns_to_drop, errors='ignore')

# 2. 选择数值列
data_numeric = data.select_dtypes(include=[np.number])

# 计算数据的相关矩阵
corr_matrix = np.corrcoef(data_numeric.T)

# 计算相关矩阵的行列式
det_corr = np.linalg.det(corr_matrix)

print(f"相关矩阵的行列式: {det_corr}")

if np.isclose(det_corr, 0):
    print("⚠️ 相关矩阵的行列式接近 0，可能存在完全线性相关变量")

# 3. 标准化数据
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_numeric)

# 4. 进行 Bartlett 球形检验（检查变量是否适合因子分析）
chi_square_value, p_value = calculate_bartlett_sphericity(data_numeric)
print(f"Bartlett Test: χ²={chi_square_value}, p-value={p_value}")
if p_value < 0.05:
    print("数据适合因子分析")
else:
    print("数据可能不适合因子分析")

# 5. KMO 检验（适合性测试）
kmo_all, kmo_model = calculate_kmo(data_numeric)
print(f"KMO Test: {kmo_model}")
if kmo_model > 0.6:
    print("KMO 值较高，适合进行因子分析")
else:
    print("KMO 值较低，数据可能不适合因子分析")

# 6. 选择最佳因子数（碎石图）
fa = FactorAnalyzer(n_factors=data_numeric.shape[1], rotation=None)
fa.fit(data_numeric)
ev, _ = fa.get_eigenvalues()  # 获取特征值

# 可视化设置
plt.rcParams['font.sans-serif'] = ['Hei']  # 显示中文
plt.rcParams['axes.unicode_minus'] = False    # 显示负号

plt.figure(figsize=(8, 5))
plt.plot(range(1, len(ev) + 1), ev, marker="o", linestyle="-")
plt.axhline(y=1, color="r", linestyle="--", label="Kaiser Criterion")
plt.xlabel("Number of Factors")
plt.ylabel("Eigenvalue")
plt.title("Scree Plot (碎石图)")
plt.legend()
plt.grid(True)
plt.show()

# 7. 选择特征值 >1 的因子数
n_factors = sum(ev > 1)
print(f"建议提取的因子数量: {n_factors}")

# 8. 进行因子分析
fa = FactorAnalysis(n_components=n_factors, random_state=42)
data_fa = fa.fit_transform(data_scaled) # 返回的就是每个样本在各个因子上的得分，也就是新的合成指标

# 将因子得分转换为 DataFrame，并给出因子名称
factor_scores = pd.DataFrame(data_fa, columns=[f"Factor {i+1} Score" for i in range(n_factors)])
factor_scores.index = data.index  # 如果需要保留原始样本索引

# 查看前几行因子得分
print(factor_scores.head())

# 将因子得分保存到 Excel
factor_scores.to_excel("./data/IMPRESS/cluster/factor_scores_v2_del.xlsx", sheet_name="Factor Scores", index=True)
print("因子得分已保存到 './data/IMPRESS/cluster/factor_scores.xlsx'")

# 9. 解释因子载荷（查看变量对因子的贡献）
factor_loadings = pd.DataFrame(fa.components_.T, index=data_numeric.columns, columns=[f"Factor {i+1}" for i in range(n_factors)])
print("因子载荷矩阵：")
print(factor_loadings)

# 7. 保存因子载荷矩阵到 Excel
factor_loadings.to_excel("./data/IMPRESS/cluster/result_with_clusters_v2_del.xlsx", sheet_name="Factor Loadings", index=True)

print("因子载荷矩阵已保存到 'factor_loadings.xlsx'")

# 绘制热图
plt.figure(figsize=(10, 8))
sns.heatmap(factor_loadings, annot=False, cmap="coolwarm", center=0, linewidths=0.5, fmt=".2f")
plt.xlabel("Factors")
plt.ylabel("Features")
plt.title("Factor Loadings Heatmap")
plt.show()


# 设定因子载荷阈值（一般 > 0.6 认为有显著贡献）
threshold = 0.6

# 找出每个因子的主要变量
for factor in factor_loadings.columns:
    high_loadings = factor_loadings[factor].abs() > threshold
    selected_vars = factor_loadings.index[high_loadings].tolist()
    print(f"{factor} 主要变量: {selected_vars}")

# # 10. 选择前 2 个因子进行可视化
# if n_factors >= 2:
#     plt.figure(figsize=(8, 6))
#     scatter = plt.scatter(data_fa[:, 0], data_fa[:, 1], alpha=0.6)
#     plt.xlabel("Factor 1")
#     plt.ylabel("Factor 2")
#     plt.title("Factor Analysis Visualization")
#     plt.grid(True)
#     plt.show()
# else:
#     print("因子数不足 2 维，无法进行 2D 可视化")
