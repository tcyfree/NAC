import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 读取数据
meta_df = pd.read_excel("cohort_meta_TNBC.xlsx")
means_df = pd.read_excel("./data/IMPRESS/TNBC/perDatasetSlideSummaries/RoiFeatureSummary_Means.xlsx")
# stds_df = pd.read_excel("./data/IMPRESS/TNBC/perDatasetSlideSummaries/RegionFeatureSummary_Stds.xlsx")

# 合并数据
df = pd.merge(meta_df, means_df, on="ID", how="inner")
# df = pd.merge(df, stds_df, on="ID", how="inner")

# 获取特征列
feature_columns = [col for col in df.columns if col not in ["ID", "pCR"]]

# 识别非数值列
non_numeric_columns = df[feature_columns].select_dtypes(exclude=["number"]).columns.tolist()

print(f"非数值列: {non_numeric_columns}")

# 删除非数值列
df_numeric = df.drop(columns=non_numeric_columns)

print(df_numeric.head())

# 计算 Pearson 相关性
correlation_matrix = df_numeric.corr(method="pearson")
correlation_with_pCR = correlation_matrix["pCR"].drop("pCR")

# 筛选相关性较高的特征（绝对值 > 0.3）
selected_features = correlation_with_pCR[correlation_with_pCR.abs() > 0.15]

# 筛选出与 pCR 相关性较高的特征
selected_features_list = selected_features.index.tolist()

print(f"Pearson 筛选后剩余的特征数: {len(selected_features_list)}")

# 筛选相关性矩阵，仅保留与这些特征相关的部分
filtered_correlation_matrix = correlation_matrix[selected_features_list].loc[selected_features_list]

# 对筛选后的相关性矩阵取绝对值
filtered_correlation_matrix_abs = filtered_correlation_matrix.abs()

# 设置热力图的显示
plt.figure(figsize=(4, 2))  # 设置图形大小
sns.heatmap(filtered_correlation_matrix, annot=False, fmt=".2f", cmap="coolwarm", cbar=True, square=True, linewidths=0.05)
plt.title("Filtered Feature Correlation Matrix", fontsize=16)
plt.show()

# 按相关性由高到低排序
sorted_selected_features = selected_features.abs().sort_values(ascending=False)

print(f"Pearson 筛选后剩余的特征数: {len(sorted_selected_features)}")
# 输出相关性由高到低排序的特征
print(sorted_selected_features)

# 将结果写入 Excel 文件
sorted_selected_features.to_excel("sorted_selected_features.xlsx", sheet_name="Correlation", header=True)
