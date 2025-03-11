import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 读取数据
meta_df = pd.read_excel("cohort_meta_TNBC.xlsx")
means_df = pd.read_excel("./data/IMPRESS/TNBC/perDatasetSlideSummaries/RoiFeatureSummary_Means.xlsx")
stds_df = pd.read_excel("./data/IMPRESS/TNBC/perDatasetSlideSummaries/RoiFeatureSummary_Stds.xlsx")

# 合并数据
df = pd.merge(meta_df, means_df, on="ID", how="inner")
df = pd.merge(df, stds_df, on="ID", how="inner")

# 获取特征列
feature_columns = [col for col in df.columns if col not in ["ID", "pCR"]]

# 识别非数值列
non_numeric_columns = df[feature_columns].select_dtypes(exclude=["number"]).columns.tolist()

print(f"非数值列: {non_numeric_columns}")

# 删除非数值列
df_numeric = df.drop(columns=non_numeric_columns)

# 计算 Pearson 相关性
correlation_matrix = df_numeric.corr(method="pearson")

# 设置热力图的显示
plt.figure(figsize=(12, 8))  # 设置图形大小
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, square=True, linewidths=0.5)
plt.title("Feature Correlation Matrix", fontsize=16)

# 保存热力图为图片文件，而不显示
plt.savefig("correlation_matrix_heatmap.png", bbox_inches="tight")

# 关闭绘图
plt.close()
