import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve

# 读取数据
meta_df = pd.read_excel("cohort_meta_TNBC.xlsx")
means_df = pd.read_excel("./data/IMPRESS/TNBC/perDatasetSlideSummaries/GlobalRoiBasedFeatures.xlsx")

# 合并数据
df = pd.merge(meta_df, means_df, on="ID", how="inner")

# 获取特征列
feature_columns = [col for col in df.columns if col not in ["ID", "pCR"]]

# 识别非数值列
non_numeric_columns = df[feature_columns].select_dtypes(exclude=["number"]).columns.tolist()
print(f"非数值列: {non_numeric_columns}")

# 删除非数值列
df_numeric = df.drop(columns=non_numeric_columns)

# 计算 Pearson 相关性
correlation_matrix = df_numeric.corr(method="pearson")
correlation_with_pCR = correlation_matrix["pCR"].drop("pCR")

# 筛选相关性较高的特征（绝对值 > 0.15）
selected_features = correlation_with_pCR[correlation_with_pCR.abs() > 0.36]
selected_features_list = selected_features.index.tolist()

print(f"Pearson 筛选后剩余的特征数: {len(selected_features_list)}")

# 筛选相关性矩阵，仅保留这些特征
filtered_correlation_matrix = correlation_matrix[selected_features_list].loc[selected_features_list]

# 相关性热力图
# plt.figure(figsize=(4, 2))
# sns.heatmap(filtered_correlation_matrix, annot=False, fmt=".2f", cmap="coolwarm", cbar=True, square=True, linewidths=0.05)
# plt.title("Filtered Feature Correlation Matrix", fontsize=16)
# plt.show()

# 相关性排序
sorted_selected_features = selected_features.abs().sort_values(ascending=False)
print(f"Pearson 筛选后剩余的特征数: {len(sorted_selected_features)}")
print(sorted_selected_features)

# 保存筛选后的特征列表
sorted_selected_features.to_excel("sorted_selected_features.xlsx", sheet_name="Correlation", header=True)

### **新增: 预测 pCR 并计算 AUC** ###
# 提取目标变量和特征
# print('selected_features_list:', selected_features_list)
X = df_numeric[selected_features_list]
y = df_numeric["pCR"]

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 训练集/测试集拆分
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 训练逻辑回归模型 "l1" 表示使用 L1 正则化，也称为 Lasso 正则化
model = LogisticRegression(penalty="l1", solver="liblinear", C=1.0)
model.fit(X_train, y_train)

# 计算预测概率
y_prob = model.predict_proba(X_test)[:, 1]

# 计算 AUC
auc_score = roc_auc_score(y_test, y_prob)
print(f"预测结果 AUC: {auc_score:.4f}")

# 绘制 ROC 曲线
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.figure(figsize=(5, 4))
plt.plot(fpr, tpr, label=f"AUC = {auc_score:.4f}", color="blue")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()
