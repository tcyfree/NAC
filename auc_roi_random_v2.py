import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor


# 读取随机分（8份）好的xlsx 文件
# df = pd.read_excel("./data/IMPRESS/random_group_eight_TNBC_v3.xlsx")
df = pd.read_excel("./data/IMPRESS/select_p_columns_data_TNBC.xlsx")
# df = pd.read_excel("./data/IMPRESS/select_p_columns_data_TNBC_v1.xlsx")
# df = pd.read_excel("./data/IMPRESS/select_p_columns_data_HER2.xlsx")

# 获取特征列
feature_columns = [col for col in df.columns if col not in ["ID", "pCR"]]

# 识别非数值列
non_numeric_columns = df[feature_columns].select_dtypes(exclude=["number"]).columns.tolist()
print(f"非数值列: {non_numeric_columns}")

# 删除非数值列
df_numeric = df.drop(columns=non_numeric_columns)

print(f"df_numeric 共有 {df_numeric.shape[0]} 行，{df_numeric.shape[1]} 列")

# 计算 Pearson 相关性
correlation_matrix = df_numeric.corr(method="pearson")
# 删除 pCR 和 ID 列
correlation_with_pCR = correlation_matrix["pCR"].drop(["pCR", "ID"])

# 筛选相关性较高的特征（绝对值 > 0.15）
threshold = 0.2
selected_features = correlation_with_pCR[correlation_with_pCR.abs() > threshold]
selected_features_list = selected_features.index.tolist()

# 相关性排序
sorted_selected_features = selected_features.sort_values(ascending=False)
print(f"阈值: {threshold}, 特征数: {len(sorted_selected_features)}")
print(sorted_selected_features)

# 保存筛选后的特征列表
# sorted_selected_features.to_excel("./data/IMPRESS/sorted_selected_features_TNBC_p.xlsx", sheet_name="Correlation", header=True)

# 筛选相关性矩阵，仅保留这些特征
filtered_correlation_matrix = correlation_matrix[selected_features_list].loc[selected_features_list]

# 相关性热力图
plt.figure(figsize=(12, 8))
sns.heatmap(filtered_correlation_matrix, annot=False, fmt=".2f", cmap="coolwarm", cbar=True, square=True, linewidths=0.05)
plt.title("Filtered Feature Correlation Matrix", fontsize=16)
plt.show()

# ===============按照病人拆分=======================
# 确保ID列是字符串类型
df_numeric["ID"] = df_numeric["ID"].astype(str)

# 按病人ID获取唯一ID列表
unique_ids = df_numeric["ID"].unique()

# 按病人ID进行训练/测试集拆分
train_ids, test_ids = train_test_split(unique_ids, test_size=0.2, random_state=42)

# print(train_ids)
# print(test_ids)

# 选择对应病人ID的数据
train_df = df_numeric[df_numeric["ID"].isin(train_ids)]
test_df = df_numeric[df_numeric["ID"].isin(test_ids)]

# 获取特征和目标变量
X_train = train_df[selected_features_list]
y_train = train_df["pCR"]
X_test = test_df[selected_features_list]
y_test = test_df["pCR"]

# 数据标准化（测试时用到）
scaler = StandardScaler()
print(f"X_train 共有 {len(X_train)} 行")
X_train_scaled = scaler.fit_transform(X_train)
print(f"X_test 共有 {len(X_test)} 行")
X_test_scaled = scaler.transform(X_test)

# 训练 sklearn 逻辑回归模型（L1 正则化）
model = LogisticRegression(penalty="l1", solver="liblinear", C=1.0)
model.fit(X_train, y_train)

# 获取特征重要性（归一化系数）
feature_importance = model.coef_.flatten()
feature_importance /= np.max(np.abs(feature_importance))

# 只保留非零权重的特征
nonzero_indices = np.where(feature_importance != 0)[0]
print(f"特征重要性大于 0 的特征个数: {len(nonzero_indices)}")
feature_importance = feature_importance[nonzero_indices]
feature_names = np.array(selected_features_list)[nonzero_indices]

# 按重要性排序
sorted_indices = np.argsort(abs(feature_importance))[::-1]
feature_importance = feature_importance[sorted_indices]
feature_names = feature_names[sorted_indices]

# 1. 确保 sorted_selected_features 是 DataFrame
df_features = sorted_selected_features.reset_index()
df_features.columns = ['Feature', 'Correlation_pCR']  # 重命名列

# 2. 给特征打上 L1_Selected 标记
df_features['L1_Selected'] = df_features['Feature'].apply(lambda x: "Yes" if x in feature_names else "No")

# 3. 重要性 feature_importance
importance_dict = dict(zip(feature_names, feature_importance))
df_features['L1_Coefficient'] = df_features['Feature'].map(importance_dict).fillna(0)  # 不在列表中的特征填充 0

# 4. 保存结果到 Excel
output_file = "./data/IMPRESS/sorted_selected_features_TNBC_p_l1_imp.xlsx"
df_features.to_excel(output_file, sheet_name="Correlation_pCR", index=False)

print(f"更新后的特征列表已保存到 {output_file}")


# 画特征重要性柱状图（可选）
plt.figure(figsize=(12, 8))
plt.barh(feature_names, feature_importance, color='b')
plt.xlabel("Coefficient Value")
plt.ylabel("Feature")
plt.title("Feature Importance (L1 Regularization)")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# 计算预测概率（sklearn模型）
y_prob = model.predict_proba(X_test)[:, 1]
auc_score = roc_auc_score(y_test, y_prob)
print(f"预测结果 AUC: {auc_score:.4f}")

# 使用标准化后的数据预测类别（仅用于 accuracy 计算）
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"预测结果 Accuracy: {accuracy:.4f}")


# 绘制 ROC 曲线
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"AUC = {auc_score:.4f}", color="blue")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()