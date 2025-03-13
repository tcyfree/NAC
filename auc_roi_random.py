import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
import numpy as np

# 读取随机分（4份）好的xlsx 文件
df = pd.read_excel("output_random_group_eight_2.xlsx")

# 获取特征列
feature_columns = [col for col in df.columns if col not in ["ID", "pCR"]]

# 识别非数值列
non_numeric_columns = df[feature_columns].select_dtypes(exclude=["number"]).columns.tolist()
print(f"非数值列: {non_numeric_columns}")

# 删除非数值列
df_numeric = df.drop(columns=non_numeric_columns)

print(f"df_numeric 共有 {df_numeric.shape[0]} 行，{df_numeric.shape[1]} 列")
print(f"df_numeric 共有 {len(df_numeric)} 行")
# df_numeric.to_excel("merged_data.xlsx", index=False)
# print("合并后的数据已保存到 merged_data.xlsx")

# 计算 Pearson 相关性
correlation_matrix = df_numeric.corr(method="pearson")
# 删除 pCR 和 ID 列
correlation_with_pCR = correlation_matrix["pCR"].drop(["pCR", "ID"])

# 筛选相关性较高的特征（绝对值 > 0.15）
threshold = 0.0001
selected_features = correlation_with_pCR[correlation_with_pCR.abs() > threshold]
selected_features_list = selected_features.index.tolist()

# 相关性排序
sorted_selected_features = selected_features.abs().sort_values(ascending=False)
print(f"阈值: {threshold}, Pearson 筛选后剩余的特征数: {len(sorted_selected_features)}")
print(sorted_selected_features)

# 保存筛选后的特征列表
# sorted_selected_features.to_excel("sorted_selected_features.xlsx", sheet_name="Correlation", header=True)

### **新增: 预测 pCR 并计算 AUC** ###
# 提取目标变量和特征
# print('selected_features_list:', selected_features_list)

# ===============按照病人拆分=======================
# 确保ID列是字符串类型
df_numeric["ID"] = df_numeric["ID"].astype(str)

# 按病人ID获取唯一ID列表
unique_ids = df_numeric["ID"].unique()

# 按病人ID进行训练/测试集拆分
train_ids, test_ids = train_test_split(unique_ids, test_size=0.2, random_state=42)

print(train_ids)
print(test_ids)

# 选择对应病人ID的数据
train_df = df_numeric[df_numeric["ID"].isin(train_ids)]
test_df = df_numeric[df_numeric["ID"].isin(test_ids)]

# 获取特征和目标变量
X_train = train_df[selected_features_list]
y_train = train_df["pCR"]
X_test = test_df[selected_features_list]
y_test = test_df["pCR"]

# 确保只包含数值型数据
X_train = X_train.apply(pd.to_numeric, errors='coerce')
X_test = X_test.apply(pd.to_numeric, errors='coerce')
# 检查是否有无穷值
if np.isinf(X_train).any().any():
    print("X_train 存在无穷值")
if np.isinf(X_test).any().any():
    print("X_test 存在无穷值")
X_train.replace([np.inf, -np.inf], np.nan, inplace=True)
X_test.replace([np.inf, -np.inf], np.nan, inplace=True)

# 数据标准化
scaler = StandardScaler()
# 删除含有 NaN 的样本
X_train = X_train.dropna()
print(f"X_train 共有 {len(X_train)} 行")
y_train = y_train.loc[X_train.index]  # 确保 y 也同步删除对应的行
X_train_scaled = scaler.fit_transform(X_train)
# 删除含有 NaN 的样本
X_test = X_test.dropna()
print(f"X_test 共有 {len(X_test)} 行")
y_test = y_test.loc[X_test.index]  # 确保 y 也同步删除对应的行
X_test_scaled = scaler.transform(X_test)

# ===============按照病人拆分=======================

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
