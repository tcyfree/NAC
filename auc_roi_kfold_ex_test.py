import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score
from sklearn.preprocessing import StandardScaler

# ------------------------------
# 1. 加载模型权重
# ------------------------------
weights_file = "./data/IMPRESS/best_fold_weights_StratifiedKFold_TNBC.xlsx"
df_weights = pd.read_excel(weights_file)
print("加载的权重模型：")
print(df_weights)

# 提取截距与各特征系数
intercept = df_weights.loc[df_weights["Feature"] == "Intercept", "Coefficient"].values[0]
# 剔除截距行，构造特征-系数字典
coef_dict = df_weights[df_weights["Feature"] != "Intercept"].set_index("Feature")["Coefficient"].to_dict()

# ------------------------------
# 2. 加载并预处理外部测试集
# ------------------------------
external_file = "./data/Post-NAT-BRCA/random_group_eight_Post_TNBC_v1.1.xlsx"  # 根据实际情况调整路径
df_external = pd.read_excel(external_file)

# 假设外部测试集中包含相同的特征和目标变量（pCR）
# 根据模型权重中的特征名称选择特征
selected_features = list(coef_dict.keys())

# 检查外部测试集是否包含所有模型要求的特征
available_features = [f for f in selected_features if f in df_external.columns]
if len(available_features) < len(selected_features):
    missing = set(selected_features) - set(available_features)
    print(f"警告：外部测试集缺失以下特征：{missing}")

# 选择外部测试集中的特征数据和目标变量
X_external = df_external[available_features]
y_external = df_external["pCR"]

# 由于模型训练时对数据进行了标准化（但未保存Scaler参数），
# 这里为了尽量保持数据一致性，我们对外部测试集进行标准化处理。
# 注意：理想情况下应使用训练集的均值和方差参数
scaler = StandardScaler()
X_external_scaled = scaler.fit_transform(X_external)

# 重新构造DataFrame，保证列顺序与模型权重中的顺序一致
X_external_scaled_df = pd.DataFrame(X_external_scaled, columns=available_features)
X_external_scaled_df = X_external_scaled_df[selected_features]  # 这里selected_features与available_features顺序保持一致

# ------------------------------
# 3. 使用权重模型进行预测
# ------------------------------
# 手动计算线性组合 z = intercept + Σ(系数 * 特征值)
z = intercept + np.dot(X_external_scaled_df.values, np.array([coef_dict[feat] for feat in selected_features]))
# 计算预测概率
y_prob = 1 / (1 + np.exp(-z))
# 根据0.5阈值确定预测类别
y_pred = (y_prob >= 0.5).astype(int)

# ------------------------------
# 4. 模型评估
# ------------------------------
accuracy = accuracy_score(y_external, y_pred)
auc = roc_auc_score(y_external, y_prob)
print(f"外部测试集 Accuracy: {accuracy:.4f}, AUC: {auc:.4f}")

# # 绘制 ROC 曲线
# fpr, tpr, thresholds = roc_curve(y_external, y_prob)
# plt.figure(figsize=(8, 6))
# plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}", color="blue")
# plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
# plt.xlabel("False Positive Rate")
# plt.ylabel("True Positive Rate")
# plt.title("外部测试集 ROC 曲线")
# plt.legend()
# plt.show()
