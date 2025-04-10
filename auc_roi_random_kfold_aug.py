import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import StratifiedKFold

# 计算置信区间
def compute_auc_ci(y_true, y_prob, n_bootstraps=1000, seed=42):
    rng = np.random.RandomState(seed)
    bootstrapped_scores = []

    for i in range(n_bootstraps):
        indices = rng.randint(0, len(y_true), len(y_true))
        if len(np.unique(y_true[indices])) < 2:
            continue
        score = roc_auc_score(y_true[indices], y_prob[indices])
        bootstrapped_scores.append(score)

    sorted_scores = np.array(bootstrapped_scores)
    sorted_scores.sort()
    lower = sorted_scores[int(0.025 * len(sorted_scores))]
    upper = sorted_scores[int(0.975 * len(sorted_scores))]
    return roc_auc_score(y_true, y_prob), lower, upper

# 读取数据（请根据实际路径调整文件路径）
# df = pd.read_excel("./data/IMPRESS/select_p_columns_data_HER2.xlsx")
# df = pd.read_excel("./data/IMPRESS/select_p_columns_data_TNBC.xlsx")
# df = pd.read_excel("./data/IMPRESS/select_p_columns_data_TNBC_v1.xlsx")
df = pd.read_excel("./data/IMPRESS/random_group_one_TNBC_v2.xlsx")

# 获取特征列，排除ID和pCR列
feature_columns = [col for col in df.columns if col not in ["ID", "pCR"]]

# 删除非数值列
non_numeric_columns = df[feature_columns].select_dtypes(exclude=["number"]).columns.tolist()
df_numeric = df.drop(columns=non_numeric_columns)

# 计算 Pearson 相关性，并筛选相关性较高的特征（绝对值大于阈值）
correlation_matrix = df_numeric.corr(method="pearson")
correlation_with_pCR = correlation_matrix["pCR"].drop(["pCR", "ID"])
threshold = 0.25
selected_features = correlation_with_pCR[correlation_with_pCR.abs() > threshold]
selected_features_list = selected_features.index.tolist()

# 相关性排序
sorted_selected_features = selected_features.sort_values(ascending=False)
print(f"阈值: {threshold}, 特征数: {len(sorted_selected_features)}")
print(sorted_selected_features)

# 保存筛选后的特征列表
# sorted_selected_features.to_excel("./data/IMPRESS/sorted_selected_features_CRCR_TNBC_p.xlsx", sheet_name="Correlation", header=True)

# 设置增强参数
n_augments = 1       # 生成几个增强副本
noise_std = 0.001     # 高斯噪声标准差

# 原始X和y
X = df_numeric[selected_features_list]
y = df_numeric["pCR"]
ids = df_numeric["ID"]

# 数据增强函数
def augment_X_y(X, y, ids, n_augments=2, noise_std=0.01):
    """
    同时增强X和y（以及ID），返回增强后的X, y, ID
    """
    X_list = [X]
    y_list = [y]
    id_list = [ids]

    for i in range(n_augments):
        noise = np.random.normal(loc=0, scale=noise_std, size=X.shape)
        X_aug = X + noise
        X_list.append(pd.DataFrame(X_aug, columns=X.columns))
        y_list.append(y.copy())
        id_list.append(ids.copy())

    X_augmented = pd.concat(X_list, ignore_index=True)
    y_augmented = pd.concat(y_list, ignore_index=True)
    ids_augmented = pd.concat(id_list, ignore_index=True)

    return X_augmented, y_augmented, ids_augmented

# 应用数据增强
X_aug, y_aug, ids_aug = augment_X_y(X, y, ids, n_augments=n_augments, noise_std=noise_std)

print(f"原始样本数为: {X.shape[0]}, 增强后样本数: {X_aug.shape[0]}")
print(f"增强后ID样本数: {ids_aug.shape[0]}, 标签样本数: {y_aug.shape[0]}")

# 获取增强后病人ID唯一值与标签（用于StratifiedKFold）
patient_df_aug = pd.DataFrame({'ID': ids_aug, 'pCR': y_aug}).groupby("ID").first().reset_index()
patient_ids = patient_df_aug["ID"].values
patient_labels = patient_df_aug["pCR"].values

# 定义一个Pipeline，其中包含标准化和逻辑回归（L1正则化），设置固定random_state保证可重复性
model_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('logreg', LogisticRegression(penalty="l1", solver="liblinear", C=1.0, random_state=42))
])

# StratifiedKFold 按照病人ID和pCR标签做分层
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

best_accuracy = -1.0
best_fold = None
best_fpr = None
best_tpr = None
fold_index = 1
best_lower_ci = 0
best_upper_ci = 0

for train_patient_idx, test_patient_idx in skf.split(patient_ids, patient_labels):
    train_ids = patient_ids[train_patient_idx]
    test_ids = patient_ids[test_patient_idx]

    # 匹配增强后的训练集和测试集
    train_mask = ids_aug.isin(train_ids)
    test_mask = ids_aug.isin(test_ids)

    X_train, X_test = X_aug[train_mask], X_aug[test_mask]
    y_train, y_test = y_aug[train_mask], y_aug[test_mask]
    print(f"\n第 {fold_index} 折 - X_train： {len(X_train)}, X_test： {len(X_test)}")

    model_pipeline.fit(X_train, y_train)
    y_pred = model_pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    y_prob = model_pipeline.predict_proba(X_test)[:, 1]
    auc, lower_ci, upper_ci = compute_auc_ci(y_test.values, y_prob)
    print(f"Accuracy: {acc:.4f}, AUC: {auc:.4f} (95% CI: {lower_ci:.4f} - {upper_ci:.4f})")

    if acc > best_accuracy:
        best_lower_ci = lower_ci
        best_upper_ci = upper_ci
        best_accuracy = acc
        best_fold = fold_index
        best_y_test = y_test
        best_y_prob = y_prob
        best_fpr, best_tpr, _ = roc_curve(y_test, y_prob)
        best_model = Pipeline([
            ('scaler', StandardScaler()),
            ('logreg', LogisticRegression(penalty="l1", solver="liblinear", C=1.0, random_state=42))
        ])
        best_model.fit(X_train, y_train)

    fold_index += 1

print(f"\n最高 Accuracy 出现在第 {best_fold} 折，Accuracy = {best_accuracy:.4f}")

# ------------------------------
# 保存最佳折的ROC曲线数据和性能指标到同一Excel文件中
# ------------------------------
# 构造ROC数据的DataFrame
roc_data = pd.DataFrame({
    'False Positive Rate': best_fpr,
    'True Positive Rate': best_tpr
})

auc_score = roc_auc_score(best_y_test, best_y_prob)

# 构造性能指标的DataFrame（AUC和Accuracy）
performance_metrics = pd.DataFrame({
    'Metric': ['Accuracy', 'AUC', 'AUC_CI_Lower', 'AUC_CI_Upper'],
    'Value': [best_accuracy, auc_score, best_lower_ci, best_upper_ci]
})

# 定义保存的Excel文件路径
output_file = "./data/IMPRESS/roc_curve_data_best_fold_one_aug_TNBC_v1.xlsx"

# 使用ExcelWriter将数据保存到同一Excel文件的不同sheet中
with pd.ExcelWriter(output_file) as writer:
    roc_data.to_excel(writer, sheet_name="ROC Data", index=False)
    performance_metrics.to_excel(writer, sheet_name="Metrics", index=False)

print(f"最佳折ROC数据和性能指标已保存到 {output_file}")


# 绘制最佳折的ROC曲线
plt.figure(figsize=(6, 6))
plt.plot(best_fpr, best_tpr, label=f"AUC = {roc_auc_score(best_y_test, best_y_prob):.4f} (95% CI: {best_lower_ci:.4f} - {best_upper_ci:.4f}) \n Accuracy = {best_accuracy:.4f}", color="blue")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(f"ROC Curve - Best Fold (Fold {best_fold})")
plt.legend()
plt.show()

# # 可视化最佳模型的特征权重（L1正则后稀疏的系数）
# logreg_model = best_model.named_steps['logreg']
# scaler = best_model.named_steps['scaler']
# coef = logreg_model.coef_.flatten()

# # 只展示非零系数的特征
# non_zero_mask = coef != 0
# non_zero_features = np.array(selected_features_list)[non_zero_mask]
# non_zero_coefs = coef[non_zero_mask]

# # 排序展示
# # sorted_idx = np.argsort(non_zero_coefs)
# sorted_idx = np.argsort(np.abs(non_zero_coefs))[::-1]  # 从大到小
# sorted_features = non_zero_features[sorted_idx]
# sorted_coefs = non_zero_coefs[sorted_idx]

# plt.figure(figsize=(8, 6))
# sns.barplot(x=sorted_coefs, y=sorted_features, color='b')
# plt.title(f"L1 Logistic Regression Coefficients (Fold {best_fold})")
# plt.xlabel("Coefficient Value")
# plt.ylabel("Feature")
# plt.axvline(0, color='black', linestyle='--', linewidth=1)
# plt.tight_layout()
# plt.show()

# # ------------------------------
# # 保存最佳折模型的权重
# # ------------------------------
# # 提取最佳模型中的逻辑回归部分
# logreg_model = best_model.named_steps['logreg']

# # 系数和截距
# coefficients = logreg_model.coef_.flatten()
# intercept = logreg_model.intercept_[0]

# # 构造DataFrame保存权重，注意X.columns与selected_features_list一致
# df_weights = pd.DataFrame({
#     "Feature": X.columns,
#     "Coefficient": coefficients
# })
# # 将截距单独存为一行
# df_intercept = pd.DataFrame({
#     "Feature": ["Intercept"],
#     "Coefficient": [intercept]
# })

# df_weights = pd.concat([df_intercept, df_weights], ignore_index=True)

# # 保存到Excel文件中（路径可根据需要修改）
# output_file = "./data/IMPRESS/best_fold_weights_StratifiedKFold_one_aug_TNBC_v1.xlsx"
# df_weights.to_excel(output_file, index=False)
# print(f"最佳折模型的权重已保存到 {output_file}")
