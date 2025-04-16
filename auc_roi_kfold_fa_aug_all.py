import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

# 读取数据（请根据实际路径调整文件路径）
df = pd.read_excel("./data/IMPRESS/cluster/cluster_factors_pearson_6.xlsx")

# 获取特征列，排除ID和pCR列
feature_columns = [col for col in df.columns if col not in ["ID", "pCR"]]

# 删除非数值列
non_numeric_columns = df[feature_columns].select_dtypes(exclude=["number"]).columns.tolist()
df_numeric = df.drop(columns=non_numeric_columns)

# 计算 Pearson 相关性，并筛选相关性较高的特征（绝对值大于阈值）
correlation_matrix = df_numeric.corr(method="pearson")
correlation_with_pCR = correlation_matrix["pCR"].drop(["pCR", "ID"])
threshold = 0
selected_features = correlation_with_pCR[correlation_with_pCR.abs() > threshold]
selected_features_list = selected_features.index.tolist()

# 相关性排序
sorted_selected_features = selected_features.sort_values(ascending=False)
print(f"阈值: {threshold}, 特征数: {len(sorted_selected_features)}")
print(sorted_selected_features)

# 准备特征和目标变量
X = df_numeric[selected_features_list]
y = df_numeric["pCR"]

# 定义一个Pipeline，其中包含标准化和逻辑回归（L1正则化）
model_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('logreg', LogisticRegression(penalty="l1", solver="liblinear", C=1.0, random_state=42))
])

# 获取每位病人的唯一标签（每个病人一行）
patient_df = df.groupby("ID").first().reset_index()
patient_ids = patient_df["ID"].values
patient_labels = patient_df["pCR"].values

# 设置增强参数，仅用于训练数据
n_augments = 1       # 生成几个增强副本
noise_std = 0.01     # 高斯噪声标准差

# 数据增强函数：对输入数据X、标签y和ID添加高斯噪声生成增强副本
def augment_X_y(X, y, ids, n_augments=1, noise_std=0.01):
    """
    对X添加高斯噪声，复制y和ids，返回增强后的X, y, ids
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

# StratifiedKFold 按病人分层
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 用于记录每个折的指标
auc_list = []
accuracy_list = []
f1_list = []
precision_list = []
recall_list = []
npv_list = []

best_auc = -1.0
best_accuracy = -1.0
best_fold = None
best_fpr = None
best_tpr = None
fold_index = 1

# 计算置信区间函数
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

# 在每个折中，仅对训练数据进行增强
for train_patient_idx, test_patient_idx in skf.split(patient_ids, patient_labels):
    train_ids = patient_ids[train_patient_idx]
    test_ids = patient_ids[test_patient_idx]

    # 从原始df中提取对应病人的数据（训练和测试）
    train_mask = df["ID"].isin(train_ids)
    test_mask = df["ID"].isin(test_ids)
    X_train = X[train_mask]
    y_train = y[train_mask]
    X_test = X[test_mask]
    y_test = y[test_mask]
    
    # 仅对训练数据进行数据增强
    X_train_aug, y_train_aug, _ = augment_X_y(X_train, y_train, df.loc[train_mask, "ID"], 
                                              n_augments=n_augments, noise_std=noise_std)
    
    print(f"\n第 {fold_index} 折 - 原始训练样本数: {len(X_train)}, 增强后训练样本数: {len(X_train_aug)}, 测试样本数: {len(X_test)}")
    
    # 用增强后的训练数据训练模型，测试数据保持原始状态
    model_pipeline.fit(X_train_aug, y_train_aug)
    y_pred = model_pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    y_prob = model_pipeline.predict_proba(X_test)[:, 1]
    auc, lower_ci, upper_ci = compute_auc_ci(y_test.values, y_prob)
    print(f"Accuracy: {acc:.4f}, AUC: {auc:.4f} (95% CI: {lower_ci:.4f} - {upper_ci:.4f})")
    
    # 计算 F1 score、Precision、Recall
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    
    # 计算 NPV：根据混淆矩阵 TN/(TN+FN)
    cm = confusion_matrix(y_test, y_pred)
    # 混淆矩阵格式: [[TN, FP],
    #               [FN, TP]]
    TN = cm[0, 0]
    FN = cm[1, 0]
    npv = TN / (TN + FN) if (TN + FN) > 0 else np.nan
    
    print(f"F1 Score: {f1:.4f}, Precision (PPV): {precision:.4f}, Recall: {recall:.4f}, NPV: {npv:.4f}")
    
    # 保存本折指标
    auc_list.append(auc)
    accuracy_list.append(acc)
    f1_list.append(f1)
    precision_list.append(precision)
    recall_list.append(recall)
    npv_list.append(npv)
    
    # 更新最佳折记录（以 AUC 为选择依据）
    if auc > best_auc:
        best_auc = auc
        best_accuracy = acc
        best_fold = fold_index
        best_y_test = y_test.copy()
        best_y_prob = y_prob.copy()
        best_y_pred = y_pred.copy()  # 保存最佳折预测结果，用于后续混淆矩阵绘制及指标计算
        best_fpr, best_tpr, _ = roc_curve(y_test, y_prob)
        best_model = Pipeline([
            ('scaler', StandardScaler()),
            ('logreg', LogisticRegression(penalty="l1", solver="liblinear", C=1.0, random_state=42))
        ])
        best_model.fit(X_train_aug, y_train_aug)
    
    fold_index += 1

# 计算各指标的均值和标准差（交叉验证汇总）
def mean_std(metric_list):
    return np.mean(metric_list), np.std(metric_list)

auc_mean, auc_std = mean_std(auc_list)
acc_mean, acc_std = mean_std(accuracy_list)
f1_mean, f1_std = mean_std(f1_list)
precision_mean, precision_std = mean_std(precision_list)
recall_mean, recall_std = mean_std(recall_list)
npv_mean, npv_std = mean_std(npv_list)

print("\n================= 交叉验证结果 =================")
print(f"AUC: {auc_mean:.4f} ± {auc_std:.4f}")
print(f"Accuracy: {acc_mean:.4f} ± {acc_std:.4f}")
print(f"F1 Score: {f1_mean:.4f} ± {f1_std:.4f}")
print(f"Precision (PPV): {precision_mean:.4f} ± {precision_std:.4f}")
print(f"Recall: {recall_mean:.4f} ± {recall_std:.4f}")
print(f"NPV: {npv_mean:.4f} ± {npv_std:.4f}")

print(f"\n最高 AUC 出现在第 {best_fold} 折，AUC = {best_auc:.4f}")

# 计算最佳折的各项指标
# 这里 sensitivity 与 Recall 相同，specificity = TN/(TN+FP)
best_cm = confusion_matrix(best_y_test, best_y_pred)
TN, FP, FN, TP = best_cm[0, 0], best_cm[0, 1], best_cm[1, 0], best_cm[1, 1]
sensitivity = TP / (TP + FN) if (TP + FN) > 0 else np.nan
specificity = TN / (TN + FP) if (TN + FP) > 0 else np.nan

# 分别计算其他指标（基于最佳折测试集）
f1_best = f1_score(best_y_test, best_y_pred)
precision_best = precision_score(best_y_test, best_y_pred)
recall_best = recall_score(best_y_test, best_y_pred)  # 与 sensitivity 相同
npv_best = TN / (TN + FN) if (TN + FN) > 0 else np.nan
auc_best = best_auc
accuracy_best = best_accuracy

# 打印最佳折指标
print("\n================= 最佳折指标 =================")
print(f"AUC: {auc_best:.4f}")
print(f"Accuracy: {accuracy_best:.4f}")
print(f"F1 Score: {f1_best:.4f}")
print(f"Precision (PPV): {precision_best:.4f}")
print(f"Recall (Sensitivity): {recall_best:.4f}")
print(f"NPV: {npv_best:.4f}")
print(f"Specificity: {specificity:.4f}")

# 绘制最佳折的ROC曲线
plt.figure(figsize=(6, 6))
plt.plot(best_fpr, best_tpr, label=f"AUC = {roc_auc_score(best_y_test, best_y_prob):.4f}  | Accuracy = {accuracy_best:.4f}", color="blue")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(f"ROC Curve - Best Fold (Fold {best_fold})")
plt.legend()
# plt.savefig('./results/auc.pdf')
plt.show()

# 绘制最佳折的混淆矩阵
plt.figure(figsize=(6, 5))
sns.heatmap(best_cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Predicted Negative", "Predicted Positive"],
            yticklabels=["Actual Negative", "Actual Positive"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title(f"Confusion Matrix - Best Fold (Fold {best_fold})")
plt.tight_layout()
# plt.savefig('./results/confusion_matrix.pdf')
plt.show()

# 可视化最佳模型的特征权重（L1正则后稀疏的系数）
logreg_model = best_model.named_steps['logreg']
coef = logreg_model.coef_.flatten()
non_zero_mask = coef != 0
non_zero_features = np.array(selected_features_list)[non_zero_mask]
non_zero_coefs = coef[non_zero_mask]
sorted_idx = np.argsort(np.abs(non_zero_coefs))[::-1]
sorted_features = non_zero_features[sorted_idx]
sorted_coefs = non_zero_coefs[sorted_idx]

plt.figure(figsize=(6, 6))
sns.barplot(x=sorted_coefs, y=sorted_features, color='b')
plt.title(f"L1 Logistic Regression Coefficients (Fold {best_fold})")
plt.xlabel("Coefficient Value")
plt.ylabel("Feature")
plt.axvline(0, color='black', linestyle='--', linewidth=1)
plt.tight_layout()
plt.show()

# # ------------------------------
# # 导出交叉验证结果及各折指标至 Excel
# # ------------------------------
# results_df = pd.DataFrame({
#     "Fold": list(range(1, len(auc_list)+1)),
#     "AUC": auc_list,
#     "Accuracy": accuracy_list,
#     "F1 Score": f1_list,
#     "Precision (PPV)": precision_list,
#     "Recall": recall_list,
#     "NPV": npv_list
# })
# summary_df = pd.DataFrame({
#     "Metric": ["AUC", "Accuracy", "F1 Score", "Precision (PPV)", "Recall", "NPV"],
#     "Mean": [auc_mean, acc_mean, f1_mean, precision_mean, recall_mean, npv_mean],
#     "Std": [auc_std, acc_std, f1_std, precision_std, recall_std, npv_std]
# })
# output_file = "./data/IMPRESS/cv_results.xlsx"
# with pd.ExcelWriter(output_file) as writer:
#     results_df.to_excel(writer, sheet_name="Fold Results", index=False)
#     summary_df.to_excel(writer, sheet_name="Summary", index=False)
# print(f"交叉验证结果已保存到 {output_file}")
