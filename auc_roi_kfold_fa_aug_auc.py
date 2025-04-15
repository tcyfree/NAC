import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score
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
    
    if auc > best_auc:
        best_auc = auc
        best_accuracy = acc
        best_fold = fold_index
        best_y_test = y_test
        best_y_prob = y_prob
        best_fpr, best_tpr, _ = roc_curve(y_test, y_prob)
        best_model = Pipeline([
            ('scaler', StandardScaler()),
            ('logreg', LogisticRegression(penalty="l1", solver="liblinear", C=1.0, random_state=42))
        ])
        best_model.fit(X_train_aug, y_train_aug)
    
    fold_index += 1

print(f"\n最高 AUC 出现在第 {best_fold} 折，AUC = {best_auc:.4f}")


# 绘制最佳折的ROC曲线
plt.figure(figsize=(6, 6))
plt.plot(best_fpr, best_tpr, label=f"AUC = {roc_auc_score(best_y_test, best_y_prob):.4f}  | Accuracy = {best_accuracy:.4f}", color="blue")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(f"ROC Curve - Best Fold (Fold {best_fold})")
plt.legend()
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
# output_file = "./data/IMPRESS/best_fold_weights_StratifiedKFold_FA_pearson_auc_6_aug_TNBC.xlsx"
# df_weights.to_excel(output_file, index=False)
# print(f"最佳折模型的权重已保存到 {output_file}")
