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
from scipy.stats import ttest_rel
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

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
df = pd.read_excel("./data/IMPRESS/random_group_one_TNBC_v2.xlsx")

# 获取特征列，排除ID和pCR列
feature_columns = [col for col in df.columns if col not in ["ID", "pCR"]]
non_numeric_columns = df[feature_columns].select_dtypes(exclude=["number"]).columns.tolist()
df_numeric = df.drop(columns=non_numeric_columns)

# 计算 Pearson 相关性，并筛选相关性较高的特征（绝对值大于阈值）
correlation_matrix = df_numeric.corr(method="pearson")
correlation_with_pCR = correlation_matrix["pCR"].drop(["pCR", "ID"])
threshold = 0.25
selected_features = correlation_with_pCR[correlation_with_pCR.abs() > threshold]
selected_features_list = selected_features.index.tolist()
sorted_selected_features = selected_features.sort_values(ascending=False)
print(f"阈值: {threshold}, 特征数: {len(sorted_selected_features)}")
print(sorted_selected_features)

# 设置高斯噪声数据增强参数
n_augments = 1       # 生成增强副本数
noise_std = 0.01    # 高斯噪声标准差

# 原始X和y
X = df_numeric[selected_features_list]
y = df_numeric["pCR"]
ids = df_numeric["ID"]

# 数据增强函数：对X添加噪声，同时复制y和ID
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

# 计算增强后的数据
X_aug, y_aug, ids_aug = augment_X_y(X, y, ids, n_augments=n_augments, noise_std=noise_std)
print(f"原始样本数为: {X.shape[0]}, 增强后样本数: {X_aug.shape[0]}")

# 构建包含增强数据的DataFrame并保存（可选）
df_augmented = X_aug.copy()
df_augmented['pCR'] = y_aug
df_augmented['ID'] = ids_aug
output_file_aug = "./data/IMPRESS/aug/augmented_data_TNBC.xlsx"
df_augmented.to_excel(output_file_aug, index=False)
print(f"增强后的数据已保存到 {output_file_aug}")

# 获取增强后病人ID唯一值与标签（用于StratifiedKFold）
patient_df_aug = pd.DataFrame({'ID': ids_aug, 'pCR': y_aug}).groupby("ID").first().reset_index()
patient_ids = patient_df_aug["ID"].values
patient_labels = patient_df_aug["pCR"].values

# 定义Pipeline：标准化 + 逻辑回归（L1正则化）
model_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('logreg', LogisticRegression(penalty="l1", solver="liblinear", C=1.0, random_state=42))
])

# StratifiedKFold（按病人分层）
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 用于记录每一折的AUC（原始数据 vs 增强数据）
auc_list_orig = []
auc_list_aug = []

# 下面首先对原始数据（未增强）进行交叉验证
# 注意：这里需要使用原始的df中的数据（以病人为单位），假设df中每个ID只有一行数据
patient_df_orig = df.groupby("ID").first().reset_index()
patient_ids_orig = patient_df_orig["ID"].values
patient_labels_orig = patient_df_orig["pCR"].values

print("\n--- 对原始数据交叉验证 ---")
fold_index = 1
for train_patient_idx, test_patient_idx in skf.split(patient_ids_orig, patient_labels_orig):
    train_ids = patient_ids_orig[train_patient_idx]
    test_ids = patient_ids_orig[test_patient_idx]
    train_mask = df["ID"].isin(train_ids)
    test_mask = df["ID"].isin(test_ids)
    X_train = X[train_mask]
    X_test = X[test_mask]
    y_train = y[train_mask]
    y_test = y[test_mask]
    model_pipeline.fit(X_train, y_train)
    y_prob = model_pipeline.predict_proba(X_test)[:, 1]
    auc, _, _ = compute_auc_ci(y_test.values, y_prob)
    auc_list_orig.append(auc)
    print(f"原始数据 - 第 {fold_index} 折: AUC = {auc:.4f}")
    fold_index += 1

# 对增强数据进行交叉验证（增强后的数据已经按ID复制）
print("\n--- 对增强数据交叉验证 ---")
fold_index = 1
for train_patient_idx, test_patient_idx in skf.split(patient_ids, patient_labels):
    train_ids = patient_ids[train_patient_idx]
    test_ids = patient_ids[test_patient_idx]
    train_mask = ids_aug.isin(train_ids)
    test_mask = ids_aug.isin(test_ids)
    X_train_aug = X_aug[train_mask]
    X_test_aug = X_aug[test_mask]
    y_train_aug = y_aug[train_mask]
    y_test_aug = y_aug[test_mask]
    model_pipeline.fit(X_train_aug, y_train_aug)
    y_prob_aug = model_pipeline.predict_proba(X_test_aug)[:, 1]
    auc_aug, _, _ = compute_auc_ci(y_test_aug.values, y_prob_aug)
    auc_list_aug.append(auc_aug)
    print(f"增强数据 - 第 {fold_index} 折: AUC = {auc_aug:.4f}")
    fold_index += 1

# 使用配对样本t检验比较原始数据与增强数据的AUC
t_stat, p_value = ttest_rel(auc_list_aug, auc_list_orig)
print("\n--- 交叉验证AUC对比结果 ---")
print(f"原始数据 AUC: {auc_list_orig}")
print(f"增强数据 AUC: {auc_list_aug}")
print(f"t检验统计量 = {t_stat:.4f}, p值 = {p_value:.4f}")

# 判断p值
if p_value < 0.05:
    print("数据增强后模型性能提升在统计上显著 (p < 0.05)")
else:
    print("数据增强后模型性能提升在统计上不显著 (p > 0.05)")

# 可选：绘制增强数据最佳折ROC曲线（这里只绘制增强数据中的最佳折作为示例）
# 此处代码仅供参考
best_fold = np.argmax(auc_list_aug) + 1  # 仅用AUC最高的折做示例
print(f"增强数据中AUC最高的折为: 第 {best_fold} 折")
