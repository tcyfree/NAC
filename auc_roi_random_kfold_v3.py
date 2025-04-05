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

# 读取数据（请根据实际路径调整文件路径）
df = pd.read_excel("./data/IMPRESS/select_p_columns_data_TNBC.xlsx")
# df = pd.read_excel("./data/IMPRESS/select_p_columns_data_TNBC_v1.xlsx")

# 获取特征列，排除ID和pCR列
feature_columns = [col for col in df.columns if col not in ["ID", "pCR"]]

# 删除非数值列
non_numeric_columns = df[feature_columns].select_dtypes(exclude=["number"]).columns.tolist()
df_numeric = df.drop(columns=non_numeric_columns)

# 计算 Pearson 相关性，并筛选相关性较高的特征（绝对值大于阈值）
correlation_matrix = df_numeric.corr(method="pearson")
correlation_with_pCR = correlation_matrix["pCR"].drop(["pCR", "ID"])
threshold = 0.2
selected_features = correlation_with_pCR[correlation_with_pCR.abs() > threshold]
selected_features_list = selected_features.index.tolist()

# 相关性排序
sorted_selected_features = selected_features.sort_values(ascending=False)
print(f"阈值: {threshold}, 特征数: {len(sorted_selected_features)}")
print(sorted_selected_features)

# 准备特征和目标变量
X = df_numeric[selected_features_list]
y = df_numeric["pCR"]

# 定义一个Pipeline，其中包含标准化和逻辑回归（L1正则化），设置固定random_state保证可重复性
model_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('logreg', LogisticRegression(penalty="l1", solver="liblinear", C=1.0, random_state=42))
])

# 获取每位病人的唯一标签（每个病人一行）
patient_df = df.groupby("ID").first().reset_index()
patient_ids = patient_df["ID"].values
patient_labels = patient_df["pCR"].values

# 使用 StratifiedKFold 对 病人 和 标签 分层
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

best_accuracy = -1.0
best_fold = None
fold_index = 1

for train_patient_idx, test_patient_idx in skf.split(patient_ids, patient_labels):
    train_ids = patient_ids[train_patient_idx]
    test_ids = patient_ids[test_patient_idx]

    # 从原始 df 中提取这些病人的所有数据（按 ID 匹配）
    train_mask = df["ID"].isin(train_ids)
    test_mask = df["ID"].isin(test_ids)

    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]

    model_pipeline.fit(X_train, y_train)
    y_pred = model_pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    y_prob = model_pipeline.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob)

    # # 统计 pCR 分布
    # pos_count = y_test.sum()
    # neg_count = len(y_test) - pos_count
    # print(f"第 {fold_index} 折 - 正类(pCR=1): {pos_count}，负类(pCR=0): {neg_count}")

    print(f"\n第 {fold_index} 折 (病人数: 训练 {len(train_ids)}, 测试 {len(test_ids)})")
    print(f"Accuracy: {acc:.4f}, AUC: {auc:.4f}")
    print(f"训练ID: {train_ids}")
    print(f"测试ID: {test_ids}")
    
    if acc > best_accuracy:
        best_accuracy = acc
        best_fold = fold_index
        best_y_test = y_test
        best_y_prob = y_prob
        best_model = Pipeline([
            ('scaler', StandardScaler()),
            ('logreg', LogisticRegression(penalty="l1", solver="liblinear", C=1.0, random_state=42))
        ])
        best_model.fit(X_train, y_train)

    fold_index += 1

print(f"\n最高 Accuracy 出现在第 {best_fold} 折，Accuracy = {best_accuracy:.4f}")

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
# output_file = "./data/IMPRESS/best_fold_weights_StratifiedKFold_TNBC.xlsx"
# df_weights.to_excel(output_file, index=False)
# print(f"最佳折模型的权重已保存到 {output_file}")
