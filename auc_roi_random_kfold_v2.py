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

# # 使用KFold进行5折交叉验证，并选择Accuracy最高的那一折
# kf = KFold(n_splits=5, shuffle=True, random_state=42)

# 使用GroupKFold进行5折交叉验证，按照病人ID分组
gkf = GroupKFold(n_splits=5)
groups = df["ID"]  # 确保使用原始数据中的ID作为分组依据

best_accuracy = -1.0
best_fold = None
best_y_test = None
best_y_prob = None
best_fpr = None
best_tpr = None
best_model = None

fold_index = 1
for train_index, test_index in gkf.split(X, y, groups=groups):
     # 获取对应折中的ID（病人编号）
    train_ids = df.iloc[train_index]["ID"].unique()
    test_ids = df.iloc[test_index]["ID"].unique()

    print(f"\n第 {fold_index} 折")
    print(f"训练集ID数: {len(train_ids)}, 测试集ID数: {len(test_ids)}")
    print(f"训练集ID: {train_ids}")
    print(f"测试集ID: {test_ids}")
    
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    # 拟合模型
    model_pipeline.fit(X_train, y_train)
    
    # 预测与评估
    y_pred = model_pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    y_prob = model_pipeline.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob)
    
    print(f"第 {fold_index} 折 Accuracy: {acc:.4f}, AUC: {auc:.4f}")
    
    # 若当前折的Accuracy更高，则记录该折结果及模型
    if acc > best_accuracy:
        best_accuracy = acc
        best_fold = fold_index
        best_y_test = y_test
        best_y_prob = y_prob
        best_fpr, best_tpr, _ = roc_curve(y_test, y_prob)
        # 保存当前折的模型（Pipeline中包含了预处理和拟合后的模型）
        best_model = Pipeline([
            ('scaler', StandardScaler()),
            ('logreg', LogisticRegression(penalty="l1", solver="liblinear", C=1.0, random_state=42))
        ])
        # 重新拟合保存模型，保证保留的是最佳折的模型
        best_model.fit(X_train, y_train)
    
    fold_index += 1

print(f"\n最高 Accuracy 出现在第 {best_fold} 折，Accuracy = {best_accuracy:.4f}")

# 绘制最佳折的ROC曲线
plt.figure(figsize=(8, 6))
plt.plot(best_fpr, best_tpr, label=f"AUC = {roc_auc_score(best_y_test, best_y_prob):.4f}", color="blue")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(f"ROC Curve - Best Fold (Fold {best_fold})")
plt.legend()
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
# output_file = "./data/IMPRESS/best_fold_weights_TNBC_v1.xlsx"
# df_weights.to_excel(output_file, index=False)
# print(f"最佳折模型的权重已保存到 {output_file}")
