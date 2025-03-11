import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve

# 读取数据
meta_df = pd.read_excel("cohort_meta_TNBC.xlsx")
means_df = pd.read_excel("./data/IMPRESS/TNBC/perDatasetSlideSummaries/RoiFeatureSummary_Means.xlsx")

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
selected_features = correlation_with_pCR[correlation_with_pCR.abs() > 0.3]
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
# 打印选择的列
# print('selected_features_list:', selected_features_list)

### **新增: 留一法交叉验证（LOO-CV）设置下的 AUC 计算** ###
X = df_numeric[selected_features_list]
y = df_numeric["pCR"]

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

loo = LeaveOneOut()
n_splits = loo.get_n_splits(X_scaled)

auc_scores = []

for seed in range(20):  # 20 次不同随机种子的实验
    np.random.seed(seed)
    y_true, y_pred_prob = [], []
    
    for train_index, test_index in loo.split(X_scaled):
        X_train, X_test = X_scaled[train_index], X_scaled[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        model = LogisticRegression(penalty="l1", solver="liblinear", C=1.0, random_state=seed)
        model.fit(X_train, y_train)
        
        y_true.append(y_test.iloc[0])
        y_pred_prob.append(model.predict_proba(X_test)[:, 1][0])
    
    auc_score = roc_auc_score(y_true, y_pred_prob)
    auc_scores.append(auc_score)
    print(f"随机种子 {seed} 的 AUC: {auc_score:.4f}")

print(f"平均 AUC: {np.mean(auc_scores):.4f} ± {np.std(auc_scores):.4f}")

# 绘制 ROC 曲线
fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
plt.figure(figsize=(5, 4))
plt.plot(fpr, tpr, label=f"Mean AUC = {np.mean(auc_scores):.4f}", color="blue")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()
