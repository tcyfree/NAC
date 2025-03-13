import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve

# 读取随机分（4份）好的xlsx 文件
df = pd.read_excel("output_random_group.xlsx")

# 选择特征列（排除ID和pCR）
feature_columns = [col for col in df.columns if col not in ["ID", "pCR"]]

# 筛选出数值型列（删除非数值列）
non_numeric_columns = df[feature_columns].select_dtypes(exclude=["number"]).columns.tolist()
print(f"非数值列: {non_numeric_columns}")
df_numeric = df.drop(columns=non_numeric_columns)
print(f"df_numeric 共有 {df_numeric.shape[0]} 行，{df_numeric.shape[1]} 列")
print(f"df_numeric 共有 {len(df_numeric)} 行")

# 计算Pearson相关性，并筛选绝对相关性大于0.2的特征
correlation_matrix = df_numeric.corr(method="pearson")
correlation_with_pCR = correlation_matrix["pCR"].drop("pCR")
selected_features = correlation_with_pCR[correlation_with_pCR.abs() > 0.35]
selected_features_list = selected_features.index.tolist()
print(f"Pearson 筛选后剩余的特征数: {len(selected_features_list)}")
print(selected_features.abs().sort_values(ascending=False))

# 提取特征数据和目标变量
# print('selected_features_list:', selected_features_list)
X = df_numeric[selected_features_list]
y = df_numeric["pCR"]

# 检查缺失值，并删除含有NaN的样本
missing_values = X.isna().sum()
print(f"缺失值统计：\n{missing_values[missing_values > 0]}")
X = X.dropna()
y = y.loc[X.index]

# 确保ID列为字符串格式（用于按病人拆分）
df_numeric["ID"] = df_numeric["ID"].astype(str)
unique_ids = df_numeric["ID"].unique()

# 重复20次实验，每次随机打乱病人ID顺序后进行LOO交叉验证
total_aucs = []
for seed in range(20):
    np.random.seed(seed)
    unique_ids_shuffled = unique_ids.copy()
    np.random.shuffle(unique_ids_shuffled)
    
    # 用于汇总所有LOO折的测试标签和预测概率
    y_true_all = []
    y_pred_all = []
    
    loo = LeaveOneOut()
    # LOO划分：每次将一个病人作为测试集，其余作为训练集
    for train_idx, test_idx in loo.split(unique_ids_shuffled):
        train_ids = unique_ids_shuffled[train_idx]
        test_ids = unique_ids_shuffled[test_idx]
        # print('test_ids:',test_ids)
        
        # 根据病人ID选择数据
        train_df = df_numeric[df_numeric["ID"].isin(train_ids)]
        test_df = df_numeric[df_numeric["ID"].isin(test_ids)]
        
        X_train, y_train = train_df[selected_features_list], train_df["pCR"]
        X_test, y_test = test_df[selected_features_list], test_df["pCR"]
        
        # 删除含有NaN的样本（确保训练和测试数据一致）
        X_train = X_train.dropna()
        y_train = y_train.loc[X_train.index]
        X_test = X_test.dropna()
        y_test = y_test.loc[X_test.index]
        
        # 如果当前折数据为空，则跳过
        if len(X_train) == 0 or len(X_test) == 0:
            continue
        
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

        # 数据标准化：只使用训练集参数
        scaler = StandardScaler()
        # 删除含有 NaN 的样本
        X_train = X_train.dropna()
        # print(f"X_train 共有 {len(X_train)} 行")
        y_train = y_train.loc[X_train.index]  # 确保 y 也同步删除对应的行
        X_train_scaled = scaler.fit_transform(X_train)
        # 删除含有 NaN 的样本
        X_test = X_test.dropna()
        # print(f"X_test 共有 {len(X_test)} 行")
        y_test = y_test.loc[X_test.index]  # 确保 y 也同步删除对应的行
        X_test_scaled = scaler.transform(X_test)
        
        # 训练逻辑回归模型（L1正则化）
        model = LogisticRegression(penalty="l1", solver="liblinear", C=1.0)
        model.fit(X_train_scaled, y_train)
        
        # 预测测试集概率
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
        
        # 将当前折的真实标签和预测概率汇总
        y_true_all.extend(y_test.tolist())
        y_pred_all.extend(y_prob.tolist())
    
    # 对当前随机种子下所有LOO折的结果计算整体AUC
    if len(np.unique(y_true_all)) < 2:
        seed_auc = np.nan
        print(f"Seed {seed}: 无法计算AUC（测试样本类别单一）")
    else:
        seed_auc = roc_auc_score(y_true_all, y_pred_all)
        print(f"Seed {seed}: AUC = {seed_auc:.4f}")
    total_aucs.append(seed_auc)

# 输出20次实验的平均AUC及标准差（忽略nan值）
valid_aucs = [auc for auc in total_aucs if not np.isnan(auc)]
mean_auc = np.mean(valid_aucs) if valid_aucs else np.nan
std_auc = np.std(valid_aucs) if valid_aucs else np.nan
print(f"Final Mean AUC: {mean_auc:.4f} ± {std_auc:.4f}")

# 绘制最后一次实验中任一折的ROC曲线作为示例（若存在有效数据）
if len(y_true_all) > 0 and len(np.unique(y_true_all)) > 1:
    fpr, tpr, _ = roc_curve(y_true_all, y_pred_all)
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_true_all, y_pred_all):.4f}", color="blue")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.show()
