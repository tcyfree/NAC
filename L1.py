import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

# 读取数据
meta_df = pd.read_excel("cohort_meta_TNBC.xlsx")
features_df = pd.read_excel("./data/IMPRESS/TNBC/perDatasetSlideSummaries/RoiFeatureSummary_Means.xlsx")

# 合并数据
df = pd.merge(meta_df, features_df, on="ID", how="inner")

# 处理缺失值（填充为均值）
imputer = SimpleImputer(strategy="mean")
feature_columns = [col for col in features_df.columns if col != "ID"]
df[feature_columns] = imputer.fit_transform(df[feature_columns])

# 选择特征和目标变量
X = df[feature_columns]
y = df["pCR"]

# 标准化特征
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 拆分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练 L1 正则化逻辑回归
model = LogisticRegression(penalty="l1", solver="liblinear", C=1.0)  # C 控制正则化强度
model.fit(X_train, y_train)

# 获取非零系数的特征
selected_features = np.array(feature_columns)[model.coef_[0] != 0]
coefficients = model.coef_[0][model.coef_[0] != 0]

# 结果整理
results_df = pd.DataFrame({
    "Feature": selected_features,
    "Coefficient": coefficients,
    "Odds Ratio": np.exp(coefficients)
})

# 按绝对值排序
results_df = results_df.reindex(results_df["Coefficient"].abs().sort_values(ascending=False).index)
print(results_df)

# 保存结果
output_filename = "./results/IMPRESS/TNBC_L1_FeatureSelection.xlsx"
results_df.to_excel(output_filename, index=False)
print(f"分析结果已保存到 {output_filename}")
