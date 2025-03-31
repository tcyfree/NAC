import pandas as pd
import numpy as np
import statsmodels.api as sm

# # 读取数据
# meta_df = pd.read_excel("cohort_meta_TNBC.xlsx")
# # features_df = pd.read_excel("RegionFeatureSummary_Means.xls")
# # features_df = pd.read_excel("RoiFeatureSummary_Means.xls")
# features_df = pd.read_excel("./data/IMPRESS/TNBC/perDatasetSlideSummaries/RoiFeatureSummary_Means.xlsx")

# # 合并数据
# df = pd.merge(meta_df, features_df, on="ID", how="inner")  # 确保ID列名称一致

df = pd.read_excel("./data/IMPRESS/random_group_eight_TNBC_v3.xlsx")
# df = pd.read_excel("./data/IMPRESS/random_group_eight_HER2_v3.xlsx")

# 检查基本情况
print(df.shape)

# 先统计每个特征的缺失比例
missing_rates = df.isna().mean()
print("各特征缺失比例：")
print(missing_rates.sort_values(ascending=False).head(10))

# 根据需要，可以对缺失值较少的特征进行填充或直接删除缺失行
# 示例：对组织学特征进行均值填充（仅作为一种处理方法）
feature_columns = [col for col in df.columns if col != "ID"]
# df[feature_columns] = df[feature_columns].fillna(df[feature_columns].mean())

# 也可以根据实际情况只分析缺失率低于一定阈值的特征
threshold = 0.2
valid_features = [col for col in feature_columns if df[col].isna().mean() < threshold]

# 假设 pCR 列为二分类变量（1=达到 pCR, 0=未达到）
results = []

for feature in feature_columns:
    # 对于每个特征，先删去该特征和 pCR 的缺失值
    df_subset = df[['pCR', feature]].dropna()
    
    # 如果该特征在删去缺失值后没有数据，则跳过
    if df_subset.shape[0] == 0:
        print(f"特征 {feature} 缺失过多，跳过分析。")
        continue
    
    # 构建自变量 X 和因变量 y
    X = df_subset[[feature]]
    X = sm.add_constant(X)
    y = df_subset["pCR"]
    
    # 检查自变量是否为零方差
    if np.all(X[feature] == X[feature].iloc[0]):
        print(f"特征 {feature} 方差为 0，跳过分析。")
        continue

    try:
        model = sm.Logit(y, X)
        result = model.fit(disp=0)
        coef = result.params[feature]
        p_value = result.pvalues[feature]
        odds_ratio = np.exp(coef)
        results.append({
            "Feature": feature,
            "Coefficient": coef,
            "Odds Ratio": odds_ratio,
            "p-value": p_value
        })
    except Exception as e:
        results.append({
            "Feature": feature,
            "Coefficient": np.nan,
            "Odds Ratio": np.nan,
            "p-value": np.nan,
            "Error": str(e)
        })
        print(f"特征 {feature} 分析出错: {e}")

results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by="p-value")
print(results_df)
# 将结果保存到 Excel 文件
output_filename = "./data/IMPRESS/p_random_group_eight_HER2_v3.xlsx"
results_df.to_excel(output_filename, index=False)

print(f"分析结果已保存到 {output_filename}")


# 筛选显著特征（例如 p < 0.05）
significant_features = results_df[results_df["p-value"] < 0.05]
print("显著预测 pCR 的组织学特征：")
print(significant_features)

with pd.ExcelWriter(output_filename) as writer:
    results_df.to_excel(writer, sheet_name="All Features", index=False)
    significant_features.to_excel(writer, sheet_name="Significant Features", index=False)

print(f"分析结果已保存到 {output_filename}")

print(f" {output_filename}")
