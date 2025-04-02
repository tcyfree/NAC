import pandas as pd

# 设定因子载荷阈值（一般 > 0.6 认为有显著贡献）
threshold = 0.15

# 读取因子载荷矩阵（假设已计算得到）
# factor_loadings = pd.read_excel("./data/IMPRESS/cluster/result_with_clusters_v2.xlsx", index_col=0)
factor_loadings = pd.read_excel("./data/IMPRESS/cluster/result_with_clusters_pca.xlsx", index_col=0)

# 记录每个因子的主要变量
factor_dict = {}

for factor in factor_loadings.columns:
    high_loadings = factor_loadings[factor].abs() > threshold
    selected_vars = factor_loadings.index[high_loadings].tolist()
    factor_dict[factor] = selected_vars

    # 打印结果
    print(f"{factor} 主要变量: {selected_vars}")

# 转换为 DataFrame，方便写入 Excel
factor_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in factor_dict.items()]))

# 保存到 Excel
output_path = "./data/IMPRESS/cluster/factor_main_variables_pca.xlsx"
factor_df.to_excel(output_path, index=False)

print(f"因子主要变量已保存至: {output_path}")
