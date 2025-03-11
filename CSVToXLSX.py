import pandas as pd

# 读取 CSV 文件
# csv_file = "./data/IMPRESS/HER2/perDatasetSlideSummaries/RoiFeatureSummary_Means.csv"  # 你的 CSV 文件路径
csv_file = "./data/IMPRESS/TNBC/perDatasetSlideSummaries/RoiFeatureSummary_Stds.csv"  # 你的 CSV 文件路径
df = pd.read_csv(csv_file)

# 保存为 Excel 文件
# xlsx_file = "./data/IMPRESS/HER2/perDatasetSlideSummaries/RoiFeatureSummary_Means.xlsx"  # 你的目标 XLSX 文件路径
xlsx_file = "./data/IMPRESS/TNBC/perDatasetSlideSummaries/RoiFeatureSummary_Stds.xlsx"  # 你的目标 XLSX 文件路径
df.to_excel(xlsx_file, index=False)

print(f"转换完成: {xlsx_file}")
