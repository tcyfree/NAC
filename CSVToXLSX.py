# import pandas as pd

# # 读取 CSV 文件
# csv_file = "./data/IMPRESS/HER2/perDatasetSlideSummaries/RoiFeatureSummary_Means.csv"  # 你的 CSV 文件路径
# df = pd.read_csv(csv_file)

# # 保存为 Excel 文件
# xlsx_file = "./data/IMPRESS/HER2/perDatasetSlideSummaries/RoiFeatureSummary_Means.xlsx"  # 你的目标 XLSX 文件路径
# df.to_excel(xlsx_file, index=False)

# print(f"转换完成: {xlsx_file}")


import os
import pandas as pd

# 文件夹路径
folder_path = "./data/IMPRESS/TNBC/perSlideROISummaries"

# 遍历文件夹中的所有 CSV 文件
for filename in os.listdir(folder_path):
    if filename.endswith(".csv"):
        # 构建 CSV 文件和目标 Excel 文件的路径
        csv_file = os.path.join(folder_path, filename)
        xlsx_file = os.path.join(folder_path, filename.replace(".csv", ".xlsx"))
        
        # 读取 CSV 文件
        df = pd.read_csv(csv_file)
        
        # 保存为 Excel 文件
        df.to_excel(xlsx_file, index=False)
        
        print(f"转换完成: {xlsx_file}")

