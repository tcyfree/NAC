import pandas as pd

# 读取 HER2+ Excel 文件
her2_file = "data/Nature-NAT/HER2+.xlsx"
df_her2 = pd.read_excel(her2_file, sheet_name="Sheet1")

# 读取 Slide metadata Excel 文件
slide_metadata_file = "data/Nature-NAT/Slide metadata.xlsx"
df_metadata = pd.read_excel(slide_metadata_file, sheet_name="Sheet2")

# 确保 Trial.ID 和 Slide.ID 存在于元数据文件中
df_metadata = df_metadata[["Trial.ID", "Slide.ID"]]

# 合并数据，将 Slide.ID 添加到 HER2+ 数据中
df_her2 = df_her2.merge(df_metadata, on="Trial.ID", how="left")

# 保存更新后的 Excel 文件
updated_file_path = "data/Nature-NAT/HER2+_updated.xlsx"
df_her2.to_excel(updated_file_path, index=False)

print(f"更新后的文件已保存为: {updated_file_path}")
