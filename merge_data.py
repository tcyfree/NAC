import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve

# 读取 cohort_meta_TNBC.xlsx 文件
meta_df = pd.read_excel("./data/Post-NAT-BRCA/Post_Clinical_TNBC.xlsx", dtype={"ID": str})

# print('meta_df:', meta_df.columns.tolist())

# 设置 perSlideROISummaries 文件夹路径
roi_folder = './data/Post-NAT-BRCA/TNBC'

# 获取文件夹中所有的xlsx文件（文件名格式为 病人ID_HE.xlsx）
file_list = [f for f in os.listdir(roi_folder) if f.endswith('.xlsx')]

# 初始化一个空的 DataFrame 来合并所有病人的数据
merged_df = pd.DataFrame()

# 遍历文件列表，读取每个病人的数据，并与 meta_df 合并
for file in file_list:
    if ".xlsx" in file:  # 确保文件符合格式
        patient_id = file.replace(".xlsx", "")  # 提取病人ID
        roi_df = pd.read_excel(os.path.join(roi_folder, file))
        roi_df['ID'] = str(patient_id)  # 添加病人ID列，并将 ID 转换为字符串类型
        merged_df = pd.concat([merged_df, roi_df], ignore_index=True)

# 合并临床信息
df = pd.merge(meta_df, merged_df, on="ID", how="inner")

# 获取特征列
feature_columns = [col for col in df.columns if col not in ["ID", "pCR"]]

# 识别非数值列
non_numeric_columns = df[feature_columns].select_dtypes(exclude=["number"]).columns.tolist()
print(f"非数值列: {non_numeric_columns}")

# 删除非数值列
df_numeric = df.drop(columns=non_numeric_columns)

print(f"df_numeric 共有 {df_numeric.shape[0]} 行，{df_numeric.shape[1]} 列")
df_numeric.to_excel("./data/Post-NAT-BRCA/merged_data_Post_TNBC.xlsx", index=False)
print("合并后的数据已保存到 merged_data_her2.xlsx")

