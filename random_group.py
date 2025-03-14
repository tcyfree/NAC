import pandas as pd
import numpy as np

def split_and_average_excel(input_file, output_file):
    # 读取 Excel 文件
    df = pd.read_excel(input_file)
    
    # 确保数据中包含 'ID' 列
    if 'ID' not in df.columns:
        raise ValueError("输入文件必须包含 'ID' 列")
    
    # 替换 NaN 或 inf 值为 0
    df.replace([np.inf, -np.inf], 0, inplace=True)  # 替换 inf 和 -inf 为 0
    df.fillna(0, inplace=True)  # 替换 NaN 为 0

    # 分组计算
    result_list = []
    split_num = 8
    for _, group in df.groupby('ID'):
        # 如果不足 8 行，直接保留原数据
        if len(group) < split_num:
            result_list.append(group)  
        else:
            group = group.sample(frac=1).reset_index(drop=True)  # 随机打乱数据
            splits = np.array_split(group, split_num)  # 分成8份
            averages = pd.DataFrame([split.mean(numeric_only=True) for split in splits])  # 计算每份的均值
            averages['ID'] = group['ID'].iloc[0]  # 重新添加 ID 列
            result_list.append(averages)
    
    # 合并结果并保存
    result_df = pd.concat(result_list, ignore_index=True)
    result_df.to_excel(output_file, index=False)
    print(f'处理完成，结果保存在 {output_file}')

# 示例调用
input_excel = "./data/Post-NAT-BRCA/merged_data_Post_TNBC.xlsx"  # 输入文件
output_excel = "./data/Post-NAT-BRCA/random_group_eight_Post_TNBC.xlsx"  # 输出文件
split_and_average_excel(input_excel, output_excel)
