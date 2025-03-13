import pandas as pd
import numpy as np

def split_and_average_excel(input_file, output_file):
    # 读取 Excel 文件
    df = pd.read_excel(input_file)
    
    # 确保数据中包含 'ID' 列
    if 'ID' not in df.columns:
        raise ValueError("输入文件必须包含 'ID' 列")
    
    # 分组计算
    result_list = []
    for _, group in df.groupby('ID'):
        group = group.sample(frac=1).reset_index(drop=True)  # 随机打乱数据
        splits = np.array_split(group, 8)  # 分成4份
        averages = pd.DataFrame([split.mean(numeric_only=True) for split in splits])  # 计算每份的均值
        averages['ID'] = group['ID'].iloc[0]  # 重新添加 ID 列
        result_list.append(averages)
    
    # 合并结果并保存
    result_df = pd.concat(result_list, ignore_index=True)
    result_df.to_excel(output_file, index=False)
    print(f'处理完成，结果保存在 {output_file}')

# 示例调用
input_excel = "merged_data.xlsx"  # 输入文件
output_excel = "output_random_group_eight_4.xlsx"  # 输出文件
split_and_average_excel(input_excel, output_excel)
