import pandas as pd
import re

# 读取 Excel 文件
df = pd.read_excel("./data/IMPRESS/sorted_selected_features_TNBC_p_l1_imp.xlsx")

# 定义主题关键词，按优先级顺序存入列表
themes = [
    ("Epithelium", "Epithelial"),
    ("Stroma", "Stroma"),
    ("TILs", "TILs"),
    ("Necrosis", "JUNK"),
    ("Interactions", "RipleysK")
]

# 通过匹配 `Feature` 列的值确定 `theme`
def assign_theme(feature):
    feature_str = str(feature)  # 确保是字符串
    match_positions = {}  # 存储匹配关键词的位置

    # 遍历主题列表，记录每个关键词在 Feature 字符串中的最早位置
    for theme, keyword in themes:
        match = re.search(keyword, feature_str)  # 查找关键词位置
        if match:
            match_positions[theme] = match.start()  # 记录关键词起始位置

    # 如果有匹配项，返回最先出现的主题
    if match_positions:
        return min(match_positions, key=match_positions.get)

    return "Other"  # 如果都不匹配，则归为 Other

df["Theme"] = df["Feature"].apply(assign_theme)

# 保存结果
df.to_excel("./data/IMPRESS/TNBC_p_l1_imp_theme.xlsx", index=False)

print(df.head())  # 查看前几行结果
