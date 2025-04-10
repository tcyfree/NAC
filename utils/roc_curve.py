import pandas as pd
import matplotlib.pyplot as plt

# 定义Excel文件路径（请根据实际情况修改）
excel_file = "./data/IMPRESS/roc_curve_data_best_fold_one_aug_TNBC_v1.xlsx"

# 读取ROC数据和性能指标
roc_data = pd.read_excel(excel_file, sheet_name="ROC Data")
metrics_data = pd.read_excel(excel_file, sheet_name="Metrics")

# 从性能指标数据中提取AUC和Accuracy值
auc = metrics_data.loc[metrics_data['Metric'] == 'AUC', 'Value'].values[0]
acc = metrics_data.loc[metrics_data['Metric'] == 'Accuracy', 'Value'].values[0]
ci_lower = metrics_data.loc[metrics_data['Metric'] == 'AUC_CI_Lower', 'Value'].values[0]
ci_upper = metrics_data.loc[metrics_data['Metric'] == 'AUC_CI_Upper', 'Value'].values[0]

# 绘制ROC曲线
plt.figure(figsize=(6, 6))
plt.plot(roc_data['False Positive Rate'], roc_data['True Positive Rate'], label=f"AUC = {auc:.4f} (95% CI: {ci_lower:.4f} - {ci_upper:.4f}) \nAccuracy = {acc:.4f}", color="blue")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(f"ROC Curve - Best Fold")
plt.legend()
plt.show()
