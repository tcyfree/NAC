## 一、特征筛选

### 1. 🎯 显著性筛选  
从 859 个 ROI 级别的特征中，筛选出与 pCR 显著相关（p < 0.05）的 496 个特征。  
🔗 [查看代码](https://github.com/tcyfree/NAC/blob/main/significant_features.py)

### 2. 📈 相关性筛选  
再根据与 pCR 的 Pearson 相关性（Pearson > 0.2）进一步筛选出 135 个特征。  
🔗 [查看代码](https://github.com/tcyfree/NAC/blob/main/auc_roi_random_kfold_v2.py)

---

## 二、因子分析

### 1. 🧮 因子提取  
使用 `FactorAnalyzer` 进行因子分析，采用最大似然法（ML）并指定斜交旋转（Promax）。  
🔗 [查看代码](https://github.com/tcyfree/NAC/blob/main/utils/cluster_FA_ml.py)

### 2. 🧑‍⚕️ 医学可解释因子识别  
基于因子载荷阈值，识别出若干具有医学可解释性的因子。  
🔗 [查看代码](https://github.com/tcyfree/NAC/blob/main/utils/cluster_main_fac.py)

### 3. 🔍 预测能力评估  
使用这些因子的样本得分，评估其对因变量（pCR）的预测能力。  
🔗 [查看代码](https://github.com/tcyfree/NAC/blob/main/auc_roi_random_kfold_fa.py)

---

## 三、外部测试

使用独立测试集进行验证，评估模型泛化能力。  
🔗 [查看代码](https://github.com/tcyfree/NAC/blob/main/auc_roi_kfold_ex_test.py)

---
