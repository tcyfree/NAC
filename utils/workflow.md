# 项目概述

本项目旨在通过AI提取WSI中的TME特征，预测TNBC患者对NAC的pCR反应，整体流程包括：

1. **特征提取**：基于两篇已有研究，从WSIs中分割组织区域和细胞核，并定量提取基于ROI的TME组织学特征。
2. **数据增强**：针对原始样本量较小的问题，采用高斯噪声数据增强方法扩充数据集。
3. **特征筛选**：通过显著性检验与相关性分析，从859个特征中筛选出与pCR密切相关的特征。
4. **因子分析**：利用因子分析提取具有医学解释性的因子，并评估其对pCR预测的能力。
5. **外部测试**：在独立测试集上验证模型的泛化能力。

---

# 一、 特征提取

### 1.1 基于公开模型

#### 参考论文1  
> Liu, Shangke, et al. "A panoptic segmentation approach for tumor-infiltrating lymphocyte assessment: development of the MuTILs model and PanopTILs dataset." *MedRxiv* (2022): 2022-01.

- **PanopTILs dataset**: https://sites.google.com/view/panoptils/home

- **代码和模型权重**: https://github.com/PathologyDataScience/MuTILs_Panoptic

使用该论文官方发布的预训练全景分割模型“MuTILs”对WSIs进行组织区域（Region）与细胞核（Nucleus）的联合分割，输出全景分割结果，用作后续特征提取的基础。

#### 参考论文2  
> Amgad M, Rathore MA, et al. A population-level digital histologic biomarker for enhanced prognosis of invasive breast cancer. *Nat Med*. 2024;30(1):85-97.

- **HiPS方法代码**: https://github.com/PathologyDataScience/HiPS

将MuTILs分割结果作为HiPS输入，定量提取859个基于ROI的TME组织学特征。

---

# 二、 数据增强

原始数据集来自公开研究（Huang, Zhi, et al. "Artificial intelligence reveals features associated with breast cancer neoadjuvant chemotherapy responses from multi-stain histopathologic images." NPJ Precision Oncology 7.1 (2023): 14.），仅有64个TNBC样本。为缓解样本量不足和提高模型鲁棒性，采用高斯噪声进行特征级别的数据增强。

### 2.1 方法说明

- **增强策略**：  
  对原始特征添加小幅高斯噪声，生成多个增强副本
  
- **关键参数**：
  - **n_augments**：控制生成增强副本的数量（例如，本代码中设为1，即生成一个增强副本）。
  - **noise_std**：高斯噪声标准差（例如，本代码中设为0.001，噪声较小，确保数据分布基本不变）。

- **增强效果**：
  统计显著性成立：p < 0.05，说明在 95% 的置信水平下，增强数据带来了显著的性能提升，可以认为数据增强确实改善了模型的泛化能力。

代码: https://github.com/tcyfree/NAC/blob/main/auc_roi_random_kfold_aug.py

---

# 三、 特征筛选

从859个TME特征中，通过两步筛选获得与pCR显著相关的特征。

### 3.1 显著性筛选

- 筛选条件：选择与pCR显著相关（p < 0.05）的特征。
- 筛选结果：获得496个特征。

代码: https://github.com/tcyfree/NAC/blob/main/significant_features.py

### 3.2 相关性筛选

- 筛选条件：基于与pCR的Pearson相关系数，选择相关性大于0.2的特征。
- 筛选结果：最终筛选出135个特征。

代码: https://github.com/tcyfree/NAC/blob/main/auc_roi_random_kfold_v3.py

---

# 四、 因子分析

基于前述筛选出的特征，进行因子分析以提取具有医学解释性的因子，并评估其预测能力。

### 4.1 因子提取

- 工具：`FactorAnalyzer`
- 方法：采用最大似然法（ML）并指定斜交旋转（Promax）

代码: https://github.com/tcyfree/NAC/blob/main/utils/cluster_FA_ml.py

### 4.2 医学可解释因子识别

- 根据因子载荷阈（factor loading）值，识别出具有医学可解释性的因子。

代码: https://github.com/tcyfree/NAC/blob/main/utils/cluster_main_fac.py

### 4.3 预测能力评估

- 使用提取因子的样本得分，评估其对因变量（pCR）的预测能力（如AUC，Accuracy）。

代码: https://github.com/tcyfree/NAC/blob/main/auc_roi_random_kfold_fa.py

---

# 五、 外部测试

在多个独立测试集上验证模型的泛化能力。

### 5.1 数据来源

- **测试集**：来自[TCIA Post-NAC BRCA 数据集](https://www.cancerimagingarchive.net/collection/post-nat-brca)，其中样本均为非pCR患者。

### 5.2 测试策略

- 由于测试集标签单一，结果以**Accuracy**为主要评估指标。

代码: https://github.com/tcyfree/NAC/blob/main/auc_roi_kfold_ex_test.py

---
