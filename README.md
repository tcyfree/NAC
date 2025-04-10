# 项目概述

本项目主要包括以下几个部分：

1. **特征提取**：基于两篇相关论文的方法，从WSIs中分割和识别组织区域和细胞核信息，并定量提取基于ROI的Histomic feature。
2. **数据增强**：针对原始样本量较小的问题，采用高斯噪声数据增强方法扩充数据集。
3. **特征筛选**：通过显著性和相关性分析从859个ROI级别特征中筛选出与pCR显著相关的特征。
4. **因子分析**：利用因子分析提取具有医学解释性的因子，并评估其对pCR预测的能力。
5. **外部测试**：在独立测试集上验证模型的泛化能力。

---

# 一、 特征提取

### 1.1 基于论文方法

#### 参考论文1  
> Liu, Shangke, et al. "A panoptic segmentation approach for tumor-infiltrating lymphocyte assessment: development of the MuTILs model and PanopTILs dataset." *MedRxiv* (2022): 2022-01.

- **PanopTILs dataset**: https://sites.google.com/view/panoptils/home

- **代码和模型权重**: https://github.com/PathologyDataScience/MuTILs_Panoptic

使用该论文官方发布的预训练全景分割模型“MuTILs”，对乳腺癌WSIs进行组织区域和细胞核分割。模型结合语义分割（区域）和对象检测（细胞核）两部分，输出WSIs的Region和Nucleus分割及识别结果。

#### 参考论文2  
> Amgad M, Rathore MA, et al. A population-level digital histologic biomarker for enhanced prognosis of invasive breast cancer. *Nat Med*. 2024;30(1):85-97.

- **HiPS方法代码**: https://github.com/PathologyDataScience/HiPS

利用MuTILs模型的输出作为HiPS方法的输入，每个WSI提取出859个基于ROI的Histomic定量特征。

---

# 二、 数据增强

由于原始数据仅有64个样本，为提高模型鲁棒性及扩充训练数据，采用高斯噪声数据增强方法。

### 2.1 方法说明

- **高斯噪声添加**：  
  在原始特征数据上添加噪声，生成多个增强副本。
  
- **关键参数**：
  - **n_augments**：控制生成增强副本的数量（例如，本代码中设为1，即生成一个增强副本）。
  - **noise_std**：高斯噪声标准差（例如，本代码中设为0.001，噪声较小，确保数据分布基本不变）。

通过对原始数据集（包含特征、标签及ID）生成多个增强版本，后续用于模型训练和交叉验证。

代码: https://github.com/tcyfree/NAC/blob/main/auc_roi_random_kfold_aug.py

---

# 三、 特征筛选

从859个ROI级特征中，通过两步筛选获得与pCR显著相关的特征。

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

- 根据因子载荷阈值，识别出具有医学可解释性的因子。

代码: https://github.com/tcyfree/NAC/blob/main/utils/cluster_main_fac.py

### 4.3 预测能力评估

- 使用提取因子的样本得分，评估其对因变量（pCR）的预测能力。

代码: https://github.com/tcyfree/NAC/blob/main/auc_roi_random_kfold_fa.py

---

# 五、 外部测试

在独立测试集上验证模型的泛化能力。

代码: https://github.com/tcyfree/NAC/blob/main/auc_roi_kfold_ex_test.py

---