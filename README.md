## 一、数据增强

### 1. 🚀 数据增强方法  
为了增加训练样本数量和模型的鲁棒性，对原始数据进行数据增强。具体方法是在原始特征数据上添加高斯噪声，从而生成多个增强副本。增强过程中两个关键参数为：  
- **n_augments**：控制生成增强副本的个数（例如，本代码中设为 1，即生成一个增强副本）。  
- **noise_std**：高斯噪声的标准差（例如，本代码中设为 0.001，噪声较小，确保数据分布基本不变）。  

主要实现逻辑如下：

```python
# 设置增强参数
n_augments = 1       # 生成几个增强副本
noise_std = 0.001    # 高斯噪声标准差

# 数据增强函数
def augment_X_y(X, y, ids, n_augments=2, noise_std=0.01):
    X_list = [X]
    y_list = [y]
    id_list = [ids]

    for i in range(n_augments):
        noise = np.random.normal(loc=0, scale=noise_std, size=X.shape)
        X_aug = X + noise
        X_list.append(pd.DataFrame(X_aug, columns=X.columns))
        y_list.append(y.copy())
        id_list.append(ids.copy())

    X_augmented = pd.concat(X_list, ignore_index=True)
    y_augmented = pd.concat(y_list, ignore_index=True)
    ids_augmented = pd.concat(id_list, ignore_index=True)

    return X_augmented, y_augmented, ids_augmented
```

通过上述函数，原始数据集（包括特征、标签和ID）生成了多个增强后的版本，并通过 `pd.concat` 将增强数据合并，最终用于后续的模型训练和交叉验证。

🔗 [查看完整代码](https://github.com/tcyfree/NAC/blob/main/auc_roi_random_kfold_aug.py)

## 二、特征筛选

### 1. 🎯 显著性筛选  
从 859 个 ROI 级别的特征中，筛选出与 pCR 显著相关（p < 0.05）的 496 个特征。  
🔗 [查看代码](https://github.com/tcyfree/NAC/blob/main/significant_features.py)

### 2. 📈 相关性筛选  
再根据与 pCR 的 Pearson 相关性（Pearson > 0.2）进一步筛选出 135 个特征。  
🔗 [查看代码](https://github.com/tcyfree/NAC/blob/main/auc_roi_random_kfold_v2.py)

---

## 三、因子分析

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

## 四、外部测试

使用独立测试集进行验证，评估模型泛化能力。  
🔗 [查看代码](https://github.com/tcyfree/NAC/blob/main/auc_roi_kfold_ex_test.py)

---
