# Task 3.2.2 复杂型数据的探究任务项目说明文档

## 团队基本信息

- 团队名：电脑一带五

- 团队成员：徐满心、李乐天、孟繁青、尚丙奇、陈垲昕

- 团队学校：同济大学

- Github Repo地址: https://github.com/xmx-521/NNI-Student-Program-2020

## 任务要求

在完成本小节任务的时候，您可以参考以下的步骤：

1. 针对原始数据做简单的特征提取。
2. 定义机器学习 / 深度学习模型，输入提取后的特征进行模型训练，记录实验结果。
3. 引入 NNI 工具，尝试定义搜索空间，进行特征生成和特征筛选（需要特别关注**特征搜索空间的设计**）。
4. 再次利用这些特征再次训练模型，分析比较两次实验的结果，并将整个过程写入报告中。

### 完成情况
1.推荐系统（Recommender System）

- [x] 产品推荐：Santander Product Recommendation

## 产品推荐：Santander Product Recommendation

#### 任务目标

1.  基于训练数据建立模型以准确预测客户要购买的产品
2. 基于NNI对数据进行特征选择，并与原模型比较经过NNI进行特征选择后的准确率

#### 模型描述

一个人可能购买很多种产品，因此产品推荐问题类似于multi-label问题，与Task3.2.1中土壤属性预测问题不同的是，产品推荐中的一个label（即产品），会极大程度上受其它labels（也都是产品）的影响。为了搭配使用NNI，将产品推荐问题拆解为多个逻辑回归问题分别用建模，并使用roc_auc_score作为metric

#### 数据清洗与预处理

对于产品推荐系统来说，最重要的特征就是顾客买过的产品，因此我们只取顾客买过的产品为可选features

```python
usecols = [
    'ncodpers', 'ind_ahor_fin_ult1', 'ind_aval_fin_ult1', 'ind_cco_fin_ult1',
    'ind_cder_fin_ult1', 'ind_cno_fin_ult1', 'ind_ctju_fin_ult1',
    'ind_ctma_fin_ult1', 'ind_ctop_fin_ult1', 'ind_ctpp_fin_ult1',
    'ind_deco_fin_ult1', 'ind_deme_fin_ult1', 'ind_dela_fin_ult1',
    'ind_ecue_fin_ult1', 'ind_fond_fin_ult1', 'ind_hip_fin_ult1',
    'ind_plan_fin_ult1', 'ind_pres_fin_ult1', 'ind_reca_fin_ult1',
    'ind_tjcr_fin_ult1', 'ind_valo_fin_ult1', 'ind_viv_fin_ult1',
    'ind_nomina_ult1', 'ind_nom_pens_ult1', 'ind_recibo_ult1'
]
```

使用pandas读入数据，使用最保守的做法，将空值填0并且保证customer id唯一

```python
train = pd.read_csv('dataset/train_ver2.csv', usecols=usecols)
sample = pd.read_csv('dataset/sample_submission.csv')
train = train.drop_duplicates(['ncodpers'], keep='last')
model_pred = {}
ids = train.ncodpers.values
train.fillna(0, inplace=True)
pred = defaultdict(list)
```

#### 训练

每个产品都可看作label，因为有24种产品，故要建立24个逻辑回归模型

```python
for col in train.columns:
    if col != 'ncodpers':
        y_train = train[col]
        x_train = train.drop(['ncodpers', col], axis=1)

        clf = LogisticRegression(max_iter=5000)
        clf.fit(x_train, y_train)
        y_pred = clf.predict_proba(x_train)[:, 1]
        model_pred[col] = y_pred

        for id, y_hat in zip(ids, y_pred):
            pred[id].append(y_hat)

        print('ROC Socre : %f' % (roc_auc_score(y_train, y_pred)))
```

因为有24个逻辑回归模型，所以要针对每一个模型都进行一次Feature Engineering

~~~diff
```python
for col in train.columns:
    if col != 'ncodpers':
        y_train = train[col]
        x_train = train.drop(['ncodpers', col], axis=1)

+        y_train_as_matrix = y_train.values
+        y_train_as_matrix = np.matrix(y_train_as_matrix.reshape(
+           (-1, 1))).astype(float)
+        fgs = FeatureGradientSelector(classification=False,
+                                      n_epochs=20,
+                                      verbose=1,
+                                     batch_size=10000000,
+                                     n_features=15)
+        fgs.fit(x_train, y_train_as_matrix)
+        print(fgs.get_selected_features())

+        selected_feature_indices = fgs.get_selected_features()
+        x_train = x_train.iloc[:, selected_feature_indices]

        clf = LogisticRegression(max_iter=5000)
        clf.fit(x_train, y_train)
        y_pred = clf.predict_proba(x_train)[:, 1]
        model_pred[col] = y_pred

        for id, y_hat in zip(ids, y_pred):
            pred[id].append(y_hat)

        print('ROC Socre : %f' % (roc_auc_score(y_train, y_pred)))
```
~~~

#### 实验结果

由于实验的label总数高达24个，实在过多，因此以kaggle的最终得分作为评价标准（分越高越好）

[![6Beqeg.png](https://s3.ax1x.com/2021/03/14/6Beqeg.png)](https://imgtu.com/i/6Beqeg)

从kaggle的最终得分可以看出，使用NNI方法后模型准确率得到了一定的提升

#### 分析与讨论

通过本次任务可以发现，NNI在产品推荐问题上有着不错的表现，在Task3.2.1熟悉使用了之后，使用过程十分顺畅。美中不足之处或许仍是Task3.2.1中也提及过的FeatureGradientSelector的构造函数参数功能不够的问题，在本问题中我们小组很想指定一个特征数量的区间范围，但现有API只允许指定固定特征数

