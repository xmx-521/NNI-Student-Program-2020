# Task 3.1 进阶任务项目说明文档

[toc]

## 团队基本信息

- 团队名：电脑一带五

- 团队成员：徐满心、李乐天、孟繁青、尚丙奇、陈垲昕

- 团队学校：同济大学

- Github Repo地址: https://github.com/xmx-521/NNI-Student-Program-2020

## 任务要求： NNI2020 Task 3.1 进阶任务

跑通NNI [Feature Engineering Sample](https://github.com/SpongebBob/tabular_automl_NNI)

  ### 文档情况

- [x] 将NNI应用于特征工程

## 1.特征工程简介

有一句话在工业界广为流传：数据和特征决定了机器学习的上界，而算法和模型只是逼近这个上限而已。特征工程便是一种最大限度地从原始数据中提取特征以供模型和算法使用的工程活动。其主要包含以下方面：
![](https://pic.downk.cc/item/5fe049d93ffa7d37b35181f4.png)

特征是一种要在其上进行分析或预测的所有独立单元的共享的属性，任何对模型有用的属性都可以是特征。相对于属性而言，特征的意义在于在问题的语境下更容易理解，特征是可能会对解决问题有所帮助的特性。

特征工程的总体过程是：
- 测试特征
- 决定要创建的特征
- 创建特征
- 检查这些特征如何运用在模型上
- 改进特征
- 创建新的特征

其中特征处理是最为核心的部分，sklearn提供了比较完整的特征处理方法。

## 2.自动特征工程简介

自动特征工程在上世纪90年代就成了研究的重要主题，关于自动机器学习的研究大致分为两个方向：其一是使用类似决策树的监督学习算法的多关系决策树学习(MRDTL)，其二是近期使用更简单的方法的进展，例如深特征合成等。

多关系决策树学习通过依次查询数据库并不断加入新子句的方式来生成特征，该算法可能以这样的形式开始：
```sql
SELECT COUNT(*) FROM ATOM t1 LEFT JOIN MOLECULE t2 ON t1.mol_id = t2.mol_id GROUP BY t1.mol_id
```
然后这样的查询可以通过连续加入例如` WHERE t1.charge <= -0.392`的方式逐渐被优化。

然而，大多数关于多关系决策树学习的研究都使用了现存的数据库，而这样会带来许多多余的操作。可以通过一些技巧来减少这些多余的操作。特别是最近，研究者发现通过增量保存的方式可以进一步提高效率，减少冗余的操作。

2015年，MIT的研究者展示了深特征合成算法并且在线上数据科学比赛中展示了其强大的效率。

基于NNI的自动特征工程方法分为两步，一是特征生成探索，二是特征选择。调参器调用AutoFETuner 生成命令获取初始特征的重要性值，然后AutoFETuner会根据定义的搜索空间找到预计的特征重要性值的排名。

## 3.自动特征工程的意义
IBM的研究者指出，自动特征工程能够帮助数据科学家减少在探索数据上所花费的时间，并让他们能够在短时间内尝试更多的新想法。从另一个角度来看，自动特征工程让对数据科学不熟悉的非专业人士在更短的时间内以更少的精力来发掘数据的价值。

## 4.简单的自动特征工程示例
### 4.1使用nni的自动特征工程工具
使用[Task3.1](https://github.com/SpongebBob/tabular_automl_NNI)给出的train.tiny.csv数据集，在代码中加入nni的部分
```diff
import nni

if __name__ == '__main__':
    file_name = 'train.tiny.csv'
    target_name = 'Label'
    id_index = 'Id'

    # read original data from csv file
    df = pd.read_csv(file_name)

    # get parameters from tuner
+   RECEIVED_FEATURE_CANDIDATES = nni.get_next_parameter()

+    if 'sample_feature' in RECEIVED_FEATURE_CANDIDATES.keys():
+        sample_col = RECEIVED_FEATURE_CANDIDATES['sample_feature']
+    # return 'feature_importance' to tuner in first iteration
+    else:
+        sample_col = []
+    df = name2feature(df, sample_col)

    feature_imp, val_score = lgb_model_train(df,  _epoch = 1000, target_name = target_name, id_index = id_index)

+    # send final result to Tuner
+    nni.report_final_result({
+        "default":val_score , 
+        "feature_importance":feature_imp
    })
```
然后定义搜索空间，储存在`search_space.json`文件中，定义好配置文件`config.yml`便可开始运行实验。

### 4.2实验结果

## 5.在Breast Cancer测试NNI的自动特征工程工具
### 5.1 Breast Cancer数据集简介

该数据集有286个实例，每个实例有9个属性，由南斯拉夫Institute of Oncology University Medical Canter Ljubljana 的Matjaz Zwitter和Milan Soklic所制作，属性分别为乳腺癌复发和未复发、年龄、绝经期、肿瘤大小、淋巴结个数、有无结节冒、肿瘤的恶行程度、左乳房或右乳房、所在象限以及是否经过放射性治疗。该数据集是加州大学欧文分校提出的用于机器学习的数据集，是一个常用的标准测试数据集。

![Cancer数据集](https://pic.downk.cc/item/5fe09b3d3ffa7d37b39a5c88.jpg)

### 5.2 加载数据集

```python
# Copyright (c) Microsoft Corporation
# All rights reserved.
#
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and
# to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED *AS IS*, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import nni
import logging
import numpy as np
import pandas as pd
import json
import sys
from sklearn.preprocessing import LabelEncoder

sys.path.append('../../')

from fe_util import *
from model import *

logger = logging.getLogger('auto-fe-examples')

if __name__ == '__main__':
    file_name = ' ./breast-cancer.data'
    target_name = 'Class'
    id_index = 'Id'

    # get parameters from tuner
    RECEIVED_PARAMS = nni.get_next_parameter()
    logger.info("Received params:\n", RECEIVED_PARAMS)
    
    # list is a column_name generate from tuner
    df = pd.read_csv(file_name, sep = ',')
    df.columns = [
        'Class', 'age', 'menopause', 'tumor-size', 'inv-nodes',
        'node-caps', 'deg-malig', 'breast', 'breast-quad', 'irradiat'
    ]
    df['Class'] = LabelEncoder().fit_transform(df['Class'])
    
    if 'sample_feature' in RECEIVED_PARAMS.keys():
        sample_col = RECEIVED_PARAMS['sample_feature']
    else:
        sample_col = []
    
    # raw feaure + sample_feature
    df = name2feature(df, sample_col, target_name)
    feature_imp, val_score = lgb_model_train(df,  _epoch = 1000, target_name = target_name, id_index = id_index)
    nni.report_final_result({
        "default":val_score, 
        "feature_importance":feature_imp
    })

```

### 5.3 单次运行结果展示

```
Final result: {"default": 0.5, "feature_importance": {"__pandas_dataframe__": {"column_order": ["feature_name", "split", "gain", "gain_percent", "split_percent", "feature_score"], "types": ["object", "int32", "float64", "float64", "float64", "float64"]}, "index": [0, 1, 2, 3, 4, 5, 6, 7, 8], "feature_name": ["age", "menopause", "tumor-size", "inv-nodes", "node-caps", "deg-malig", "breast", "breast-quad", "irradiat"], "split": [0, 0, 0, 0, 0, 0, 0, 0, 0], "gain": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], "gain_percent": [NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN], "split_percent": [NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN], "feature_score": [NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN]}}
```

### 5.4 数据预处理

通过加载数据集，我们得到的特征可能存在如下的问题

- 量纲不同，无法比较

- 定性特征无法输入某些机器学习算法，可将其转化为定量特征

- 数据集中存在缺失值，应进行补充

我们可以使用sklarn中的preprocessing库来进行与预处理工作，已解决上述问题。
#### 5.4.1 标准化处理
```python
from sklearn.preprocessing import StandardScaler
StandardScalar().fit_transform(breast-cancer.data)#返回标准化后的数据
```

#### 5.4.2 区间放缩法 

```python
from sklearn.preprocessing import MinMaxScaler
MinMaxScaler().fit_transform(breast-cancer.data)#返回区间放缩到[0,1]的数据
```

#### 5.4.3 归一化处理

```python
from sklearn.preprocessing import Normalizer
Normalizer().fit_transform(breast-cancer.data)#返回归一化后的数据
```

#### 5.4.4 将定性特征转换为定量特征

```python
#将label转换成0~1之间的数
from sklearn.preprocessing import LabelEncoder
LabelEncoder().fit_transform(df['Class'])
```

### 5.5 自动特征工程调参器

#### 5.5.1 AutoFETuner代码

```python
# Copyright (c) Microsoft Corporation
# All rights reserved.
#
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and
# to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED *AS IS*, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import copy
import json
import logging
import random
import numpy as np
from itertools import combinations

from enum import Enum, unique

from nni.tuner import Tuner
from nni.utils import extract_scalar_reward, OptimizeMode

from const import FeatureType, AGGREGATE_TYPE

logger = logging.getLogger('autofe-tuner')


class AutoFETuner(Tuner):
    def __init__(self, optimize_mode = 'maximize', feature_percent = 0.6):
        """Initlization function
        count : 
        optimize_mode : contains "Maximize" or "Minimize" mode.
        search_space : define which features that tuner need to search
        feature_percent : @mengjiao
        default_space : @mengjiao 
        epoch_importance : @mengjiao
        estimate_sample_prob : @mengjiao
        """
        self.count = -1
        self.optimize_mode = OptimizeMode(optimize_mode)
        self.search_space = None
        self.feature_percent = feature_percent
        self.default_space = []
        self.epoch_importance = []
        self.estimate_sample_prob = None

        logger.debug('init aufo-fe done.')


    def generate_parameters(self, parameter_id, **kwargs):
        """Returns a set of trial graph config, as a serializable object.
        parameter_id : int
        """
        self.count += 1
        if self.count == 0:
            return {'sample_feature': []}
        else:
            sample_p = np.array(self.estimate_sample_prob) / np.sum(self.estimate_sample_prob)
            sample_size = min(128, int(len(self.candidate_feature) * self.feature_percent))
            sample_feature = np.random.choice(
                self.candidate_feature, 
                size = sample_size, 
                p = sample_p, 
                replace = False
                )
            gen_feature = list(sample_feature)
            r = {'sample_feature': gen_feature}
            return r  


    def receive_trial_result(self, parameter_id, parameters, value, **kwargs):
        '''
        Record an observation of the objective function
        parameter_id : int
        parameters : dict of parameters
        value: final metrics of the trial
        '''
        # get the default feature importance

        if self.search_space is None:
            self.search_space = value['feature_importance']
            self.estimate_sample_prob = self.estimate_candidate_probility()
        else:
            self.epoch_importance.append(value['feature_importance'])
            # TODO
            self.update_candidate_probility()
        reward = extract_scalar_reward(value)
        if self.optimize_mode is OptimizeMode.Minimize:
            reward = -reward

        logger.info('receive trial result is:\n')
        logger.info(str(parameters))
        logger.info(str(reward))
        return


    def update_search_space(self, data):
        '''
        Input: data, search space object.
        {
            'op1' : [col1, col2, ....]
            'op2' : [col1, col2, ....]
            'op1_op2' : [col1, col2, ....]
        }
        '''

        self.default_space = data
        self.candidate_feature = self.json2space(data)


    def update_candidate_probility(self):
        """
        Using true_imp score to modify candidate probility.
        """
        # get last importance
        last_epoch_importance = self.epoch_importance[-1]
        last_sample_feature = list(last_epoch_importance.feature_name)
        for index, f in enumerate(self.candidate_feature):
            if f in last_sample_feature:
                score = max(float(last_epoch_importance[last_epoch_importance.feature_name == f]['feature_score']), 0.00001)
                self.estimate_sample_prob[index] = score
        
        logger.debug("Debug UPDATE ", self.estimate_sample_prob)


    def estimate_candidate_probility(self):
        """
        estimate_candidate_probility use history feature importance, first run importance.
        """
        raw_score_dict = self.impdf2dict()
        logger.debug("DEBUG feature importance\n", raw_score_dict)

        gen_prob = []
        for i in self.candidate_feature:
            _feature = i.split('_')
            score = [raw_score_dict[i] for i in _feature if i in raw_score_dict.keys()]
            if len(score) == 1:
                gen_prob.append(np.mean(score))
            else:
                generate_score = np.mean(score) * 0.9 # TODO
                gen_prob.append(generate_score)
        return gen_prob


    def impdf2dict(self):
        return dict([(i,j) for i,j in zip(self.search_space.feature_name, self.search_space.feature_score)])


    def json2space(self, default_space):
        """
        parse json to search_space 
        """
        result = []
        for key in default_space.keys():
            if key == FeatureType.COUNT:
                for i in default_space[key]:
                    name = (FeatureType.COUNT + '_{}').format(i)
                    result.append(name)         
            
            elif key == FeatureType.CROSSCOUNT:
                for i in default_space[key][0]:
                    for j in default_space[key][1]:
                        if i == j:
                            continue
                        cross = [i,j] 
                        cross.sort()
                        name = (FeatureType.CROSSCOUNT + '_') + '_'.join(cross)
                        result.append(name)         
                        
            
            elif key == FeatureType.AGGREGATE:
                for i in default_space[key][0]:
                    for j in default_space[key][1]:
                        for stat in AGGREGATE_TYPE:
                            name = (FeatureType.AGGREGATE + '_{}_{}_{}').format(stat, i, j)
                            result.append(name)
            
            elif key == FeatureType.NUNIQUE:
                for i in default_space[key][0]:
                    for j in default_space[key][1]:
                        name = (FeatureType.NUNIQUE + '_{}_{}').format(i, j)
                        result.append(name)
            
            elif key == FeatureType.HISTSTAT:
                for i in default_space[key][0]:
                    for j in default_space[key][1]:
                        name = (FeatureType.HISTSTAT + '_{}_{}').format(i, j)
                        result.append(name)
            
            elif key == FeatureType.TARGET:
                for i in default_space[key]:
                    name = (FeatureType.TARGET + '_{}').format(i)
                    result.append(name) 
            
            elif key == FeatureType.EMBEDDING:
                for i in default_space[key]:
                    name = (FeatureType.EMBEDDING + '_{}').format(i)
                    result.append(name) 
            
            else:
                raise RuntimeError('feature ' + str(key) + ' Not supported now')
        return result
```
### 5.6使用NNI进行自动调参的实验结果

