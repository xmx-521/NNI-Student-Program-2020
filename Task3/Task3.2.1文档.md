# Task 3.2.1 表格型数据的进阶任务项目说明文档

## 团队基本信息

- 团队名：电脑一带五

- 团队成员：徐满心、李乐天、孟繁青、尚丙奇、陈垲昕

- 团队学校：同济大学

- Github Repo地址: https://github.com/xmx-521/NNI-Student-Program-2020

## 任务要求： NNI2020 Task 3.2.1  表格型数据的进阶任务

针对表格类型的数据，在使用 NNI 的时候，您需要思考如何设计特征搜索空间（Search Space），在设计的空间里搜索尝试哪一种组合更好；思考如何设计特征抽取模块，包括但不限于以下几个问题：

1. 数据预处理：数据清洗（包括缺失值的填充、异常值的处理等）和稀疏特征的处理等

2. 数据编码方式：针对类别、数值、多值、时间数据等做不同的处理

3. 高阶特征的挖掘：如何挖掘高阶特征

4. 基于其他分类器的特征提取：如基于KNN的特征、tree的特征
   
### 完成情况
   - [x] [电影票房预测：TMDB Box Office Prediction](https://www.kaggle.com/c/tmdb-box-office-prediction/data)
   - [x] [旧金山犯罪分类：San Francisco Crime Classification](https://www.kaggle.com/c/sf-crime/data)
   - [x] [土壤属性预测：Africa Soil Property Prediction Challenge](https://www.kaggle.com/c/afsis-soil-properties/data)

## 1. 电影票房预测

#### 任务目标：

1. 基于训练数据建立模型以准确预测全球电影的票房
2. 基于NNI对数据进行特征选择，并与原模型比较经过NNI进行特征选择后的准确率

#### 模型描述：

该问题为一个典型的监督学习回归任务，在此我们使用了随机森林回归来建立预测模型，模型的准确度用均方根误差(RMSE)来衡量。
$$
RMSE = \sqrt{\sum_{i=1}^{n}\frac{({\hat{y_i}}-y_i)^2}{n}}
$$

#### 导入数据以及数据可视化

```python
import numpy as np
import pandas as pd
    
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
train_data.shape, test_data.shape
```

在使用NNI进行自动特征工程前，我们使用了传统的手工方式来对数据进行筛选，基于以下的数据可视化图像，我们可以获得一些筛选的灵感。

![分布](https://img.imgdb.cn/item/604dc3725aedab222cefdca6.png)

根据下图可以看出，预算、流行度与票房有着很强的正相关性。

![](https://img.imgdb.cn/item/604dc3725aedab222cefdca4.png)

由于训练数据中票房偏度比较大，为了使其更服从高斯分布，从而在后续模型中得到更好的效果，我们使用对数函数对其进行转换。

![](https://img.imgdb.cn/item/604dc3725aedab222cefdca1.png)

下图为每种语言的电影的平均票房分布。

![](https://img.imgdb.cn/item/604dc3725aedab222cefdc9e.png)

下图为电影票房与流行度的散点图，从中能看出比较明显的正相关性。![](https://img.imgdb.cn/item/604dc3725aedab222cefdcac.png)

该词云图为训练集电影中出现频度最高的关键词。

![](https://img.imgdb.cn/item/604dc3775aedab222cefe216.png)

由于该数据集中的数值数据较少，为了能够更加准确地训练模型预测票房，我们采用了例如独热编码、去量纲的经典方法对数据进行了预处理，并对部分特殊缺失值进行了填补。

```python
class TextToDictTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, features):
        self.features = features
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        for column in self.features:
            X[column] = X[column].apply(lambda x: {} if pd.isna(x) else literal_eval(x))
        return X
    
class BooleanTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, features):
        self.features = features
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        try:
            for column in self.features:
                X[column] = X[column].apply(lambda x: 1 if x != {} and pd.isna(x) == False else 0)
        except Exception as ex:
            print("Boolean transformer error:", ex)
        return X
    
class OneHotTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, features, top_values):
        self.features = features
        self.top_values = top_values
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        try:
            i = 0
            for feature in self.features:
                for name in self.top_values[i]:
                    X[f'{feature}_{name}'] = X[feature].apply(lambda x: 1 if name in str(x) else 0)
                i += 1
                    
            X = X.drop(self.features, axis=1)
        except Exception as ex:
            print("One hot tansformer error:", ex)
        return X
    
class CastTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, top_cast_names, top_cast_chars):
        self.top_cast_names = top_cast_names
        self.top_cast_chars = top_cast_chars
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        try:
            X['cast_len'] = X['cast'].apply(lambda x: len(x) if x != {} else 0)
            
            for name in self.top_cast_names:
                X[f'cast_name_{name}'] = X['cast'].apply(lambda x: 1 if name in str(x) else 0)
                
            for name in self.top_cast_chars:
                X[f'cast_char_{name}'] = X['cast'].apply(lambda x: 1 if name in str(x) else 0)
            
            X['cast_gender_undef'] = X['cast'].apply(lambda x: sum([1 for i in x if i['gender'] == 0]))
            X['cast_gender_male'] = X['cast'].apply(lambda x: sum([1 for i in x if i['gender'] == 1]))
            X['cast_gender_female'] = X['cast'].apply(lambda x: sum([1 for i in x if i['gender'] == 2]))
            
            X = X.drop('cast', axis=1)
        except Exception as ex:
            print("Cast transformer error:", ex)
        return X
    
class CrewTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, top_crew_names, top_crew_jobs, top_crew_departments):
        self.top_crew_names = top_crew_names
        self.top_crew_jobs = top_crew_jobs
        self.top_crew_departments = top_crew_departments
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        try:
            X['crew_len'] = X['crew'].apply(lambda x: len(x) if x != {} else 0)
            
            for name in self.top_crew_names:
                X[f'crew_name_{name}'] = X['crew'].apply(lambda x: 1 if name in str(x) else 0)
                
            for name in self.top_crew_jobs:
                X[f'crew_job_{name}'] = X['crew'].apply(lambda x: 1 if name in str(x) else 0)
                
            for name in self.top_crew_departments:
                X[f'crew_department_{name}'] = X['crew'].apply(lambda x: 1 if name in str(x) else 0)
            
            X['crew_gender_undef'] = X['crew'].apply(lambda x: sum([1 for i in x if i['gender'] == 0]))
            X['crew_gender_male'] = X['crew'].apply(lambda x: sum([1 for i in x if i['gender'] == 1]))
            X['crew_gender_female'] = X['crew'].apply(lambda x: sum([1 for i in x if i['gender'] == 2]))
            
            X = X.drop('crew', axis=1)
        except Exception as ex:
            print("Crew transformer error:", ex)
        return X
    
class DateTransformer(BaseEstimator, TransformerMixin):        

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        try:            
            X['year'] = pd.Series(pd.DatetimeIndex(X['release_date']).year)
            X['month'] = pd.Series(pd.DatetimeIndex(X['release_date']).month)
            X['day'] = pd.Series(pd.DatetimeIndex(X['release_date']).day)
            X = X.drop('release_date', axis=1)
        except Exception as ex:
            print("Date transformer pipeline error:", ex)
        return X
    
class FixRevenueTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        try:
            X.loc[X['id'] == 16,'revenue'] = 192864          # Skinning
            X.loc[X['id'] == 90,'budget'] = 30000000         # Sommersby          
            X.loc[X['id'] == 118,'budget'] = 60000000        # Wild Hogs
            X.loc[X['id'] == 149,'budget'] = 18000000        # Beethoven
            X.loc[X['id'] == 313,'revenue'] = 12000000       # The Cookout 
            X.loc[X['id'] == 451,'revenue'] = 12000000       # Chasing Liberty
            X.loc[X['id'] == 464,'budget'] = 20000000        # Parenthood
            X.loc[X['id'] == 470,'budget'] = 13000000        # The Karate Kid, Part II
            X.loc[X['id'] == 513,'budget'] = 930000          # From Prada to Nada
            X.loc[X['id'] == 797,'budget'] = 8000000         # Welcome to Dongmakgol
            X.loc[X['id'] == 819,'budget'] = 90000000        # Alvin and the Chipmunks: The Road Chip
            X.loc[X['id'] == 850,'budget'] = 90000000        # Modern Times
            X.loc[X['id'] == 1112,'budget'] = 7500000        # An Officer and a Gentleman
            X.loc[X['id'] == 1131,'budget'] = 4300000        # Smokey and the Bandit   
            X.loc[X['id'] == 1359,'budget'] = 10000000       # Stir Crazy 
            X.loc[X['id'] == 1542,'budget'] = 1              # All at Once
            X.loc[X['id'] == 1570,'budget'] = 15800000       # Crocodile Dundee II
            X.loc[X['id'] == 1571,'budget'] = 4000000        # Lady and the Tramp
            X.loc[X['id'] == 1714,'budget'] = 46000000       # The Recruit
            X.loc[X['id'] == 1721,'budget'] = 17500000       # Cocoon
            X.loc[X['id'] == 1865,'revenue'] = 25000000      # Scooby-Doo 2: Monsters Unleashed
            X.loc[X['id'] == 2268,'budget'] = 17500000       # Madea Goes to Jail budget
            X.loc[X['id'] == 2491,'revenue'] = 6800000       # Never Talk to Strangers
            X.loc[X['id'] == 2602,'budget'] = 31000000       # Mr. Holland's Opus
            X.loc[X['id'] == 2612,'budget'] = 15000000       # Field of Dreams
            X.loc[X['id'] == 2696,'budget'] = 10000000       # Nurse 3-D
            X.loc[X['id'] == 2801,'budget'] = 10000000       # Fracture
            X.loc[X['id'] == 3889,'budget'] = 15000000       # Colossal
            X.loc[X['id'] == 6733,'budget'] = 5000000        # The Big Sick
            X.loc[X['id'] == 3197,'budget'] = 8000000        # High-Rise
            X.loc[X['id'] == 6683,'budget'] = 50000000       # The Pink Panther 2
            X.loc[X['id'] == 5704,'budget'] = 4300000        # French Connection II
            X.loc[X['id'] == 6109,'budget'] = 281756         # Dogtooth
            X.loc[X['id'] == 7242,'budget'] = 10000000       # Addams Family Values
            X.loc[X['id'] == 7021,'budget'] = 17540562       #  Two Is a Family
            X.loc[X['id'] == 5591,'budget'] = 4000000        # The Orphanage
            X.loc[X['id'] == 4282,'budget'] = 20000000       # Big Top Pee-wee

            power_six = X.id[X.budget > 1000][X.revenue < 100]

            for k in power_six :
                X.loc[X['id'] == k,'revenue'] =  X.loc[X['id'] == k,'revenue'] * 1000000
                
            return X
        
        except Exception as ex:
            print("Fix revenue transformer error:", ex)
            
class DropFeaturesTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, features):
        self.features = features
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        try:
            return X.drop(self.features, axis=1)
        except Exception as ex:
            print("Drop features transformer error:", ex)
            

class TrainTestTransformer(BaseEstimator, TransformerMixin):        
    def __init__(self, impute=False, normalize=False):
        self.impute = impute
        self.normalize = normalize
        
    def fit(self, X, y=None):
        
        if self.impute:
            X = X.fillna(X.median())
    
        self.X = X.drop('revenue', axis=1)    
        self.y = X['revenue']
        
        if self.normalize:
            self.X = MinMaxScaler().fit_transform(self.X)
        
        return self
    
    def transform(self, X):
        return train_test_split(self.X, self.y, test_size=0.10)

def top_values(X, column, attribute):

    try:
        values = X[column].apply(lambda x: [i[attribute] for i in x] if x != {} else []).values
        top_values = Counter([j for i in values for j in i]).most_common(30)
        top_values = [i[0] for i in top_values]
        return top_values
    except Exception as ex:
        print(ex)
```

我们选用了随机森林模型，利用`sklearn`库中的实现来训练随机森林模型，为了获得更好的效果，我们单词采用5个随机森林模型并从中选出效果最好的一个。

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

for i in range(num_models):
    forest_reg = RandomForestRegressor(n_estimators=100)
    forest_reg.fit(X_train, np.log1p(y_train))

    preds = forest_reg.predict(sample_data)
    forest_mse = mean_squared_error(sample_labels, preds)
    forest_rmse = np.sqrt(forest_mse)
    
    forest_reg_models.append((forest_reg, forest_rmse))
```

![](https://img.imgdb.cn/item/604dc3775aedab222cefe219.png)

为了得到更好的模型效果，我们采用了网格搜索(Grid Search)的方法对模型进行了进一步的参数优化。

```python
from sklearn.model_selection import GridSearchCV

param_grid = [
    {'bootstrap': [False], 'n_estimators': [200, 250, 300], 'max_features': [60, 80, 100]},
    {'oob_score': [True, False], 'n_estimators': [150, 180, 200], 'max_features': [40, 50, 60]},
] 

grid_search = GridSearchCV(RandomForestRegressor(), param_grid, cv=10, scoring='neg_mean_squared_error', return_train_score=True)
grid_search.fit(X_train, np.log1p(y_train))
```

最终在使用NNI前，我们所得出的最佳模型均方根误差(RMSE)为：**2.102**

在使用NNI中的`gradient_selector`进行特征选择后，经测试当使用16个维度的特征进行建模时效果最佳，模型均方根误差(RMSE)减小到了**1.96**，表现有了显著提高。

```diff
+ from nni.algorithms.feature_engineering.gradient_selector import FeatureGradientSelector
+ from nni.algorithms.feature_engineering.gbdt_selector import GBDTSelector
+ fgs = FeatureGradientSelector(n_features=16, classification=False)

+ fgs.fit(X_train, np.log1p(y_train))
print(fgs.get_selected_features())
+ selected_features=fgs.get_selected_features()

+ X_train_selected=X_train.iloc[:,selected_features]
+ X_valid_selected=X_valid.iloc[:,selected_features]
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
+ sample_data = X_valid_selected[:]
+ sample_labels = np.log1p(y_valid[:])
num_models = 5
forest_reg_models = []
for i in range(num_models):
    forest_reg = RandomForestRegressor(n_estimators=100)
    forest_reg.fit(X_train_selected, np.log1p(y_train))

    preds = forest_reg.predict(sample_data)
    forest_mse = mean_squared_error(sample_labels, preds)
    forest_rmse = np.sqrt(forest_mse)
    
    forest_reg_models.append((forest_reg, forest_rmse))
```



## 实验结果

|  Dataset   | baseline accuracy/RMSE | automl accuracy/RMSE |  dataset link|
|  ----  | ----  | ----  | ----  |
| 电影票房预测：TMDB Box Office Prediction | 2.102 | 1.960                | [Link]((https://www.kaggle.com/c/tmdb-box-office-prediction/data)) |
| 旧金山犯罪分类：San Francisco Crime Classification      |  |                      | [Link](https://www.kaggle.com/c/sf-crime/data) |
| 土壤属性预测：Africa Soil Property Prediction Challenge ||                      | [Link](https://www.kaggle.com/c/afsis-soil-properties/data) |
