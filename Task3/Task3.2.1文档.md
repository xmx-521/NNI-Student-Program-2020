

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

## 2. 旧金山犯罪分类

### 任务目标

1. 通过数据训练神经网络实现旧金山市犯罪分类
2. 使用NNI进行特征工程并与原数据训练结果进行对比

### 数据处理

可以观察到test set中的特征中并没有提到train set中的Descript，Resolution相关的特征，因此将这两个特征直接删除

```python
train_data.drop(['Descript','Resolution'],inplace=True,axis=1)
```

又观察到address特征的值域过于庞大，难以进行独热编码，并且与经纬度作标特征表达出的信息有一定的重合度，故直接去除

```python
train_data.drop(['Descript','Resolution'],inplace=True,axis=1)
train_data.drop('Address',inplace=True,axis=1)
test_data.drop('Address',inplace=True,axis=1)
```

同时对经纬度缺失的作标，使用同PdDistrict数据的经纬度平均值替代

```python
train_data.drop_duplicates(inplace=True)
train_data.replace({'X': -120.5, 'Y': 90.0}, np.NaN, inplace=True)
test_data.replace({'X': -120.5, 'Y': 90.0}, np.NaN, inplace=True)

imp = SimpleImputer(strategy='mean')

for district in train_data['PdDistrict'].unique():
    train_data.loc[train_data['PdDistrict'] == district, ['X', 'Y']] = imp.fit_transform(
        train_data.loc[train_data['PdDistrict'] == district, ['X', 'Y']])
    test_data.loc[test_data['PdDistrict'] == district, ['X', 'Y']] = imp.transform(
        test_data.loc[test_data['PdDistrict'] == district, ['X', 'Y']])
```

对星期与年月日时分进行数据化

```python
train_data['DayOfWeek'].replace(to_replace=['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'],value=[i for i in range(0,7)],inplace=True)
test_data['DayOfWeek'].replace(to_replace=['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'],value=[i for i in range(0,7)],inplace=True)

ls1=list(train_data['Dates'])
for i in range(len(ls1)):
    ls1[i]=ls1[i][:4:]
train_data['Year']=ls1 

ls1=list(train_data['Dates'])
for i in range(len(ls1)):
    ls1[i]=ls1[i][5:7:]
train_data['Month']=ls1 

ls1=list(train_data['Dates'])
for i in range(len(ls1)):
    ls1[i]=ls1[i][8:10:]
train_data['Day']=ls1 

ls1=list(train_data['Dates'])
for i in range(len(ls1)):
    ls1[i]=ls1[i][11:-6:]
train_data['Hours']=ls1

ls1=list(train_data['Dates'])
for i in range(len(ls1)):
    ls1[i]=ls1[i][-5:-3:]
train_data['Minutes']=ls1

train_data.drop('Dates',axis=1,inplace=True)

ls1=list(test_data['Dates'])
for i in range(len(ls1)):
    ls1[i]=ls1[i][:4:]
test_data['Year']=ls1 

ls1=list(test_data['Dates'])
for i in range(len(ls1)):
    ls1[i]=ls1[i][5:7:]
test_data['Month']=ls1 

ls1=list(test_data['Dates'])
for i in range(len(ls1)):
    ls1[i]=ls1[i][8:10:]
test_data['Day']=ls1 

ls1=list(test_data['Dates'])
for i in range(len(ls1)):
    ls1[i]=ls1[i][11:-6:]
test_data['Hours']=ls1

ls1=list(test_data['Dates'])
for i in range(len(ls1)):
    ls1[i]=ls1[i][-5:-3:]
test_data['Minutes']=ls1

test_data.drop('Dates',axis=1,inplace=True)
```

对PdDistrict进行独热编码

```python
dummies_train=pd.get_dummies(train_data['PdDistrict'])
dummies_test=pd.get_dummies(test_data['PdDistrict'])

train_data=pd.concat([train_data,dummies_train],axis=1)
train_data.drop('PdDistrict',inplace=True,axis=1)

test_data=pd.concat([test_data,dummies_test],axis=1)
test_data.drop('PdDistrict',inplace=True,axis=1)
```

### 特征工程

使用NNI的FeatureGradientSelector进行特征工程

```python
fgs = FeatureGradientSelector(n_features=14,classification=True,batch_size=731225)
fgs.fit(X_train, y_train)
res=fgs.get_selected_features()
```

尽管进行了独热编码，但是本问题的特征数还是只有18条之少，因此NNI特征工程的结果并不理想，指定挑出了14条特征，进行后续的训练

### 训练数据

使用原始数据的训练数据

```
Epoch 1/20
24630/24630 - 38s - loss: 2.6556 - accuracy: 0.2157 - val_loss: 2.6149 - val_accuracy: 0.2232
Epoch 2/20
24630/24630 - 37s - loss: 2.5970 - accuracy: 0.2288 - val_loss: 2.6254 - val_accuracy: 0.2134
Epoch 3/20
24630/24630 - 36s - loss: 2.5891 - accuracy: 0.2306 - val_loss: 2.5979 - val_accuracy: 0.2255
Epoch 4/20
24630/24630 - 36s - loss: 2.5803 - accuracy: 0.2344 - val_loss: 2.5657 - val_accuracy: 0.2325
Epoch 5/20
24630/24630 - 37s - loss: 2.5752 - accuracy: 0.2355 - val_loss: 2.5645 - val_accuracy: 0.2376
Epoch 6/20
24630/24630 - 37s - loss: 2.5646 - accuracy: 0.2392 - val_loss: 2.5571 - val_accuracy: 0.2385
Epoch 7/20
24630/24630 - 37s - loss: 2.5686 - accuracy: 0.2366 - val_loss: 2.5783 - val_accuracy: 0.2322
Epoch 8/20
24630/24630 - 37s - loss: 2.5883 - accuracy: 0.2295 - val_loss: 2.5928 - val_accuracy: 0.2288
Epoch 9/20
24630/24630 - 37s - loss: 2.5911 - accuracy: 0.2306 - val_loss: 2.5897 - val_accuracy: 0.2306
Epoch 10/20
24630/24630 - 37s - loss: 2.5886 - accuracy: 0.2333 - val_loss: 2.5911 - val_accuracy: 0.2310
Epoch 11/20
24630/24630 - 37s - loss: 2.5706 - accuracy: 0.2380 - val_loss: 2.5633 - val_accuracy: 0.2383
Epoch 12/20
24630/24630 - 37s - loss: 2.5619 - accuracy: 0.2391 - val_loss: 2.5507 - val_accuracy: 0.2388
Epoch 13/20
24630/24630 - 37s - loss: 2.5674 - accuracy: 0.2382 - val_loss: 2.5648 - val_accuracy: 0.2318
Epoch 14/20
24630/24630 - 34s - loss: 2.5658 - accuracy: 0.2378 - val_loss: 2.5539 - val_accuracy: 0.2387
Epoch 15/20
24630/24630 - 35s - loss: 2.5589 - accuracy: 0.2412 - val_loss: 2.5471 - val_accuracy: 0.2418
Epoch 16/20
24630/24630 - 34s - loss: 2.5571 - accuracy: 0.2419 - val_loss: 2.5546 - val_accuracy: 0.2398
Epoch 17/20
24630/24630 - 34s - loss: 2.5548 - accuracy: 0.2430 - val_loss: 2.5462 - val_accuracy: 0.2429
Epoch 18/20
24630/24630 - 34s - loss: 2.5521 - accuracy: 0.2433 - val_loss: 2.5503 - val_accuracy: 0.2428
Epoch 19/20
24630/24630 - 34s - loss: 2.5535 - accuracy: 0.2431 - val_loss: 2.5519 - val_accuracy: 0.2422
Epoch 20/20
24630/24630 - 34s - loss: 2.5511 - accuracy: 0.2440 - val_loss: 2.5596 - val_accuracy: 0.2391
```

使用NNI特征工程筛选后的训练数据

```
Epoch 1/20
24630/24630 - 34s - loss: 2.5795 - accuracy: 0.2332 - val_loss: 2.5569 - val_accuracy: 0.2387
Epoch 2/20
24630/24630 - 33s - loss: 2.5345 - accuracy: 0.2417 - val_loss: 2.5344 - val_accuracy: 0.2380
Epoch 3/20
24630/24630 - 32s - loss: 2.5277 - accuracy: 0.2424 - val_loss: 2.5181 - val_accuracy: 0.2424
Epoch 4/20
24630/24630 - 34s - loss: 2.5303 - accuracy: 0.2400 - val_loss: 2.5215 - val_accuracy: 0.2408
Epoch 5/20
24630/24630 - 32s - loss: 2.5266 - accuracy: 0.2412 - val_loss: 2.5147 - val_accuracy: 0.2407
Epoch 6/20
24630/24630 - 40s - loss: 2.5274 - accuracy: 0.2403 - val_loss: 2.5150 - val_accuracy: 0.2405
Epoch 7/20
24630/24630 - 44s - loss: 2.5328 - accuracy: 0.2381 - val_loss: 2.5457 - val_accuracy: 0.2302
Epoch 8/20
24630/24630 - 46s - loss: 2.5327 - accuracy: 0.2385 - val_loss: 2.5230 - val_accuracy: 0.2375
Epoch 9/20
24630/24630 - 31s - loss: 2.5294 - accuracy: 0.2387 - val_loss: 2.5217 - val_accuracy: 0.2380
Epoch 10/20
24630/24630 - 29s - loss: 2.5279 - accuracy: 0.2395 - val_loss: 2.5183 - val_accuracy: 0.2408
Epoch 11/20
24630/24630 - 31s - loss: 2.5273 - accuracy: 0.2398 - val_loss: 2.5222 - val_accuracy: 0.2374
Epoch 12/20
24630/24630 - 35s - loss: 2.5272 - accuracy: 0.2395 - val_loss: 2.5349 - val_accuracy: 0.2344
Epoch 13/20
24630/24630 - 34s - loss: 2.5271 - accuracy: 0.2394 - val_loss: 2.5353 - val_accuracy: 0.2355
Epoch 14/20
24630/24630 - 33s - loss: 2.5305 - accuracy: 0.2393 - val_loss: 2.5395 - val_accuracy: 0.2348
Epoch 15/20
24630/24630 - 33s - loss: 2.5458 - accuracy: 0.2354 - val_loss: 2.5613 - val_accuracy: 0.2305
Epoch 16/20
24630/24630 - 36s - loss: 2.5365 - accuracy: 0.2375 - val_loss: 2.5293 - val_accuracy: 0.2370
Epoch 17/20
24630/24630 - 33s - loss: 2.5291 - accuracy: 0.2401 - val_loss: 2.5208 - val_accuracy: 0.2395
Epoch 18/20
24630/24630 - 34s - loss: 2.5307 - accuracy: 0.2383 - val_loss: 2.5275 - val_accuracy: 0.2377
Epoch 19/20
24630/24630 - 36s - loss: 2.5281 - accuracy: 0.2393 - val_loss: 2.5189 - val_accuracy: 0.2387
Epoch 20/20
24630/24630 - 34s - loss: 2.5246 - accuracy: 0.2405 - val_loss: 2.5192 - val_accuracy: 0.2398
```

### 结论分析

由于本问题的特征过少，因此NNI进行特征工程的空间并不大，可以看出实验的结果并不理想，未使用特征工程的原始数据训练的结果好于进行特征工程的训练数据

## 3. 土壤属性预测

#### 任务目标：

1. 基于训练数据建立模型以准确预测土壤属性
2. 基于NNI对数据进行特征选择，并与原模型比较经过NNI进行特征选择后的准确率

#### 模型描述：

土壤属性不止一个，因此该问题为multi-label问题，为了使用NNI，将multi-label问题拆解为多个回归问题分别用建模，并使用mean squared error衡量预测的准确程度

#### 数据清洗与预处理

无论是传统方法还是使用NNI的autoML方法，数据的清洗与预处理都是必要的

先使用pandas了解训练集的基本情况

``` python
import pandas as pd
# MARK: - Read dataset as df
datasetAsDf = pd.read_csv('dataset/training.csv')
datasetAsDf.info()
```

```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 1157 entries, 0 to 1156
Columns: 3600 entries, PIDN to Sand
dtypes: float64(3598), object(2)
memory usage: 31.8+ MB
```

feature数量竟有接近3600列之多，而sample数量仅有稀少的1000+。

不存在空值是个好消息，然而存在不为数值类型的feature，我们选择one-hot编码的方式将其转为数据类型，在此之前，先将labels与features分离开，并将labels做进一步拆解将multi-label问题转为多个回归问题。

```python
# MARK: - Split dataset into input features and labels
datasetFeaturesAsDf = datasetAsDf.iloc[:, 1:3595]
datasetLabelsAsDf = datasetAsDf.iloc[:, 3595:3600].astype(float)

datasetCaAsDf = datasetLabelsAsDf.loc[:, 'Ca'].astype(float)
datasetPAsDf = datasetLabelsAsDf.loc[:, 'P'].astype(float)
datasetpHAsDf = datasetLabelsAsDf.loc[:, 'pH'].astype(float)
datasetSOCAsDf = datasetLabelsAsDf.loc[:, 'SOC'].astype(float)
datasetSandAsDf = datasetLabelsAsDf.loc[:, 'Sand'].astype(float)

# MARK: - One-hot encode for feature Depth
datasetFeaturesAsDf['Depth'] = pd.get_dummies(datasetFeaturesAsDf['Depth'])
datasetFeaturesAsDf = datasetFeaturesAsDf.astype(float)
```

One-hot编码完成后，为了让训练数据与使用库函数的type与shape相适配，我们对数据的type与shape做一定转换

```python
import numpy as np
import torch
# MARK: - Transform df to array
datasetFeaturesAsArray = datasetFeaturesAsDf.values
datasetLabelsAsArray = datasetLabelsAsDf.values

datasetCaAsArray = datasetCaAsDf.values
datasetPAsArray = datasetPAsDf.values
datasetpHAsArray = datasetpHAsDf.values
datasetSOCAsArray = datasetSOCAsDf.values
datasetSandAsArray = datasetSandAsDf.values

# MARK: - Reshape array as Matrix
datasetCaAsMatrix = np.matrix(datasetCaAsArray.reshape(
    (-1, 1))).astype(float)
datasetPAsMatrix = np.matrix(datasetPAsArray.reshape(
    (-1, 1))).astype(float)
datasetpHAsMatrix = np.matrix(datasetpHAsArray.reshape(
    (-1, 1))).astype(float)
datasetSOCAsMatrix = np.matrix(datasetSOCAsArray.reshape(
    (-1, 1))).astype(float)
datasetSandAsMatrix = np.matrix(datasetSandAsArray.reshape(
    (-1, 1))).astype(float)
```

到此为止，数据的清洗与预处理基本完成

#### FeatureEngineering

接下来，我们将使用NNI的FeatureGradientSelector做特征工程。

由于数据集的特征数非常之多，我们先不指定目标特征的数量，让NNI自主选择保留的feature数。

然而这种做法还是出现了一定问题，对于'P'这个label来说，若不指定保留feature数，FeatureGradientSelector返回的特征为空列表，因此此处武断地指定n_features = 30。对其余label的处理完全一致，因此文档以对label 'Ca' 的处理为例。在FeatureGradientSelecto进行选择前，先将原始数据划分为训练集与测试集。

```python
# MARK: - NNI Feature Engineering and Kernel Ridge Regression for every target

# MARK: - Ca
XTrainCa, XTestCa, yTrainCa, yTestCa = train_test_split(
    datasetFeaturesAsDf.copy(),
    datasetCaAsMatrix,
    test_size=0.2,
    random_state=42)

fgsCa = FeatureGradientSelector(verbose=1,
                                n_epochs=100,
                                learning_rate=1e-1,
                                classification=False,
                                shuffle=True)
fgsCa.fit(XTrainCa, yTrainCa)
CaSelectedFeatureIndices = fgsCa.get_selected_features()
```

完成特征的选择后，我们对训练集进行进一步取舍

```
XTrainCa = XTrainCa.iloc[:, CaSelectedFeatureIndices]
XTestCa = XTestCa.iloc[:, CaSelectedFeatureIndices]
```

至此FeatureEngineering结束

#### 模型训练

我们使用sklearn库中的KernelRidge进行Kernel Ridge Regression，并通过mean squared error衡量模型准确率。

同样以对label 'Ca' 的处理为例

```python
from sklearn.model_selection import train_test_split
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error

krCa = KernelRidge(alpha=0.1, kernel='polynomial', degree=7, coef0=2.5)
krCa.fit(XTrainCa, yTrainCa)
yPredCa = krCa.predict(XTestCa)
print(mean_squared_error(yPredCa, yTestCa))
```

#### 实验结果

| Label | 使用NNI前 (MSE)     | 使用NNI后（MSE)     |
| ----- | ------------------- | ------------------- |
| Ca    | 0.12243475082247697 | 0.21281624866017917 |
| P     | 0.7704961827290547  | 0.6273964529539843  |
| pH    | 0.16336261500494445 | 0.29011283323164444 |
| SOC   | 0.06636032308609842 | 0.06095249184385086 |
| Sand  | 0.06972229859748705 | 0.0753824333242412  |

从实验结果中可看出，使用NNI后绝部分模型准确率反而有所下降，这一点在kaggle上也有明显体现，传统方法得分排名在800/1200左右，而使用NNI后排名为1100/1200左右

#### 分析与讨论

实验结果有些令人惊讶，NNI效果不佳的原因总结分析后可能有以下两点：

1. 对于数据特征的挖掘不够，没有把高阶数据挖掘出来供NNI选择

2. 训练集的特征数过大，NNI在提取过大基数的特征时表现不佳



然而实验过程中还出现了一个更加有趣的现象，那就是将FeatureGradientSelector的classification参数设置为True时，kaggle得分排名从1100/1200上涨到1000/1200，然而本任务并非分类任务，出现这种情况的具体原因仍不清楚



总的来说，NNI的特征工程功能还是比较好用易用的，未来或许可以从以下几点继续改进：

1. 提高文档的详细程度。文档中的API使用说明让使用者感到有点迷惑，尤其是输入数据类型的说明不够明确
2. 通过增加参数满足使用者的某些需求。例如FeatureGradientSelector的构造函数中可以增加参数指定最小特征数
3. FeatureGradientSelector似乎在特征数量过大时表现不佳，算法或许可以从这个方面进一步优化

## 实验结果

|  Dataset   | baseline RMSE/accuracy/MSE | automl RMSE/accuracy/MSE |  dataset link|
|  ----  | ----  | ----  | ----  |
| 电影票房预测：TMDB Box Office Prediction | 2.102 | 1.960                | [Link]((https://www.kaggle.com/c/tmdb-box-office-prediction/data)) |
| 旧金山犯罪分类：San Francisco Crime Classification      | 0.2440 | 0.2405 | [Link](https://www.kaggle.com/c/sf-crime/data) |
| 土壤属性预测：Africa Soil Property Prediction Challenge |0.12\|0.77\|0.16\|0.07\|0.07| 0.21\|0.63\|0.30\|0.06\|0.08 | [Link](https://www.kaggle.com/c/afsis-soil-properties/data) |
