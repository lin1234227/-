#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import r2_score
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
import warnings
import datetime
warnings.filterwarnings('ignore')
sns.set_style('darkgrid')
get_ipython().run_line_magic('matplotlib', 'inline')
# plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei']


# # 数据分析

# In[2]:


train = pd.read_csv('../data/train_data.csv')
test = pd.read_csv('../data/test_data.csv')


# ## 缺失值分析

# In[4]:


train['tradeNewMeanPrice']


# ### 一行代码
# 

# In[3]:


train.isnull().sum().sort_values()


# In[4]:


test.isnull().sum().sort_values()


# ### 封装函数

# In[5]:


def missing_values(df):
    alldata_na = pd.DataFrame(df.isnull().sum(), columns={'missingNum'})
    alldata_na['existNum'] = len(df) - alldata_na['missingNum']
    alldata_na['sum'] = len(df)
    alldata_na['missingRatio'] = alldata_na['missingNum']/len(df)*100
    alldata_na['dtype'] = df.dtypes
    #ascending：默认True升序排列；False降序排列
    alldata_na = alldata_na[alldata_na['missingNum']>0].reset_index().sort_values(by=['missingNum','index'],ascending=[False,True])
    alldata_na.set_index('index',inplace=True)
    return alldata_na


# In[6]:


missing_values(train)


# In[7]:


missing_values(test)


# ## 特征值分析

# ### 单调特征

# In[8]:


#是否有单调特征列(单调的特征列很大可能是时间)
def incresing(vals):
    cnt = 0
    len_ = len(vals)
    for i in range(len_-1):
        if vals[i+1] > vals[i]:
            cnt += 1
    return cnt

fea_cols = [col for col in train.columns]
for col in fea_cols:
    cnt = incresing(train[col].values)
    if cnt / train.shape[0] >= 0.55:
        print('单调特征：',col)
        print('单调特征值个数：', cnt)
        print('单调特征值比例：', cnt / train.shape[0])
        


# ### 特征nunique分布

# In[9]:


train.nunique().sort_values()


# In[10]:


test.nunique().sort_values()


# In[11]:


cat = ['rentType', 'houseType', 'houseFloor', 'houseToward', 'houseDecoration',  'city', 'region', 'plate', 'buildYear', 'tradeTime']


# In[12]:


train[cat].nunique().plot(kind='bar',rot=45)


# In[13]:


test[cat].nunique().plot(kind='bar',rot=45)


# In[14]:


# 统计特征值出现频次大于100的特征
fea_cols = train.columns
interesting_cols = []
for col in fea_cols:
    if train[col].value_counts().iloc[0] > 1000:
        print(col)
        print(train[col].value_counts().iloc[:3])
        interesting_cols.append(col)


# In[15]:


interesting_cols


# ## Label分布

# In[146]:


sns.distplot(train['tradeMoney'],hist=True)


# In[147]:


train['tradeMoney'].describe()


# In[148]:


sns.distplot(train[train['tradeMoney']<55000]['tradeMoney'],hist=True)


# In[149]:


sns.distplot(np.log1p(train[(train['tradeMoney']<55000)&(train['tradeMoney']>300)]['tradeMoney']),hist=True)
plt.title('log on tradeMoney')
plt.show()


# In[159]:


fig,axes = plt.subplots(2,3,figsize=(20,5))
fig.set_size_inches(20,12)
sns.distplot(train['tradeMoney'],ax=axes[0][0])
sns.distplot(train[(train['tradeMoney']<=15000)]['tradeMoney'],ax=axes[0][1])
sns.distplot(train[(train['tradeMoney']>15000)&(train['tradeMoney']<=20000)]['tradeMoney'],ax=axes[0][2])
sns.distplot(train[(train['tradeMoney']>20000)&(train['tradeMoney']<=50000)]['tradeMoney'],ax=axes[1][0])
sns.distplot(train[(train['tradeMoney']>50000)&(train['tradeMoney']<=100000)]['tradeMoney'],ax=axes[1][1])
sns.distplot(train[(train['tradeMoney']>100000)]['tradeMoney'],ax=axes[1][2])

print("money<=15000 ",len(train[(train['tradeMoney']<=15000)]['tradeMoney']))
print("10000<money<=20000 ",len(train[(train['tradeMoney']>16000)&(train['tradeMoney']<=20000)]['tradeMoney']))
print("20000<money<=50000 ",len(train[(train['tradeMoney']>20000)&(train['tradeMoney']<=50000)]['tradeMoney']))
print("50000<money<=100000 ",len(train[(train['tradeMoney']>50000)&(train['tradeMoney']<=100000)]['tradeMoney']))
print("100000<money ",len(train[(train['tradeMoney']>100000)]['tradeMoney']))


# ## 数据清洗

# ### 清洗前

# In[176]:


# 数据清洗
data = train.copy()
g= sns.lmplot('area','tradeMoney',hue='rentType',col='region', col_wrap=3,data=data,sharex=False, sharey=False,palette='husl',scatter_kws={'alpha':0.3} )
plt.tight_layout()
plt.show()


# ### 清洗后

# In[177]:


# Clean Data
def cleanData(data):    
    data.drop(data[(data['tradeMoney']>16000)].index,inplace=True) 
    data.drop(data[(data['area']>160)].index,inplace=True)
    data.drop(data[(data['tradeMoney']<100)].index,inplace=True)
    data.drop(data[(data['totalFloor']==0)].index,inplace=True)

    # 深度清理
    data.drop(data[(data['region']=='RG00001')&(data['tradeMoney']<1000)&(data['area']>50)].index,inplace=True)
    data.drop(data[(data['region']=='RG00001') & (data['tradeMoney']>25000)].index,inplace=True)
    data.drop(data[(data['region']=='RG00001') & (data['area']>250)&(data['tradeMoney']<20000)].index,inplace=True)
    data.drop(data[(data['region']=='RG00001') & (data['area']>400)&(data['tradeMoney']>50000)].index,inplace=True)
    data.drop(data[(data['region']=='RG00001') & (data['area']>100)&(data['tradeMoney']<2000)].index,inplace=True)
    data.drop(data[(data['region']=='RG00002') & (data['area']<100)&(data['tradeMoney']>60000)].index,inplace=True)
    data.drop(data[(data['region']=='RG00003') & (data['area']<300)&(data['tradeMoney']>30000)].index,inplace=True)
    data.drop(data[(data['region']=='RG00003')&(data['tradeMoney']<500)&(data['area']<50)].index,inplace=True)
    data.drop(data[(data['region']=='RG00003')&(data['tradeMoney']<1500)&(data['area']>100)].index,inplace=True)
    data.drop(data[(data['region']=='RG00003')&(data['tradeMoney']<2000)&(data['area']>300)].index,inplace=True)
    data.drop(data[(data['region']=='RG00003')&(data['tradeMoney']>5000)&(data['area']<20)].index,inplace=True)
    data.drop(data[(data['region']=='RG00003') & (data['area']>600)&(data['tradeMoney']>40000)].index,inplace=True)
    data.drop(data[(data['region']=='RG00004')&(data['tradeMoney']<1000)&(data['area']>80)].index,inplace=True)
    data.drop(data[(data['region']=='RG00006') & (data['tradeMoney']<200)].index,inplace=True)
    data.drop(data[(data['region']=='RG00005')&(data['tradeMoney']<2000)&(data['area']>180)].index,inplace=True)
    data.drop(data[(data['region']=='RG00005')&(data['tradeMoney']>50000)&(data['area']<200)].index,inplace=True)
    data.drop(data[(data['region']=='RG00006') & (data['area']>200)&(data['tradeMoney']<2000)].index,inplace=True)
    data.drop(data[(data['region']=='RG00007') & (data['area']>100)&(data['tradeMoney']<2500)].index,inplace=True)
    data.drop(data[(data['region']=='RG00010') & (data['area']>200)&(data['tradeMoney']>25000)].index,inplace=True)
    data.drop(data[(data['region']=='RG00010') & (data['area']>400)&(data['tradeMoney']<15000)].index,inplace=True)
    data.drop(data[(data['region']=='RG00010')&(data['tradeMoney']<3000)&(data['area']>200)].index,inplace=True)
    data.drop(data[(data['region']=='RG00010')&(data['tradeMoney']>7000)&(data['area']<75)].index,inplace=True)
    data.drop(data[(data['region']=='RG00010')&(data['tradeMoney']>12500)&(data['area']<100)].index,inplace=True)
    data.drop(data[(data['region']=='RG00004') & (data['area']>400)&(data['tradeMoney']>20000)].index,inplace=True)
    data.drop(data[(data['region']=='RG00008')&(data['tradeMoney']<2000)&(data['area']>80)].index,inplace=True)
    data.drop(data[(data['region']=='RG00009') & (data['tradeMoney']>40000)].index,inplace=True)
    data.drop(data[(data['region']=='RG00009') & (data['area']>300)].index,inplace=True)
    data.drop(data[(data['region']=='RG00009')&(data['area']>100)&(data['tradeMoney']<2000)].index,inplace=True)
    data.drop(data[(data['region']=='RG00011')&(data['tradeMoney']<10000)&(data['area']>390)].index,inplace=True)
    data.drop(data[(data['region']=='RG00012') & (data['area']>120)&(data['tradeMoney']<5000)].index,inplace=True)
    data.drop(data[(data['region']=='RG00013') & (data['area']<100)&(data['tradeMoney']>40000)].index,inplace=True)
    data.drop(data[(data['region']=='RG00013') & (data['area']>400)&(data['tradeMoney']>50000)].index,inplace=True)
    data.drop(data[(data['region']=='RG00013')&(data['area']>80)&(data['tradeMoney']<2000)].index,inplace=True)
    data.drop(data[(data['region']=='RG00014') & (data['area']>300)&(data['tradeMoney']>40000)].index,inplace=True)
    data.drop(data[(data['region']=='RG00014')&(data['tradeMoney']<1300)&(data['area']>80)].index,inplace=True)
    data.drop(data[(data['region']=='RG00014')&(data['tradeMoney']<8000)&(data['area']>200)].index,inplace=True)
    data.drop(data[(data['region']=='RG00014')&(data['tradeMoney']<1000)&(data['area']>20)].index,inplace=True)
    data.drop(data[(data['region']=='RG00014')&(data['tradeMoney']>25000)&(data['area']>200)].index,inplace=True)
    data.drop(data[(data['region']=='RG00014')&(data['tradeMoney']<20000)&(data['area']>250)].index,inplace=True)
    data.drop(data[(data['region']=='RG00005')&(data['tradeMoney']>30000)&(data['area']<100)].index,inplace=True)
    data.drop(data[(data['region']=='RG00005')&(data['tradeMoney']<50000)&(data['area']>600)].index,inplace=True)
    data.drop(data[(data['region']=='RG00005')&(data['tradeMoney']>50000)&(data['area']>350)].index,inplace=True)
    data.drop(data[(data['region']=='RG00006')&(data['tradeMoney']>4000)&(data['area']<100)].index,inplace=True)
    data.drop(data[(data['region']=='RG00006')&(data['tradeMoney']<600)&(data['area']>100)].index,inplace=True)
    data.drop(data[(data['region']=='RG00006')&(data['area']>165)].index,inplace=True)
    data.drop(data[(data['region']=='RG00012')&(data['tradeMoney']<800)&(data['area']<30)].index,inplace=True)
    data.drop(data[(data['region']=='RG00007')&(data['tradeMoney']<1100)&(data['area']>50)].index,inplace=True)
    data.drop(data[(data['region']=='RG00004')&(data['tradeMoney']>8000)&(data['area']<80)].index,inplace=True)
    data.loc[(data['region']=='RG00002')&(data['area']>50)&(data['rentType']=='合租'),'rentType']='整租'
    data.loc[(data['region']=='RG00014')&(data['rentType']=='合租')&(data['area']>60),'rentType']='整租'
    data.drop(data[(data['region']=='RG00008')&(data['tradeMoney']>15000)&(data['area']<110)].index,inplace=True)
    data.drop(data[(data['region']=='RG00008')&(data['tradeMoney']>20000)&(data['area']>110)].index,inplace=True)
    data.drop(data[(data['region']=='RG00008')&(data['tradeMoney']<1500)&(data['area']<50)].index,inplace=True)
    data.drop(data[(data['region']=='RG00008')&(data['rentType']=='合租')&(data['area']>50)].index,inplace=True)
    data.drop(data[(data['region']=='RG00015') ].index,inplace=True)
    data.drop(data[(data['region']=='RG00014')&(data['tradeMoney']>13000)].index,inplace=True)
    data.drop(data[(data['region']=='RG00005')&(data['tradeMoney']<1200)&(data['area']>80)].index,inplace=True)
    data.drop(data[(data['region']=='RG00005')&(data['tradeMoney']>13000)&(data['area']<90)].index,inplace=True)
    data.drop(data[(data['region']=='RG00005')&(data['tradeMoney']<1600)&(data['area']>100)].index,inplace=True)
    data.drop(data[(data['region']=='RG00002')&(data['tradeMoney']>13000)&(data['area']<100)].index,inplace=True)
    data.reset_index(drop=True, inplace=True)
    
    return data


# In[178]:


data = cleanData(data)
g= sns.lmplot('area','tradeMoney',hue='rentType',col='region', col_wrap=3,data=data,sharex=False, sharey=False,palette='husl',scatter_kws={'alpha':0.3} )
plt.tight_layout()
plt.show()


# ## Train & Test分布探索

# ### 画图分析

# #### 原始分布

# In[180]:


def compare_train_test(x_list, train,test, kind, cols=1):
    rows = int(np.ceil(len(x_list)/cols))
    if rows == 1 and len(x_list)<cols:
        cols = len(x_list)
    fig, ax = plt.subplots(nrows=rows, ncols=cols, figsize=[cols*5, rows*4])
    if rows == 1 and cols ==1:
        ax = np.array([ax])
    ax = ax.reshape(rows, cols)
    
    i = 0
    for n in range(len(x_list)):
        xlabel = x_list[n]
        if kind == 'kde':
            sns.distplot(test[xlabel],hist=False,label='test',ax=ax[int(i/cols),i%cols])
            sns.distplot(train[xlabel],hist=False,label='train',ax=ax[int(i/cols),i%cols])
        if kind == 'bar':
            temp_train = pd.DataFrame({'count':train[xlabel].value_counts()}).reset_index()
            temp_train['data_type']='train'
            temp_test = pd.DataFrame({'count':test[xlabel].value_counts()}).reset_index()
            temp_test['data_type']='test'
            data = pd.concat([temp_train,temp_test],axis=0)
            data.columns = [xlabel,'count','data_type']
            sns.barplot(x=xlabel,y='count',hue='data_type',data=data, ax=ax[int(i/cols),i%cols])
        i +=1


# In[41]:


cat_list = ['rentType',  'houseFloor', 'houseToward', 'houseDecoration',  'city', 'region', 'plate', 'buildYear']
compare_train_test(cat_list,train, test,kind='bar', cols=3)


# In[735]:


num_list = [col for col in train.columns if (train[col].dtype!='object')&(col not in ['ID','tradeMoney','tradeTime','houseType',])]
compare_train_test(num_list,train,test, kind='kde', cols=3)


# In[43]:


sns.distplot(test['area'],hist=True,label='test')
sns.distplot(train[train.area<160]['area'],hist=True,label='train')
plt.title('area (train vs test)')
plt.legend()


# In[152]:


sns.distplot(test['area'],hist=True,label='test')
sns.distplot(train[(train.area<160)&(train.area>17)]['area'],hist=True,label='train')
plt.title('area (train vs test)')
plt.legend()


# #### 过大量级值取log平滑(针对线性模型有效)

# In[736]:


big_num_cols = ['totalTradeMoney','totalTradeArea','tradeMeanPrice','totalNewTradeMoney', 'totalNewTradeArea',
                'tradeNewMeanPrice','remainNewNum', 'supplyNewNum', 'supplyLandArea',
                'tradeLandArea','landTotalPrice','landMeanPrice','totalWorkers','newWorkers',
                'residentPopulation','pv','uv']
compare_train_test(big_num_cols,train,test, kind='kde', cols=3)


# In[741]:


for col in big_num_cols:
        train[col] = train[col].map(lambda x: np.log1p(x))
        test[col] = test[col].map(lambda x: np.log1p(x))
compare_train_test(big_num_cols,train,test, kind='kde', cols=3)


# #### 数据清洗后的分布

# In[154]:


# 异常处理
# def cleanOutlier(data):
#     data = data[(data['tradeMoney']<55000)&(data['tradeMoney']>300)]
#     data = data[(data['area']<160)&(data['area']>17)]
#     return data
# data = cleanOutlier(train)
# 数据清洗
data = cleanData(data)


# In[181]:


cat_list = ['rentType',  'houseFloor', 'houseToward', 'houseDecoration',  'city', 'region', 'plate', 'buildYear']
compare_train_test(cat_list,data, test,kind='bar', cols=3)


# In[182]:


num_list = [col for col in train.columns if (train[col].dtype!='object')&(col not in ['ID','tradeMoney','tradeTime','houseType',])]
compare_train_test(num_list,data,test, kind='kde', cols=3)


# ### 对抗验证分析
# ---
# - 若AUC>0.5，说明train和test的数据分布可分，两者分布不统一
# - 若AUC接近0.5，近似随机选择，说明train和test分布相似

# In[293]:


from sklearn.model_selection import StratifiedKFold
import datetime
import lightgbm as lgb

def obj2cate_1(data):
    # 转换object类型数据
    columns = [ 'buildYear', 'houseType', 'tradeTime', 'rentType', 'houseFloor', 'houseToward', 'houseDecoration', 'communityName', 'region', 'plate']
    for col in columns:
        data[col] = data[col].astype('category')
    return data

def validation_prediction_lgb(X, y,feature_names):
    n_fold = 5
    folds = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=42)
    params = {
        'bagging_freq': 5,
        'boost_from_average': 'false',
        'boost': 'gbdt',
        'learning_rate': 0.01,
        'max_depth': 3,
        'metric': 'auc',
        'min_data_in_leaf': 50,
        'min_sum_hessian_in_leaf': 10.0,
        'tree_learner': 'serial',
        'objective': 'binary',
        'verbosity': 1,
        'scale_pos_weight': 20 #for unbalanced labels
    }

    importances = pd.DataFrame()
    eval_score = 0
    n_estimators = 0
    models = []
    eval_preds = np.zeros(X.shape[0])
    for fold_n, (train_index, valid_index) in enumerate(folds.split(X, y)):
        print( "\n[{}] Fold {}".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), fold_n+1))
        eval_results ={}
        # 数据
        X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
        y_train, y_valid = y[train_index], y[valid_index]
        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_valid, label=y_valid)
        # 训练
        model = lgb.train(params, train_data, num_boost_round=20000,valid_names = ['train', 'valid'],
                          valid_sets=[train_data, valid_data], evals_result = eval_results,
                          verbose_eval=200, early_stopping_rounds=200)
        # 验证结果
        print("\nRounds:", model.best_iteration)
        print("AUC: ", eval_results['valid']['auc'][model.best_iteration-1])
        n_estimators += model.best_iteration
        eval_score += eval_results['valid']['auc'][model.best_iteration-1]
        # 预测
        eval_preds[valid_index] += model.predict(X_valid, num_iteration = model.best_iteration)
        # 特征重要性
        imp_df = pd.DataFrame()
        imp_df['Feature'] = feature_names
        imp_df['split'] = model.feature_importance()
        imp_df['gain'] = model.feature_importance(importance_type='gain')
        imp_df['fold'] = fold_n + 1
        importances = pd.concat([importances, imp_df], axis=0)
        # 保存模型
        models.append(model)
    # 均值
    n_estimators = int(round(n_estimators/n_fold,0))
    eval_score = round(eval_score/n_fold,6)

    print("\nModel Report")
    print("Rounds: ", n_estimators)
    print("AUC: ", eval_score)
    return models, importances,eval_preds
def generate_adversarial_validation_set(train):
    threshod = 0.5
    train, val = train[train['predicted_probs'] < threshod], train[train['predicted_probs'] >= threshod]
    train.reset_index(drop=True,inplace=True)
    val.reset_index(drop=True,inplace=True)
    
    train = train.drop(["is_test", "predicted_probs"], 1)
    val = val.drop(["is_test", "predicted_probs"], 1)

    x_train, y_train = train.drop("tradeMoney", 1), train['tradeMoney']
    x_val, y_val = val.drop("tradeMoney", 1), val['tradeMoney']
    print("\nTrain shape: {}\nValidation shape: {}\n".format(x_train.shape, x_val.shape))
    
    return x_train, y_train, x_val, y_val

def get_training_set_with_test_set_similarity_predictions(X_train, Y_train, X_test):
    print("############# Generate adversarial validation set #############")
    data_train = X_train.copy()
    data_test = X_test.copy()
    data_train['tradeMoney'] = Y_train

    data_train['is_test'] = 0 #设置标签
    data_test['is_test'] = 1
    
    trte = pd.concat([data_train, data_test], axis=0, ignore_index=True) #合并train和test
    trte = obj2cate_1(trte) #转换object为category
    train_cols = [col for col in trte.columns if col not in ['city','tradeMoney', 'ID', 'is_test']]
    models, importances ,eval_preds= validation_prediction_lgb(trte[train_cols], trte['is_test'], train_cols)
    trte['predicted_probs'] = eval_preds
   
    print("Generating training set")
    trte = trte[trte['is_test'] == 0]
    
    print("Sorting according to predictions")
    train_set_with_predictions_for_test_set_similarity = trte.sort_values(["predicted_probs"], ascending = False)
    
    return train_set_with_predictions_for_test_set_similarity


# In[294]:


train = pd.read_csv('../data/train_data.csv')
test = pd.read_csv('../data/test_data.csv')
data = cleanData(train)
X_train = data.drop(['ID','tradeMoney'],axis=1)
X_test = test.drop(['ID'],axis=1)
Y_train = data['tradeMoney']
train_set_with_predictions_for_test_set_similarity=get_training_set_with_test_set_similarity_predictions(X_train,Y_train,X_test)
x_train, y_train, x_val, y_val = generate_adversarial_validation_set(train_set_with_predictions_for_test_set_similarity)


# - 结论：train和test数据分布不同

# In[295]:


data = pd.concat([x_train,y_train],axis=1)
figure=plt.figure(figsize=(8,6))
sns.scatterplot('area','tradeMoney',data=data)
plt.title('train not similar to test')
plt.show()


# In[296]:


data = pd.concat([x_val,y_val],axis=1)
figure=plt.figure(figsize=(8,6))
sns.scatterplot('area','tradeMoney',data=data)
plt.title('train similar to test')
plt.show()


# # 模型训练

# ### 数据处理
# - 特征变换
# ---
# 主要针对一些长尾分布的特征，需要进行幂变换或者对数变换，使得模型（LR或者DNN）能更好的优化。需要注意的是，Random Forest 和 GBDT 等模型对单调的函数变换不敏感。其原因在于树模型在求解分裂点的时候，只考虑排序分位点

# In[89]:

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from gensim.models import Word2Vec
import multiprocessing
path = '../'
save_path = path+'w2v'
all_train = pd.read_csv('../data/train_data.csv') 
all_test = pd.read_csv('../data/test_data.csv')                            
all_data = pd.concat([all_train, all_test]) 

all_data = parseData(all_data,is_test=True)

sentence = []
w2v_list = ['communityName']
for line in list(all_data[w2v_list].values):
    sentence.append([str(l) for idx, l in enumerate(line)])
print('training...')
L=10
model = Word2Vec(sentence, size=L, window=2, min_count=1, workers=multiprocessing.cpu_count(),
                 iter=10)
print('outputing...')
for fea in w2v_list:
    values = []
    for line in list(all_data[fea].values):
        values.append(line)
    values = set(values)
#         print(fea,len(values))
    # 提取每个词的词向量
    w2v = []
    for i in values:
        a = [i]
        a.extend(model[str(i)])
        w2v.append(a)
    out_df = pd.DataFrame(w2v)
    # 设置列名
    name = [fea]
    for i in range(L):
        name.append(name[0] + 'W' + str(i))
    out_df.columns = name
    out_df.to_csv(save_path + '/' + fea + '.csv', index=False)


# In[96]:


# Parse Data
def parseData(data,is_test=True):       
    # buildYear 暂无信息 处理(众数填充)
    tmp = data['buildYear'].copy()
    tmp2 = tmp[tmp!='暂无信息'].astype('int')
    tmp[tmp=='暂无信息'] = tmp2.mode()[0]
    data['buildYear'] = tmp
    data['buildYear'] = data['buildYear'].astype('int')
    
    # 缺失值处理
    data['pv'].replace(to_replace=np.nan,value=data['pv'].mean(), inplace=True)
    data['uv'].replace(to_replace=np.nan,value=data['uv'].mean(), inplace=True)

 
    # 拆分 houseType
    data[['bedroom', 'hall', 'wc', 'null']] = data['houseType'].str.split('[室厅卫]', expand=True)
    data[['bedroom', 'hall', 'wc']] = data[['bedroom', 'hall', 'wc']].astype('int64')
    data.drop(['null','houseType'], axis=1, inplace=True)
    # rentType未知方式 处理
    data.loc[(data['rentType']=='--','rentType')] = '未知方式'

    
    
    # 拆分 tradeTime 
    data['tradeMonth'] = data['tradeTime'].apply(lambda x:x.split('/')[1]).astype('int64')
    data = data.drop(['tradeTime'],axis=1)
    if is_test:
        # 去除ID列
        data.drop(['ID'], axis=1, inplace=True)
    # 去除部分特征
    data.drop(['city'], axis=1, inplace=True)
    return data
# Clean Data
def cleanData(data):    
    data.drop(data[(data['tradeMoney']>16000)].index,inplace=True) 
    data.drop(data[(data['area']>160)].index,inplace=True)
    data.drop(data[(data['tradeMoney']<800)].index,inplace=True)
    data.drop(data[(data['totalFloor']==0)].index,inplace=True)

    # 深度清理
    data.drop(data[(data['region']=='RG00001')&(data['tradeMoney']<1000)&(data['area']>50)].index,inplace=True)
    data.drop(data[(data['region']=='RG00001') & (data['tradeMoney']>25000)].index,inplace=True)
    data.drop(data[(data['region']=='RG00001') & (data['area']>250)&(data['tradeMoney']<20000)].index,inplace=True)
    data.drop(data[(data['region']=='RG00001') & (data['area']>400)&(data['tradeMoney']>50000)].index,inplace=True)
    data.drop(data[(data['region']=='RG00001') & (data['area']>100)&(data['tradeMoney']<2000)].index,inplace=True)
    data.drop(data[(data['region']=='RG00002') & (data['area']<100)&(data['tradeMoney']>60000)].index,inplace=True)
    data.drop(data[(data['region']=='RG00003') & (data['area']<300)&(data['tradeMoney']>30000)].index,inplace=True)
    data.drop(data[(data['region']=='RG00003')&(data['tradeMoney']<500)&(data['area']<50)].index,inplace=True)
    data.drop(data[(data['region']=='RG00003')&(data['tradeMoney']<1500)&(data['area']>100)].index,inplace=True)
    data.drop(data[(data['region']=='RG00003')&(data['tradeMoney']<2000)&(data['area']>300)].index,inplace=True)
    data.drop(data[(data['region']=='RG00003')&(data['tradeMoney']>5000)&(data['area']<20)].index,inplace=True)
    data.drop(data[(data['region']=='RG00003') & (data['area']>600)&(data['tradeMoney']>40000)].index,inplace=True)
    data.drop(data[(data['region']=='RG00004')&(data['tradeMoney']<1000)&(data['area']>80)].index,inplace=True)
    data.drop(data[(data['region']=='RG00006') & (data['tradeMoney']<200)].index,inplace=True)
    data.drop(data[(data['region']=='RG00005')&(data['tradeMoney']<2000)&(data['area']>180)].index,inplace=True)
    data.drop(data[(data['region']=='RG00005')&(data['tradeMoney']>50000)&(data['area']<200)].index,inplace=True)
    data.drop(data[(data['region']=='RG00006') & (data['area']>200)&(data['tradeMoney']<2000)].index,inplace=True)
    data.drop(data[(data['region']=='RG00007') & (data['area']>100)&(data['tradeMoney']<2500)].index,inplace=True)
    data.drop(data[(data['region']=='RG00010') & (data['area']>200)&(data['tradeMoney']>25000)].index,inplace=True)
    data.drop(data[(data['region']=='RG00010') & (data['area']>400)&(data['tradeMoney']<15000)].index,inplace=True)
    data.drop(data[(data['region']=='RG00010')&(data['tradeMoney']<3000)&(data['area']>200)].index,inplace=True)
    data.drop(data[(data['region']=='RG00010')&(data['tradeMoney']>7000)&(data['area']<75)].index,inplace=True)
    data.drop(data[(data['region']=='RG00010')&(data['tradeMoney']>12500)&(data['area']<100)].index,inplace=True)
    data.drop(data[(data['region']=='RG00004') & (data['area']>400)&(data['tradeMoney']>20000)].index,inplace=True)
    data.drop(data[(data['region']=='RG00008')&(data['tradeMoney']<2000)&(data['area']>80)].index,inplace=True)
    data.drop(data[(data['region']=='RG00009') & (data['tradeMoney']>40000)].index,inplace=True)
    data.drop(data[(data['region']=='RG00009') & (data['area']>300)].index,inplace=True)
    data.drop(data[(data['region']=='RG00009')&(data['area']>100)&(data['tradeMoney']<2000)].index,inplace=True)
    data.drop(data[(data['region']=='RG00011')&(data['tradeMoney']<10000)&(data['area']>390)].index,inplace=True)
    data.drop(data[(data['region']=='RG00012') & (data['area']>120)&(data['tradeMoney']<5000)].index,inplace=True)
    data.drop(data[(data['region']=='RG00013') & (data['area']<100)&(data['tradeMoney']>40000)].index,inplace=True)
    data.drop(data[(data['region']=='RG00013') & (data['area']>400)&(data['tradeMoney']>50000)].index,inplace=True)
    data.drop(data[(data['region']=='RG00013')&(data['area']>80)&(data['tradeMoney']<2000)].index,inplace=True)
    data.drop(data[(data['region']=='RG00014') & (data['area']>300)&(data['tradeMoney']>40000)].index,inplace=True)
    data.drop(data[(data['region']=='RG00014')&(data['tradeMoney']<1300)&(data['area']>80)].index,inplace=True)
    data.drop(data[(data['region']=='RG00014')&(data['tradeMoney']<8000)&(data['area']>200)].index,inplace=True)
    data.drop(data[(data['region']=='RG00014')&(data['tradeMoney']<1000)&(data['area']>20)].index,inplace=True)
    data.drop(data[(data['region']=='RG00014')&(data['tradeMoney']>25000)&(data['area']>200)].index,inplace=True)
    data.drop(data[(data['region']=='RG00014')&(data['tradeMoney']<20000)&(data['area']>250)].index,inplace=True)
    data.drop(data[(data['region']=='RG00005')&(data['tradeMoney']>30000)&(data['area']<100)].index,inplace=True)
    data.drop(data[(data['region']=='RG00005')&(data['tradeMoney']<50000)&(data['area']>600)].index,inplace=True)
    data.drop(data[(data['region']=='RG00005')&(data['tradeMoney']>50000)&(data['area']>350)].index,inplace=True)
    data.drop(data[(data['region']=='RG00006')&(data['tradeMoney']>4000)&(data['area']<100)].index,inplace=True)
    data.drop(data[(data['region']=='RG00006')&(data['tradeMoney']<600)&(data['area']>100)].index,inplace=True)
    data.drop(data[(data['region']=='RG00006')&(data['area']>165)].index,inplace=True)
    data.drop(data[(data['region']=='RG00012')&(data['tradeMoney']<800)&(data['area']<30)].index,inplace=True)
    data.drop(data[(data['region']=='RG00007')&(data['tradeMoney']<1100)&(data['area']>50)].index,inplace=True)
    data.drop(data[(data['region']=='RG00004')&(data['tradeMoney']>8000)&(data['area']<80)].index,inplace=True)
    data.loc[(data['region']=='RG00002')&(data['area']>50)&(data['rentType']=='合租'),'rentType']='整租'
    data.loc[(data['region']=='RG00014')&(data['rentType']=='合租')&(data['area']>60),'rentType']='整租'
    data.drop(data[(data['region']=='RG00008')&(data['tradeMoney']>15000)&(data['area']<110)].index,inplace=True)
    data.drop(data[(data['region']=='RG00008')&(data['tradeMoney']>20000)&(data['area']>110)].index,inplace=True)
    data.drop(data[(data['region']=='RG00008')&(data['tradeMoney']<1500)&(data['area']<50)].index,inplace=True)
    data.drop(data[(data['region']=='RG00008')&(data['rentType']=='合租')&(data['area']>50)].index,inplace=True)
    data.drop(data[(data['region']=='RG00015') ].index,inplace=True)
    data.drop(data[(data['region']=='RG00014')&(data['tradeMoney']>13000)].index,inplace=True)
    data.drop(data[(data['region']=='RG00005')&(data['tradeMoney']<1200)&(data['area']>80)].index,inplace=True)
    data.drop(data[(data['region']=='RG00005')&(data['tradeMoney']>13000)&(data['area']<90)].index,inplace=True)
    data.drop(data[(data['region']=='RG00005')&(data['tradeMoney']<1600)&(data['area']>100)].index,inplace=True)
    data.drop(data[(data['region']=='RG00002')&(data['tradeMoney']>13000)&(data['area']<100)].index,inplace=True)
    data.reset_index(drop=True, inplace=True)
    
    return data

# Make Features
def MakeFeatures_1(data):
    # 合并车站
    data['bus_sub_num'] = data['subwayStationNum']+data['busStationNum']
    # 合并学校
    data['school_num'] = data['interSchoolNum']+data['schoolNum']+data['privateSchoolNum']
    # 合并医疗
    data['help_sum'] = data['hospitalNum']+data['drugStoreNum']
    # 合并生活设施
    data['play_sum'] = data['gymNum']+data['parkNum']+data['bankNum']
    # 合并购物
    data['shop_num'] = data['shopNum']+data['mallNum']+data['superMarketNum']
    # 其他合并
    data['totalNewTradeMoney_Workers'] = data['totalNewTradeMoney'] + data['totalWorkers']
    data['bankNum_Workers'] = data['bankNum'] + data['totalWorkers']
    data['gym_bankNum'] = data['bankNum'] + data['gymNum']
    # "板块二手房价"
    data['area_mean_price'] = (data['area']*data['tradeMeanPrice'])/1000
    # "板块新房房价"
    data['New_area_mean_price'] = (data['area']*data['tradeNewMeanPrice'])/1000
    # "板块房价"
#     data['Mean_price'] = (data['tradeMeanPrice']+data['tradeNewMeanPrice'])/2 #变差
#     data['Mean_area_mean_price'] = (data['area']*data['Mean_price'])/1000 #变差
############### Hero 特征 ###################################################################
    def get_train_mode(train, data, by, fea):
        def Encode_houseFloor(x):
            if x =='低':
                return 1
            if x =='中':
                return 2
            if x =='高':
                return 3
        if fea=='area':
            train[fea] = train[fea].apply(lambda x: round(x,-1))
        if fea=='tradeMoney':
            train[fea] = train[fea].apply(lambda x: round(x,-2)) 
        if fea=='houseFloor':
            train[fea] = train[fea].apply(Encode_houseFloor)
        gp = train.groupby(by)[fea].agg(lambda x: np.mean(pd.Series.mode(x))).reset_index()     
        gp = gp.rename(columns={fea:f'trn_{"_".join(by)}_{fea}_mode'})
        data = pd.merge(data, gp, how='left', on=by)
        return data 
    def get_train_mean(train, data, by, fea):
        gp = train.groupby(by)[fea].mean().reset_index()
        gp = gp.rename(columns={fea:f'trn_{"_".join(by)}_{fea}_mean'})
        data = pd.merge(data, gp, how='left', on=by)
        return data          
    def get_train_median(train, data, by, fea):
        gp = train.groupby(by)[fea].median().reset_index()
        gp = gp.rename(columns={fea:f'trn_{"_".join(by)}_{fea}_median'})
        data = pd.merge(data, gp, how='left', on=by)
        return data 
    def get_train_std(train, data, by, fea):
        gp = train.groupby(by)[fea].std().reset_index()
        gp = gp.rename(columns={fea:f'trn_{"_".join(by)}_{fea}_std'})
        data = pd.merge(data, gp, how='left', on=by)
        data[f'trn_{"_".join(by)}_{fea}_std'] = data[f'trn_{"_".join(by)}_{fea}_std'].fillna(0)
        return data
    def get_train_max(train, data, by, fea):
        gp = train.groupby(by)[fea].max().reset_index()
        gp = gp.rename(columns={fea:f'trn_{"_".join(by)}_{fea}_max'})
        data = pd.merge(data, gp, how='left', on=by)
        return data
    def get_train_min(train, data, by, fea):
        gp = train.groupby(by)[fea].min().reset_index()
        gp = gp.rename(columns={fea:f'trn_{"_".join(by)}_{fea}_min'})
        data = pd.merge(data, gp, how='left', on=by)
        return data
    def groupby(data,by_list): 
        # 读取全量数据                        
        train = pd.read_csv('../data/train_data.csv')
        train = parseData(train)
        # 数据处理
        train.loc[(train['area']>1000),'area'] = 1000
        train['area_mean_price'] = (train['area']*train['tradeMeanPrice'])/1000  
        train['New_area_mean_price'] = (train['area']*train['tradeNewMeanPrice'])/1000
#         # "板块房价"
#         train['Mean_price'] = (train['tradeMeanPrice']+train['tradeNewMeanPrice'])/2
#         train['Mean_area_mean_price'] = (train['area']*train['Mean_price'])/1000                        
#         train['pv_uv_ratio'] = train['pv']/(train['uv']+1) 
        # 对 area_mean_price统计                       
#         data = get_train_mode(train, data, by, 'area_mean_price')
        data = get_train_mean(train, data, by, 'area_mean_price')
        data = get_train_std(train, data, by, 'area_mean_price')
#         data = get_train_median(train, data, by, 'area_mean_price')
#         # 对 New_area_mean_price统计
        data = get_train_mean(train, data, by, 'New_area_mean_price')
        #         data = get_train_std(train, data, by, 'New_area_mean_price')
        # #         data = get_train_mean(train, data, by, 'Mean_price')
#         data = get_train_std(train, data, by, 'Mean_price')                        
#         data = get_train_mean(train, data, by, 'Mean_area_mean_price')
#         data = get_train_std(train, data, by, 'Mean_area_mean_price')                        
                                
#         # 对pv_uv_ratio统计
#         data = get_train_mode(train, data, by, 'pv_uv_ratio')
#         data = get_train_mean(train, data, by, 'pv_uv_ratio')
#         data = get_train_std(train, data, by, 'pv_uv_ratio')
#         data = get_train_median(train, data, by, 'pv_uv_ratio')                        
#         # 对面积做统计特征，（有用）
        data = get_train_mode(train, data, by, 'area')
#         data = get_train_mean(train, data, by, 'area')
        data = get_train_std(train, data, by, 'area')
        data = get_train_median(train, data, by, 'area')                                                       
#     #     data = get_train_max(train, data, by, 'area') #加了过拟合
#     #     data = get_train_min(train, data, by, 'area') #加了过拟合                               
#     ############## 一把梭 ########################################################### 
#         # 众数                            
#         data = get_train_mode(train, data, by, 'tradeMonth') #加了变差
#         data = get_train_mode(train, data, by, 'houseFloor') #加了变差                           
#         data = get_train_mode(train, data, by, 'buildYear') #加了变差
#         data = get_train_mode(train, data, by, 'totalFloor')
        data = get_train_mode(train, data, by, 'pv')
#         data = get_train_mode(train, data, by, 'uv')
        data = get_train_mode(train, data, by, 'tradeMeanPrice')
        data = get_train_mode(train, data, by, 'tradeNewMeanPrice')                               
#         data = get_train_mode(train, data, by, 'totalTradeMoney')
        data = get_train_mode(train, data, by, 'totalNewTradeMoney')
        data = get_train_mode(train, data, by, 'saleSecHouseNum')
        data = get_train_mode(train, data, by, 'totalTradeArea')
#         data = get_train_mode(train, data, by, 'tradeSecNum')
#         data = get_train_mode(train, data, by, 'totalNewTradeArea') 
#         data = get_train_mode(train, data, by, 'tradeNewNum')
        data = get_train_mode(train, data, by, 'remainNewNum')
        data = get_train_mode(train, data, by, 'supplyNewNum')                           
        data = get_train_mode(train, data, by, 'newWorkers')  
        data = get_train_mode(train, data, by, 'bedroom')
#         data = get_train_mode(train, data, by, 'totalWorkers')                            
#         # 均值                            
#         data = get_train_mean(train, data, by, 'tradeMonth') #加了变差
#         data = get_train_mean(train, data, by, 'houseFloor') #加了变差                           
#         data = get_train_mean(train, data, by, 'buildYear') #加了变差
#         data = get_train_mean(train, data, by, 'totalFloor')
#         data = get_train_mean(train, data, by, 'pv')
#         data = get_train_mean(train, data, by, 'uv')
#         data = get_train_mean(train, data, by, 'tradeMeanPrice')
#         data = get_train_mean(train, data, by, 'tradeNewMeanPrice')                            
#         data = get_train_mean(train, data, by, 'totalTradeMoney')
#         data = get_train_mean(train, data, by, 'totalNewTradeMoney')
        data = get_train_mean(train, data, by, 'saleSecHouseNum')
#         data = get_train_mean(train, data, by, 'totalTradeArea')
#         data = get_train_mean(train, data, by, 'tradeSecNum')
#         data = get_train_mean(train, data, by, 'totalNewTradeArea') 
#         data = get_train_mean(train, data, by, 'tradeNewNum')
#         data = get_train_mean(train, data, by, 'remainNewNum')
        data = get_train_mean(train, data, by, 'supplyNewNum')
        data = get_train_mean(train, data, by, 'newWorkers')
#         data = get_train_mean(train, data, by, 'bedroom')
#         data = get_train_mean(train, data, by, 'totalWorkers') # 特征选择删除了    
#         # 方差
#         data = get_train_std(train, data, by, 'tradeMonth') #加了变差
#         data = get_train_std(train, data, by, 'houseFloor') #加了变差                           
#         data = get_train_std(train, data, by, 'buildYear') #加了变差                            
        data = get_train_std(train, data, by, 'totalFloor')
        data = get_train_std(train, data, by, 'pv')
#         data = get_train_std(train, data, by, 'uv')
        data = get_train_std(train, data, by, 'tradeMeanPrice')
        data = get_train_std(train, data, by, 'tradeNewMeanPrice')                               
        data = get_train_std(train, data, by, 'totalTradeMoney')
        data = get_train_std(train, data, by, 'totalNewTradeMoney')
#         data = get_train_std(train, data, by, 'saleSecHouseNum')
        data = get_train_std(train, data, by, 'totalTradeArea')
        data = get_train_std(train, data, by, 'tradeSecNum')
#         data = get_train_std(train, data, by, 'totalNewTradeArea') 
#         data = get_train_std(train, data, by, 'tradeNewNum')
        data = get_train_std(train, data, by, 'remainNewNum')
        data = get_train_std(train, data, by, 'supplyNewNum')
#         data = get_train_std(train, data, by, 'newWorkers')  
        data = get_train_std(train, data, by, 'bedroom')                                
#         data = get_train_std(train, data, by, 'totalWorkers') # 特征选择删除了
#         #中位数
#         data = get_train_median(train, data, by, 'tradeMonth') #加了变差
#         data = get_train_median(train, data, by, 'houseFloor') #加了变差                           
#         data = get_train_median(train, data, by, 'buildYear') #加了变差                              
#         data = get_train_median(train, data, by, 'totalFloor')
#         data = get_train_median(train, data, by, 'pv')
#         data = get_train_median(train, data, by, 'uv')
#         data = get_train_median(train, data, by, 'tradeMeanPrice')
#         data = get_train_median(train, data, by, 'tradeNewMeanPrice')                               
        data = get_train_median(train, data, by, 'totalTradeMoney')
        data = get_train_median(train, data, by, 'totalNewTradeMoney')
#         data = get_train_median(train, data, by, 'saleSecHouseNum') # 特征选择删除了
#         data = get_train_median(train, data, by, 'totalTradeArea')
#         data = get_train_median(train, data, by, 'tradeSecNum')
#         data = get_train_median(train, data, by, 'totalNewTradeArea') 
#         data = get_train_median(train, data, by, 'tradeNewNum')
#         data = get_train_median(train, data, by, 'remainNewNum')
#         data = get_train_median(train, data, by, 'supplyNewNum') # 特征选择删除了
#         data = get_train_median(train, data, by, 'newWorkers')  
#         data = get_train_median(train, data, by, 'bedroom')  # 特征选择删除了
#         data = get_train_median(train, data, by, 'totalWorkers') # 特征选择删除了
        return data                        
####################################################################################                                
    by=['communityName']
    data = groupby(data, by)
    by=['plate']
    data = groupby(data, by)                           
################## 平滑处理 ################################################################                                
    big_num_cols = ['totalTradeMoney','totalTradeArea','tradeMeanPrice','totalNewTradeMoney', 'totalNewTradeArea',
                'tradeNewMeanPrice','remainNewNum', 'supplyNewNum', 'supplyLandArea',
                'tradeLandArea','landTotalPrice','landMeanPrice','totalWorkers','newWorkers',
                'residentPopulation','pv','uv',
                   ]
    num_cols = ['area','subwayStationNum', 'busStationNum', 'interSchoolNum', 'schoolNum',
                'privateSchoolNum', 'hospitalNum', 'drugStoreNum', 'gymNum', 'bankNum', 'shopNum',
                'parkNum', 'mallNum', 'superMarketNum', 'totalTradeMoney', 'totalTradeArea', 'tradeMeanPrice',
                'tradeSecNum', 'totalNewTradeMoney', 'totalNewTradeArea', 'tradeNewMeanPrice', 'tradeNewNum',
                'remainNewNum', 'supplyNewNum', 'supplyLandNum', 'supplyLandArea', 'tradeLandNum',
                'tradeLandArea', 'landTotalPrice', 'landMeanPrice', 'totalWorkers', 'newWorkers',
                'residentPopulation', 'pv', 'uv', 'lookNum'] 
    for col in num_cols:
        high = np.percentile(data[col].values, 99.8)
        low = np.percentile(data[col].values, 0.2)
        data.loc[data[col] > high, col] = high
        data.loc[data[col] < low, col] = low
    # 过大量级值取log平滑
    for col in big_num_cols:
        data[col] = data[col].map(lambda x: np.log1p(x))                            

##################################################################################                              
    return data
def w2v(data):
    path = '../'
    w2v_path = path+'w2v'
    w2v_features = []
    for col in ['communityName']:
        df = pd.read_csv(w2v_path + '/' + col + '.csv')
        fs = list(df)
        fs.remove(col)
        w2v_features += fs
        data = pd.merge(data, df, on=col, how='left')
    print('word2vec:')
    print(w2v_features)                            
    return data
                                
def remove(data):               
    # 删除特征
    remove_col = [ 
        'totalNewTradeMoney', 
        'help_sum', 
        'tradeNewNum', 
        'supplyLandArea', 
        'totalTradeMoney',
        'landTotalPrice', 
        'uv', 
        'shop_num',
        'shopNum', 
        'schoolNum', 
        'totalNewTradeArea', 
        'region', 
        'landMeanPrice', 
        'totalTradeArea', 
        'superMarketNum', 
        'tradeLandArea',
        'rentType', 
        'gym_bankNum', 
        'communityName', 
        'play_sum',
        'bus_sub_num',
        'area_mean_price',
        'New_area_mean_price',
#         'Mean_area_mean_price',
#         'Mean_price',
                 ]   
    data = data.drop(remove_col,axis=1) 
    return data
# 删除特征值单一的特征 
def overfit(train,test):
    overfit = []
    for i in train.columns:
        if i not in ['tradeMoney']:
            counts = train[i].value_counts()
            values = counts.iloc[0]
            if values / len(train) * 100 > 99:
                overfit.append(i)
    train.drop(overfit, axis=1,inplace=True)
    test.drop(overfit, axis=1,inplace=True)
    print('特征值单一的特征：', overfit)
    return train,test
# category 处理
def obj2cat(data):
    # 转换object类型数据
#     ['rentType', 'houseFloor', 'houseToward', 'houseDecoration', 'communityName', 'region', 'plate']
    columns = ['rentType', 'houseFloor', 'houseToward', 'houseDecoration', 'communityName','region', 'plate']
    for col in columns:
        data[col] = data[col].astype('category')
    return data
def cat2obj(data):
    # 转换数据
    columns = [ 'buildYear', 'houseType', 'tradeTime', 'rentType', 'houseFloor', 'houseToward', 'houseDecoration', 'communityName', 'region', 'plate']
    for col in columns:
        data[col] = data[col].astype('object')
    return data

# 数据预处理
def getData(is_test=True,x_train=None,x_val=None,y_train=None):
    ### Load data ###
    if is_test:
        train = pd.read_csv('../data/train_data.csv')
        test = pd.read_csv('../data/test_data.csv')
    else:
        train = pd.concat([x_train,y_train],axis=1)
        test = x_val.copy()
        train = cat2obj(train)
        test = cat2obj(test)
        
        
        
    print('### Before data preprocessing:')
    print("Train: ", train.shape[0], "samples, and ", train.shape[1], "cols")
    print("Test: ", test.shape[0], "samples, and ", test.shape[1], "cols")

    ### Data processing ###
    train = parseData(train,is_test)
    test = parseData(test,is_test)

    train = cleanData(train)

    train = MakeFeatures_1(train)
    test = MakeFeatures_1(test)
                        
    train = w2v(train)
    test = w2v(test)                            
                       
    train = obj2cat(train)
    test = obj2cat(test)
#     train,test = overfit(train,test)
    train = remove(train)
    test = remove(test)
    print('### After data preprocessing:')
    X = train.drop(['tradeMoney'],axis=1)
    y = train['tradeMoney'].copy()
    
    X_test = test.copy()

    print("Train: ", X.shape[0], "samples, and ", X.shape[1], "features")
    print("Test: ", X_test.shape[0], "samples, and ", X_test.shape[1], "features")
    print('Features:', X.columns)
    return X,y,X_test


# In[548]:


X_train,Y_train,X_val = getData(is_test=False,x_train=x_train,x_val=x_val,y_train=y_train)


# In[540]:


data = pd.concat([X_train,Y_train],axis=1)
figure=plt.figure(figsize=(8,6))
sns.scatterplot('trn_communityName_rentType_area_mean','tradeMoney',data=data)
plt.show()


# In[480]:


# sns.distplot(data['area_mean_price'],hist=True)


# In[549]:


corrmat = X_train.corr()
plt.subplots(figsize=(20,30))
sns.heatmap(corrmat, vmax=0.9, square=True)
plt.show()


# ### 对抗验证（验证模型）
# + 为了保证线下和线上得分接近

# In[448]:


# LightGBM Model
def run_lgb(x_train, y_train, x_val, y_val):
    print("############# Build LightGBM model #############")
    params = {
    'num_leaves': 31,
    'min_data_in_leaf': 20,
    'min_child_samples':20,
    'objective': 'regression',
    'learning_rate': 0.01,
    "boosting": "gbdt",
    "feature_fraction": 0.8,
    "bagging_freq": 1,
    "bagging_fraction": 0.85,
    "bagging_seed": 23,
    "metric": 'rmse',
    "lambda_l1": 0.2,
    "nthread": 4,
#     'random_state':42,
    }
    
    print("Validating.....")
    trn_data = lgb.Dataset(x_train, label=y_train)
    val_data = lgb.Dataset(x_val, label=y_val)
    num_round = 10000
    clf = lgb.train(params, trn_data, num_round,valid_sets=[trn_data, val_data], verbose_eval=200,early_stopping_rounds=200)
    y_pred_val = clf.predict(x_val, num_iteration=clf.best_iteration)
    print('Val r2 score:{:.6f}'.format(r2_score(y_pred_val, y_val)))
    return y_pred_val


# In[550]:


# 验证和训练
y_pred_test = run_lgb(X_train, Y_train, X_val, y_val)


# In[ ]:


from MLFeatureSelection import sequence_selection, importance_selection, coherence_selection,tools
def lossfunction(y_test,y_pred):
    """define your own loss function with y_pred and y_test
    return score
    """
    return r2_score(y_test,y_pred)
def validate(X, y, features, clf, lossfunction):
    """define your own validation function with 5 parameters
    input as X, y, features, clf, lossfunction
    clf is set by SetClassifier()
    lossfunction is import earlier
    features will be generate automatically
    function return score and trained classfier
    """
    
    #对抗验证
    x_train = X[X['is_train']==1].drop(['tradeMoney'],axis=1)
    x_val = X[X['is_train']==0].drop(['tradeMoney'],axis=1)
    x_train.drop(['is_train'],axis=1,inplace=True)
    x_val.drop(['is_train'],axis=1,inplace=True)
    y_train = X[X['is_train']==1]['tradeMoney']
    y_val = X[X['is_train']==0]['tradeMoney']
    x_train = x_train[features].copy()
    x_val = x_val[features].copy()
    clf.fit(x_train, y_train,eval_names=['train','valid'],eval_metric='r2_score',eval_set=[(x_train,y_train),(x_val,y_val)],
             early_stopping_rounds=200,verbose=200)
    
    y_pred_val = clf.predict(x_val[features])
    score = lossfunction(y_val, y_pred_val)
    '''
    # K-Flod CV
    X = X[features].copy()
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
    out_of_fold = np.zeros(len(X))

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        # Train
        clf.fit(X_train, y_train)
        out_of_fold[val_idx] = clf.predict(X_val)
    score = lossfunction(out_of_fold, y)
    '''
    return score, clf
def imp(df,f,estimator):
    sf = importance_selection.Select(_FeaturesLimit=30) #initialized selector
    sf.ImportDF(df,label = 'tradeMoney') #import dataset
    sf.ImportLossFunction(lossfunction, direction = 'ascend')  #import loosfunction and improve direction
    sf.InitialFeatures(f)  #define list initial features combination
    sf.SelectRemoveMode(batch = 1) #define remove features quantity each iteration
    sf.clf = estimator #define selected estimator
    sf.SetLogFile('record_imp.log') #set the log file name
    return sf.run(validate) #start running
def coh(df,f,estimator):
    sf = coherence_selection.Select() #initialized selector
    sf.ImportDF(df,label = 'tradeMoney') #import dataset
    sf.ImportLossFunction(lossfunction, direction = 'ascend') #import loosfunction and improve direction 
    sf.InitialFeatures(f) #define list initial features combination
    sf.SelectRemoveMode(batch=1, lowerbound = 0.5) #define remove features quantity each iteration and selection threshold
    sf.clf = estimator #define selected estimator
    sf.SetLogFile('record_coh.log') #set the log file name
    return sf.run(validate) #start running
def main():
    from lightgbm import LGBMRegressor
    from sklearn.preprocessing import LabelEncoder
    temp_train =pd.concat([X_train,Y_train],axis=1) # read dataset
    temp_val =pd.concat([X_val,y_val],axis=1)
    temp_train['is_train'] = 1
    temp_val['is_train'] = 0
    df =pd.concat([temp_train,temp_val],axis=0)
    columns = ['rentType', 'houseFloor', 'houseToward', 'houseDecoration', 'communityName','region', 'plate']
    for col in columns:
        df[col] = LabelEncoder().fit_transform(df[col])
    params = {
    'num_leaves': 31,
    'min_data_in_leaf': 20,
    'min_child_samples':20,
    'objective': 'regression',
    'learning_rate': 0.01,
    "boosting": "gbdt",
    "feature_fraction": 0.8,
    "bagging_freq": 1,
    "bagging_fraction": 0.85,
    "bagging_seed": 23,
    "metric": 'rmse',
    "lambda_l1": 0.2,
    "nthread": 4,
#     'random_state':42,
    'n_estimators':10000,
    }
    clf = LGBMRegressor(**params)
    uf = X_train.columns.tolist()
    print('importance selection')
    uf = imp(df,uf,clf)
#     print('coherence selection')
#     uf = coh(df,uf,clf)
if __name__ == "__main__":
    main()


# In[ ]:


col = ['area', 'bankNum', 'buildYear', 'busStationNum', 'communityName', 
       'drugStoreNum', 'gymNum', 'hospitalNum', 'houseDecoration', 'houseFloor', 
       'houseToward', 'interSchoolNum', 'landMeanPrice', 'landTotalPrice', 'lookNum', 
       'mallNum', 'newWorkers', 'parkNum', 'plate', 'privateSchoolNum', 'pv', 'remainNewNum', 
       'rentType', 'residentPopulation', 'saleSecHouseNum', 'schoolNum', 'shopNum', 'subwayStationNum', 
       'superMarketNum', 'supplyLandArea', 'supplyNewNum', 'totalFloor', 'totalNewTradeArea', 
       'totalNewTradeMoney', 'totalTradeArea', 'totalTradeMoney', 'totalWorkers', 'tradeLandArea', 
       'tradeLandNum', 'tradeMeanPrice', 'tradeNewMeanPrice', 'tradeNewNum', 'tradeSecNum', 'uv', 
       'bedroom', 'hall', 'wc', 'tradeMonth', 'bus_sub_num', 'school_num', 'help_sum', 'play_sum',
       'shop_num', 'totalNewTradeMoney_Workers', 'bankNum_Workers', 'gym_bankNum', 'tradeFromBuildYear',
       'trn_communityName_area_mode', 'trn_communityName_area_mean', 'trn_communityName_area_std', 
       'trn_communityName_area_median', 'trn_communityName_totalFloor_mean', 'trn_communityName_pv_mean', 
       'trn_communityName_uv_mean', 'trn_communityName_tradeMeanPrice_mean', 'trn_communityName_tradeNewMeanPrice_mean',
       'trn_communityName_bedroom_mode', 'trn_communityName_totalTradeMoney_mean', 'trn_communityName_totalNewTradeMoney_mean',
       'trn_communityName_saleSecHouseNum_mean', 'trn_communityName_totalTradeArea_mean', 
       'trn_communityName_tradeSecNum_mean', 'trn_communityName_totalNewTradeArea_mean', 
       'trn_communityName_tradeNewNum_mean', 'trn_communityName_remainNewNum_mean',
       'trn_communityName_supplyNewNum_mean', 'trn_communityName_newWorkers_mean', 
       'trn_communityName_totalFloor_std', 'trn_communityName_pv_std', 'trn_communityName_uv_std',
       'trn_communityName_tradeMeanPrice_std', 'trn_communityName_tradeNewMeanPrice_std',
       'trn_communityName_bedroom_mean', 'trn_communityName_bedroom_std', 'trn_communityName_totalTradeMoney_std', 
       'trn_communityName_totalNewTradeMoney_std', 'trn_communityName_saleSecHouseNum_std', 'trn_communityName_totalTradeArea_std',
       'trn_communityName_tradeSecNum_std', 'trn_communityName_totalNewTradeArea_std', 
       'trn_communityName_tradeNewNum_std', 'trn_communityName_remainNewNum_std', 'trn_communityName_supplyNewNum_std',
       'trn_communityName_newWorkers_std']
print('删除了的特征:')
print(set(X_train.columns).difference(set(col)))


# In[218]:


col = ['area', 'bankNum', 'buildYear', 'busStationNum', 'communityName', 
       'drugStoreNum', 'gymNum', 'hospitalNum', 'houseDecoration', 'houseFloor', 
       'houseToward', 'interSchoolNum', 'landMeanPrice', 'landTotalPrice', 'lookNum', 
       'mallNum', 'newWorkers', 'parkNum', 'plate', 'privateSchoolNum', 'pv', 'remainNewNum', 
       'rentType', 'residentPopulation', 'saleSecHouseNum', 'schoolNum', 'shopNum', 'subwayStationNum', 
       'superMarketNum', 'supplyLandArea', 'supplyNewNum', 'totalFloor', 'totalNewTradeArea', 
       'totalNewTradeMoney', 'totalTradeArea', 'totalTradeMoney', 'totalWorkers', 'tradeLandArea', 
       'tradeLandNum', 'tradeMeanPrice', 'tradeNewMeanPrice', 'tradeNewNum', 'tradeSecNum', 'uv', 
       'bedroom', 'hall', 'wc', 'tradeMonth', 'bus_sub_num', 'school_num', 'help_sum', 'play_sum',
       'shop_num', 'totalNewTradeMoney_Workers', 'bankNum_Workers', 'gym_bankNum', 'tradeFromBuildYear',
       'trn_communityName_area_mode', 'trn_communityName_area_mean', 'trn_communityName_area_std', 
       'trn_communityName_area_median', 'trn_communityName_totalFloor_mean', 'trn_communityName_pv_mean', 
       'trn_communityName_uv_mean', 'trn_communityName_tradeMeanPrice_mean', 'trn_communityName_tradeNewMeanPrice_mean',
       'trn_communityName_bedroom_mode', 'trn_communityName_totalTradeMoney_mean', 'trn_communityName_totalNewTradeMoney_mean',
       'trn_communityName_saleSecHouseNum_mean', 'trn_communityName_totalTradeArea_mean', 
       'trn_communityName_tradeSecNum_mean', 'trn_communityName_totalNewTradeArea_mean', 
       'trn_communityName_tradeNewNum_mean', 'trn_communityName_remainNewNum_mean',
       'trn_communityName_supplyNewNum_mean', 'trn_communityName_newWorkers_mean', 
       'region', 
       'supplyLandNum', 
#        'trn_communityName_totalWorkers_std',
       'trn_communityName_totalFloor_std', 'trn_communityName_pv_std', 'trn_communityName_uv_std',
       'trn_communityName_tradeMeanPrice_std', 'trn_communityName_tradeNewMeanPrice_std',
       'trn_communityName_bedroom_mean', 'trn_communityName_bedroom_std', 'trn_communityName_totalTradeMoney_std', 
       'trn_communityName_totalNewTradeMoney_std', 'trn_communityName_saleSecHouseNum_std', 'trn_communityName_totalTradeArea_std',
       'trn_communityName_tradeSecNum_std', 'trn_communityName_totalNewTradeArea_std', 
       'trn_communityName_tradeNewNum_std', 'trn_communityName_remainNewNum_std', 'trn_communityName_supplyNewNum_std',
       'trn_communityName_newWorkers_std']

# 验证和训练
y_pred_test = run_lgb(X_train[col], Y_train, X_val[col], y_val)


# 
#  - 0.890556 area
#  - 0.925722 area+Money(mean)(过拟合)
#  - 0.927755 area+Money(mean+median)(过拟合)
#  - 0.891311 area+totalFloor
#  - 0.891932 area+totalFloor+pv+uv
#  - 0.892682 area+totalFloor+pv+uv(area median)
#  - 0.893827 area+totalFloor+pv+uv+tradeMeanPrice+tradeNewMeanPrice
#  - 0.894212 area+totalFloor+pv+uv+tradeMeanPrice+tradeNewMeanPrice(area median+mean+min+max+std+mod)(过拟合)
#  - 0.893796 area+totalFloor+pv+uv+tradeMeanPrice+tradeNewMeanPrice(area median+mean+std+mod)
#  - 0.894341 area+totalFloor+pv+uv+tradeMeanPrice+tradeNewMeanPrice(area median+mean+std+mod)+平滑
#  - 0.894726 area+totalFloor+pv+uv+tradeMeanPrice+tradeNewMeanPrice(area median+mean+std+mod)+平滑+totalTradeMoney
#  - 0.894126 area+totalFloor+pv+uv+tradeMeanPrice+tradeNewMeanPrice(area median+mean+std+mod)+平滑+totalTradeMoney(删除了help_sum、shop_num、trn_communityName_totalWorkers_mean特征)
#  - 0.895166 area+totalFloor+pv+uv+tradeMeanPrice+tradeNewMeanPrice(area median+mean+std+mod)+平滑+totalTradeMoney（特征选择）
#  - 0.895996 area+totalFloor+pv+uv+tradeMeanPrice+tradeNewMeanPrice(area median+mean+std+mod)+平滑+totalTradeMoney（一把梭特征+特征选择）
#  - 0.897116 area+totalFloor+pv+uv+tradeMeanPrice+tradeNewMeanPrice(area median+mean+std+mod)+平滑+totalTradeMoney（一把梭特征）
#  - 0.893841 area+totalFloor+pv+uv+tradeMeanPrice+tradeNewMeanPrice(area median+mean+std+mod)+平滑+totalTradeMoney（一把梭特征）+tradeMoney>800,tradeMoney<16000
#  - 0.894219 area+totalFloor+pv+uv+tradeMeanPrice+tradeNewMeanPrice(area median+mean+std+mod)+平滑+totalTradeMoney（一把梭特征）+tradeMoney>800,tradeMoney<16000(一把梭median，mode)
#  - 0.892106 area+totalFloor+pv+uv+tradeMeanPrice+tradeNewMeanPrice(area median+mean+std+mod)+平滑+totalTradeMoney（一把梭特征）+tradeMoney>800,tradeMoney<16000(一把梭median，mode)+area_mean_price
# 

# In[269]:


data = pd.concat([X_val,y_val],axis=1)
figure=plt.figure(figsize=(8,6))
sns.scatterplot('area','tradeMoney',data=data)
plt.show()


# ---

# In[270]:


data = pd.concat([X_val,y_val],axis=1)
error_index = y_val[np.abs(y_pred_test-y_val)<=100].index
data = data.iloc[error_index]
figure=plt.figure(figsize=(8,6))
sns.scatterplot('area','tradeMoney',data=data)
plt.title('误差小')
plt.show()


# In[271]:


data = pd.concat([X_val,y_val],axis=1)
error_index = y_val[np.abs(y_pred_test-y_val)>2000].index
data = data.iloc[error_index]
figure=plt.figure(figsize=(8,6))
sns.scatterplot('area','tradeMoney',data=data)
plt.title('误差大')
plt.show()


# In[483]:


# submission = pd.DataFrame({'pred':np.trunc(predictions)})
# submission.to_csv('submit_baseline_normal.csv', index = False, header=False,encoding='utf-8')


# ### 10折CV（训练模型）

# In[97]:


X_train,Y_train,X_test = getData(is_test=True)


# In[24]:


X_train.columns.tolist()


# In[67]:


# from MLFeatureSelection import sequence_selection, importance_selection, coherence_selection,tools
# def lossfunction(y_test,y_pred):
#     """define your own loss function with y_pred and y_test
#     return score
#     """
#     return r2_score(y_test,y_pred)
# def validate(X, y, features, clf, lossfunction):
#     """define your own validation function with 5 parameters
#     input as X, y, features, clf, lossfunction
#     clf is set by SetClassifier()
#     lossfunction is import earlier
#     features will be generate automatically
#     function return score and trained classfier
#     """
    
#     '''对抗验证
#     x_train = X[X['is_train']==1].drop(['tradeMoney'],axis=1)
#     x_val = X[X['is_train']==0].drop(['tradeMoney'],axis=1)
#     x_train.drop(['is_train'],axis=1,inplace=True)
#     x_val.drop(['is_train'],axis=1,inplace=True)
#     y_train = X[X['is_train']==1]['tradeMoney']
#     y_val = X[X['is_train']==0]['tradeMoney']
#     x_train = x_train[features].copy()
#     x_val = x_val[features].copy()
#     clf.fit(x_train, y_train,eval_names=['train','valid'],eval_metric='r2_score',eval_set=[(x_train,y_train),(x_val,y_val)],
#              early_stopping_rounds=200,verbose=200)
    
#     y_pred_val = clf.predict(x_val[features])
#     score = lossfunction(y_val, y_pred_val)
#     '''
#     # K-Flod CV
#     X = X[features].copy()
#     skf = KFold(n_splits=3, shuffle=True, random_state=0)
#     out_of_fold = np.zeros(len(X))

#     for fold, (train_idx, val_idx) in enumerate(skf.split(X)):
#         X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
#         y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
#         # Train
#         clf.fit(X_train, y_train)
#         out_of_fold[val_idx] = clf.predict(X_val)
#     score = lossfunction(out_of_fold, y)
#     return score, clf
# def imp(df,f,estimator):
#     sf = importance_selection.Select() #initialized selector
#     sf.ImportDF(df,label = 'tradeMoney') #import dataset
#     sf.ImportLossFunction(lossfunction, direction = 'ascend')  #import loosfunction and improve direction
#     sf.InitialFeatures(f)  #define list initial features combination
#     sf.SelectRemoveMode(batch = 5) #define remove features quantity each iteration
#     sf.clf = estimator #define selected estimator
#     sf.SetLogFile('record_imp.log') #set the log file name
#     return sf.run(validate) #start running
# def coh(df,f,estimator):
#     sf = coherence_selection.Select() #initialized selector
#     sf.ImportDF(df,label = 'tradeMoney') #import dataset
#     sf.ImportLossFunction(lossfunction, direction = 'ascend') #import loosfunction and improve direction 
#     sf.InitialFeatures(f) #define list initial features combination
#     sf.SelectRemoveMode(batch=5, lowerbound = 0.8) #define remove features quantity each iteration and selection threshold
#     sf.clf = estimator #define selected estimator
#     sf.SetLogFile('record_coh.log') #set the log file name
#     return sf.run(validate) #start running
# def main():
#     from lightgbm import LGBMRegressor
#     from sklearn.preprocessing import LabelEncoder
#     df =pd.concat([X_train,Y_train],axis=1)
# #     columns = [ 'houseFloor', 'houseToward', 'houseDecoration', 'communityName','region', 'plate']
# #     for col in columns:
# #         df[col] = LabelEncoder().fit_transform(df[col])
#     params = {
#     'num_leaves': 31,
#     'min_data_in_leaf': 20,
#     'min_child_samples':20,
#     'objective': 'regression',
#     'learning_rate': 0.01,
#     "boosting": "gbdt",
#     "feature_fraction": 0.8,
#     "bagging_freq": 1,
#     "bagging_fraction": 0.85,
#     "bagging_seed": 23,
#     "metric": 'rmse',
#     "lambda_l1": 0.2,
#     "nthread": 4,
# #     'random_state':42,
#     'n_estimators':10000,
#     }
# #     clf = LGBMRegressor(**params)
#     clf = LGBMRegressor()
#     uf = X_train.columns.tolist()
# #     print('importance selection')
# #     uf = imp(df,uf,clf)
#     print('coherence selection')
#     uf = coh(df,uf,clf)
# if __name__ == "__main__":
#     main()


# In[68]:


# # 特征选择
# col = ['area', 'houseFloor', 'totalFloor', 'houseToward', 'houseDecoration', 'plate', 'buildYear', 'saleSecHouseNum', 'subwayStationNum', 'busStationNum', 'interSchoolNum', 'privateSchoolNum', 'hospitalNum', 'gymNum', 'bankNum', 'parkNum', 'mallNum', 'tradeMeanPrice', 'tradeSecNum', 'tradeNewMeanPrice', 'remainNewNum', 'supplyNewNum', 'supplyLandNum', 'tradeLandNum', 'totalWorkers', 'newWorkers', 'residentPopulation', 'pv', 'lookNum', 'bedroom', 'hall', 'wc', 'tradeMonth', 'school_num', 'totalNewTradeMoney_Workers', 'bankNum_Workers', 'trn_communityName_area_mean_price_mean', 'trn_communityName_area_mean_price_std', 'trn_communityName_Mean_price_mean', 'trn_communityName_Mean_area_mean_price_mean', 'trn_communityName_Mean_area_mean_price_std', 'trn_communityName_area_mode', 'trn_communityName_area_std', 'trn_communityName_area_median', 'trn_communityName_pv_mode', 'trn_communityName_tradeMeanPrice_mode', 'trn_communityName_totalNewTradeMoney_mode', 'trn_communityName_saleSecHouseNum_mode', 'trn_communityName_totalTradeArea_mode', 'trn_communityName_supplyNewNum_mode', 'trn_communityName_newWorkers_mode', 'trn_communityName_bedroom_mode', 'trn_communityName_saleSecHouseNum_mean', 'trn_communityName_supplyNewNum_mean', 'trn_communityName_newWorkers_mean', 'trn_communityName_totalFloor_std', 'trn_communityName_pv_std', 'trn_communityName_tradeMeanPrice_std', 'trn_communityName_tradeNewMeanPrice_std', 'trn_communityName_totalTradeMoney_std', 'trn_communityName_totalNewTradeMoney_std', 'trn_communityName_totalTradeArea_std', 'trn_communityName_tradeSecNum_std', 'trn_communityName_remainNewNum_std', 'trn_communityName_supplyNewNum_std', 'trn_communityName_bedroom_std', 'trn_communityName_totalTradeMoney_median', 'trn_communityName_totalNewTradeMoney_median', 'trn_plate_area_mean_price_std', 'trn_plate_New_area_mean_price_mean', 'trn_plate_New_area_mean_price_std', 'trn_plate_area_mode', 'trn_plate_area_std', 'trn_plate_area_median', 'trn_plate_tradeNewMeanPrice_mode', 'trn_plate_totalNewTradeMoney_mode', 'trn_plate_saleSecHouseNum_mode', 'trn_plate_remainNewNum_mode', 'trn_plate_supplyNewNum_mode', 'trn_plate_newWorkers_mode', 'trn_plate_bedroom_mode', 'trn_plate_saleSecHouseNum_mean', 'trn_plate_supplyNewNum_mean', 'trn_plate_newWorkers_mean', 'trn_plate_totalFloor_std', 'trn_plate_pv_std', 'trn_plate_totalTradeMoney_std', 'trn_plate_totalNewTradeMoney_std', 'trn_plate_tradeSecNum_std', 'trn_plate_bedroom_std', 'trn_plate_totalNewTradeMoney_median', 'communityNameW0', 'communityNameW1', 'communityNameW2', 'communityNameW3', 'communityNameW4', 'communityNameW5', 'communityNameW6', 'communityNameW7', 'communityNameW8', 'communityNameW9']
# print('删除了的特征:')
# print(set(X_train.columns).difference(set(col)))


# In[69]:


# X_train = X_train[col]
# X_test = X_test[col]


# In[98]:


##### K-Fold CV #####
# 10-Fold CV
kf = KFold(n_splits=5, shuffle=True, random_state=0)
out_of_fold = np.zeros(len(X_train))
predictions = np.zeros(len(X_test))
feature_importance_df = pd.DataFrame()
params = {
    'num_leaves': 31,
    'min_data_in_leaf': 20,
    'min_child_samples':20,
    'objective': 'regression',
    'learning_rate': 0.01,
    "boosting": "gbdt",
    "feature_fraction": 0.8,
    "bagging_freq": 1,
    "bagging_fraction": 0.85,
    "bagging_seed": 23,
    "metric": 'rmse',
    "lambda_l1": 0.2,
    "nthread": 4,
}
for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    print( "\n[{}] Fold {}".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), fold+1))
    trn_data = lgb.Dataset(X_train.iloc[train_idx], label=Y_train[train_idx])
    val_data = lgb.Dataset(X_train.iloc[val_idx], label=Y_train[val_idx])
    # Train
    num_round = 10000
    clf = lgb.train(params, trn_data, num_round,valid_sets=[trn_data, val_data], verbose_eval=200,early_stopping_rounds=200)
    out_of_fold[val_idx] = clf.predict(X_train.iloc[val_idx], num_iteration=clf.best_iteration)

    
    # Predict test data
    predictions += clf.predict(X_test, num_iteration=clf.best_iteration) / kf.n_splits

    # Feature Importance
    fold_importance_df = pd.DataFrame()
    fold_importance_df["Feature"] = X_train.columns
    fold_importance_df["importance"] = clf.feature_importance()
    fold_importance_df["fold"] = fold + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
#     trn_data = lgb.Dataset(X_train.iloc[train_idx], label=Y_train[train_idx]/X_train.iloc[train_idx]['area'])
#     val_data = lgb.Dataset(X_train.iloc[val_idx], label=Y_train[val_idx]/X_train.iloc[val_idx]['area'])

#     # Train
#     num_round = 10000
#     clf = lgb.train(params, trn_data, num_round,valid_sets=[trn_data, val_data], verbose_eval=200,early_stopping_rounds=200)
#     out_of_fold[val_idx] = (clf.predict(X_train.iloc[val_idx], num_iteration=clf.best_iteration))*X_train.iloc[val_idx]['area']

    
#     # Predict test data
#     predictions += (clf.predict(X_test, num_iteration=clf.best_iteration) / kf.n_splits)*X_test['area']

#     # Feature Importance
#     fold_importance_df = pd.DataFrame()
#     fold_importance_df["Feature"] = X_train.columns
#     fold_importance_df["importance"] = clf.feature_importance()
#     fold_importance_df["fold"] = fold + 1
#     feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

print('K-Fold score:{:.6f}'.format(r2_score(out_of_fold, Y_train)))

##### Feature Importance #####
cols = (feature_importance_df[["Feature", "importance"]]
        .groupby("Feature")
        .mean()
        .sort_values(by="importance", ascending=False)[:500].index)

best_features = feature_importance_df.loc[feature_importance_df.Feature.isin(
    cols)]

plt.figure(figsize=(10, 15))
sns.barplot(x="importance",
            y="Feature",
            data=best_features.sort_values(by="importance",
                                           ascending=False))
plt.title('LightGBM Features (avg over folds)')
plt.tight_layout()


# In[102]:


submission = pd.DataFrame({'pred':np.trunc(predictions)})
submission.to_csv('../submission/submit.csv', index = False, header=False,encoding='utf-8')


# In[100]:


good = pd.read_csv('../submission/submit_0.903715.csv',header=None)
r2_score(good,submission)


# In[101]:


good = pd.read_csv('../combine/combine_0.906198.csv',header=None)
r2_score(good,submission)


# In[74]:


data = pd.concat([X_train,Y_train],axis=1)
error_index = Y_train[np.abs(out_of_fold-Y_train)>2000].index
data_error = data.iloc[error_index]
figure=plt.figure(figsize=(8,6))
sns.scatterplot('area','tradeMoney',data=data)
sns.scatterplot('area','tradeMoney',data=data_error,label='big error')
plt.show()


# - 线下0.917602 | 线上0.904225  | baseline (5fold)
# - 线下0.917749 | 线上0.905396  | com_New_area_mean_price (5fold)
# - 线下0.918186 | 线上0.901026  | com_New_area_mean_price (5fold)+预测每平方米租金
# - 线下0.921427 | 线上0.894879  | com_New_area_mean_price (5fold) tradeMoney<25000
# - 线下0.917583 | 线上0.904383  | com_New_area_mean_price (5fold)+com_,plate_(二手和新房的均值，均值*area)
# - 线下0.917563 | 线上0.904217  | com_New_area_mean_price (5fold)+com_,plate_(二手和新房的均值，均值*area)+特征选择
# - 线下0.917176 | 线上0.906198  | com_New_area_mean_price (5fold) （莫名其妙又高了）
# 

# 
# + 线下0.891096 | 线上0.881447 | area<160,money<17000,无深度清洗
# + 线下0.909971 | 线上0.879594 | area<160,money<17000,有深度清洗
# + 线下0.914264 | 线上0.889846 | area<160,money<17000,有深度清洗+trn_communityName_area_mean
# + 线下0.930327 | 线上0.876083 | area<160,money<17000,有深度清洗+trn_communityName_area+trn_communityName_tradeMoney(mean+median)
# + 线下0.915355 | 线上 | area<160,money<17000,有深度清洗+area+totalFloor+pv+uv
# + 线下0.916975 | 线上 | area<160,money<17000,有深度清洗+area+totalFloor+pv+uv
# ---

# - Acv0.894212 | 线下0.916975 | 线上0.143911 |area+totalFloor+pv+uv+tradeMeanPrice+tradeNewMeanPrice(area median+mean+min+max+std+mod)
# - Acv0.893796 | 线下0.915289 | 线上0.894413 |area+totalFloor+pv+uv+tradeMeanPrice+tradeNewMeanPrice(area median+mean+std+mod)
# - Acv0.894341 | 线下0.915397 | 线上0.894832 |area+totalFloor+pv+uv+tradeMeanPrice+tradeNewMeanPrice(area median+mean+std+mod)+平滑
# - Acv0.894726 | 线下0.915657 | 线上0.895784 |area+totalFloor+pv+uv+tradeMeanPrice+tradeNewMeanPrice(area median+mean+std+mod)+平滑+totalTradeMoney等
# ---
# 
# 
# - Acv0.894126 | 线下0.915973 | 线上0.897248 |area+totalFloor+pv+uv+tradeMeanPrice+tradeNewMeanPrice(area median+mean+std+mod)+平滑+totalTradeMoney等(删除了help_sum、shop_num、trn_communityName_totalWorkers_mean特征)
# - Acv0.895166 | 线下0.915994 | 线上0.897577 |area+totalFloor+pv+uv+tradeMeanPrice+tradeNewMeanPrice(area median+mean+std+mod)+平滑+totalTradeMoney等(删除了特征选择删除的所有特征)
# - Acv0.895996 | 线下0.916577 | 线上0.898005 |area+totalFloor+pv+uv+tradeMeanPrice+tradeNewMeanPrice(area median+mean+std+mod)+平滑+totalTradeMoney等(一把梭特征+删除了特征选择删除的所有特征)
# - Acv0.897116 | 线下0.916784 | 线上0.898403 |area+totalFloor+pv+uv+tradeMeanPrice+tradeNewMeanPrice(area median+mean+std+mod)+平滑+totalTradeMoney等(一把梭特征)
# - Acv0.897116 | 线下0.916789 | 线上0.898380 |area+totalFloor+pv+uv+tradeMeanPrice+tradeNewMeanPrice(area median+mean+std+mod)+平滑+totalTradeMoney等(一把梭特征)+一把梭的特征也平滑处理
# - Acv0.896435 | 线下0.916784 | 线上0.898177 |area+totalFloor+pv+uv+tradeMeanPrice+tradeNewMeanPrice(area median+mean+std+mod)+平滑+totalTradeMoney等(一把梭特征)+PinkMan特征
# - Acv0.897116 | 线下0.919838 | 线上0.892716 |area+totalFloor+pv+uv+tradeMeanPrice+tradeNewMeanPrice(area median+mean+std+mod)+平滑+totalTradeMoney等(一把梭特征)+tradeMoney<20000
# - Acv0.893384 | 线下0.916828 | 线上0.899564 |area+totalFloor+pv+uv+tradeMeanPrice+tradeNewMeanPrice(area median+mean+std+mod)+平滑+totalTradeMoney等(一把梭特征)+tradeMoney<16000
# - Acv0.873614 | 线下0.911903 | 线上0.896008 |area+totalFloor+pv+uv+tradeMeanPrice+tradeNewMeanPrice(area median+mean+std+mod)+平滑+totalTradeMoney等(一把梭特征)+tradeMoney<14000
# - Acv0.888292 | 线下0.914891 | 线上0.899244 |area+totalFloor+pv+uv+tradeMeanPrice+tradeNewMeanPrice(area median+mean+std+mod)+平滑+totalTradeMoney等(一把梭特征)+tradeMoney<15000,tradeMoney>800
# - Acv0.893841 | 线下0.916682 | 线上0.899893 |area+totalFloor+pv+uv+tradeMeanPrice+tradeNewMeanPrice(area median+mean+std+mod)+平滑+totalTradeMoney等(一把梭特征)+tradeMoney<16000,tradeMoney>800
# - Acv0.892640 | 线下0.916975 | 线上0.900572 |area+totalFloor+pv+uv+tradeMeanPrice+tradeNewMeanPrice(area median+mean+std+mod)+平滑+totalTradeMoney等(一把梭特征)+tradeMoney<16000,tradeMoney>800+一把梭（median）
# - Acv忘了 | 线下0.917029 | 线上0.900785 |area+totalFloor+pv+uv+tradeMeanPrice+tradeNewMeanPrice(area median+mean+std+mod)+平滑+totalTradeMoney等(一把梭特征)+tradeMoney<16000,tradeMoney>800+一把梭（median）(mode)(多了一个bedroom mode)
# - Acv0.894219 | 线下0.917114 | 线上 |area+totalFloor+pv+uv+tradeMeanPrice+tradeNewMeanPrice(area median+mean+std+mod)+平滑+totalTradeMoney等(一把梭特征)+tradeMoney<16000,tradeMoney>800+一把梭（median）(mode)(删除了多了的bedroom mode)
# 

# - 线下0.916723 | 线上0.893542 | baseline
# - 线下0.918327 | 线上0.901778  | communityName groupby + 特征选择
# ---
# ```
# col = ['area', 'houseFloor', 'totalFloor', 'houseToward', 'houseDecoration', 'plate', 'buildYear', 
#  'saleSecHouseNum', 'subwayStationNum', 'busStationNum',  'interSchoolNum', 'privateSchoolNum', 'hospitalNum', 'gymNum',
#  'bankNum', 'parkNum', 'mallNum', 'tradeMeanPrice', 'tradeSecNum','tradeNewMeanPrice', 'remainNewNum', 'supplyNewNum', 'supplyLandNum',
#  'tradeLandNum', 'totalWorkers', 'newWorkers', 'residentPopulation', 'pv',
#  'lookNum', 'bedroom', 'hall', 'wc', 'tradeMonth', 'school_num', 'totalNewTradeMoney_Workers', 'bankNum_Workers', 
#  'trn_communityName_area_mode', 
#  'trn_communityName_area_std', 
#  'trn_communityName_area_median', 
#  'trn_communityName_bedroom_mode', 
#  'trn_communityName_pv_mode', 
#  'trn_communityName_tradeMeanPrice_mode',
#  'trn_communityName_tradeNewMeanPrice_mode', 
#  'trn_communityName_totalNewTradeMoney_mode', 
#  'trn_communityName_saleSecHouseNum_mode', 
#  'trn_communityName_totalTradeArea_mode', 
#  'trn_communityName_remainNewNum_mode', 
#  'trn_communityName_supplyNewNum_mode',
#  'trn_communityName_newWorkers_mode', 
#  'trn_communityName_saleSecHouseNum_mean', 
#  'trn_communityName_supplyNewNum_mean', 
#  'trn_communityName_newWorkers_mean', 
#  'trn_communityName_totalFloor_std',
#  'trn_communityName_pv_std', 
#  'trn_communityName_tradeMeanPrice_std', 
#  'trn_communityName_tradeNewMeanPrice_std', 
#  'trn_communityName_totalTradeMoney_std',
#  'trn_communityName_totalNewTradeMoney_std', 
#  'trn_communityName_totalTradeArea_std', 
#  'trn_communityName_tradeSecNum_std', 
#  'trn_communityName_remainNewNum_std',
#  'trn_communityName_supplyNewNum_std',
#  'trn_communityName_bedroom_std', 
#  'trn_communityName_totalTradeMoney_median', 
#  'trn_communityName_totalNewTradeMoney_median']
#  ```
#  ```
#  drop
# 
# 'pv_uv_ratio', 
# 'tradeFromBuildYear', 
# 'totalNewTradeMoney',
# 'help_sum', 'tradeNewNum', 'supplyLandArea', 
# 'totalTradeMoney',
# 'landTotalPrice', 
# 'uv', 
# 'shop_num',
# 'shopNum', 
# 'schoolNum', 
# 'totalNewTradeArea', 
# 'area_mean_price', 
# 'region', 'landMeanPrice', 
# 'totalTradeArea', 
# 'superMarketNum', 
# 'tradeLandArea',
# 'rentType', 
# 'gym_bankNum', 
# 'communityName', 
# 'play_sum',
# 'bus_sub_num', 
# 
# ```
# - 线下0.916294 | 线上0.899872  | communityName groupby + plate groupby + 特征选择
# - 线下0.918135 | 线上0.902116  | communityName groupby + 特征选择(baseline)
# - 线下0.918140 | 线上0.903715  | communityName groupby + 特征选择 + communityName w2v
# - 线下0.916789 | 线上0.902416  | communityName groupby + 特征选择 + communityName+area w2v
# - 线下0.917250 | 线上0.900725  | communityName groupby + 特征选择 + communityName w2v + 加上了communityName
# - 线下0.916321 | 线上0.900522  | communityName groupby + 特征选择 + communityName w2v + area_mean_price (5fold)
# - 线下0.917592 | 线上0.904252  | communityName groupby + 特征选择 + communityName w2v + com_area_mean_price(std+mean) (5fold)
# - 线下0.917143 | 线上0.903676  | communityName groupby + 特征选择 +(communityName+buildYear"删除了buildYear")(w2v) + com_area_mean_price(std+mean) (5fold)
# - 线下0.917602 | 线上0.904225  | communityName groupby + 特征选择 +(communityName+buildYear"保留了buildYear")(w2v) + com_area_mean_price(std+mean) (5fold)
# 

# + 2019未来杯高校AI挑战赛（城市-房产租金预测）
# ---
# 赛题是一个关于使用房屋、小区和配套设施等信息来预测房租的（回归）问题，应用数据挖掘技术构造特征，训练回归模型来预测房租，包括：
# 
# ---
# + 数据清洗、缺失值处理、特征选择和数据变换；
# + 挖掘特征：房子使用年限、卧室数量、学校总数、医院总数等；
# + 测试集和训练集分布不同，使用对抗验证来生成验证集，以此验证模型；
# + 训练LightGBM模型，得出预测结果。

# In[ ]:




