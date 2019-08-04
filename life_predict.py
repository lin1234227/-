#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from time import time
import pandas as pd
import lightgbm as lgb
# import matplotlib.pyplot as plt
# import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score,mean_squared_log_error,mean_squared_error,mean_absolute_error
# plt.rcParams['font.sans-serif'] = [u'SimHei']
# plt.rcParams['axes.unicode_minus'] = False



# In[2]:


def load_data():
	train_data = np.load('./data_10/train_10/instance.npy')
	train_label = np.load('./data_10/train_10/target.npy')
	train_weight = np.load('./data_10/train_10/weight.npy')
	test_data = np.load('./data_10/test1_10/instance.npy')
	test_label = np.load('./data_10/test1_10/target.npy')
	test_file = np.load('./data_10/test1_10/file.npy')
	return [train_data, train_label, train_weight, test_data, test_label, test_file]


train_data, train_label, train_weight, test_data, test_label, test_file = load_data()
print(train_data.shape, train_label.shape, train_weight.shape)


# In[4]:
# sns.kdeplot(train_label,  color="Blue", shade= True)


# In[5]:
# 分析筛选目标变量阈值大小
# num = len(train_label)
# for i in range(1, 10,1):
#     s = i*10000
#     print('{:<6}: {:.5f}'.format(s, len(train_label[train_label < s]) / num))


# In[6]:


features=['部件工作时长', '累积量参数1', '累积量参数2', '转速信号1', '转速信号2',
          '压力信号1', '压力信号2', '温度信号', '流量信号', '电流信号', '开关1信号', '开关2信号', '告警信号1']
# 0，1，2,9,12,13
# In[7]:

train=pd.DataFrame(train_data,columns=features)
test=pd.DataFrame(test_data,columns=features)
target=pd.DataFrame(train_label,columns=['target'])
train_weight=pd.DataFrame(train_weight)

# In[8]:

df_train = pd.concat([train,target],axis=1)

df_train = df_train[df_train['target']<30000]
df_train = df_train[df_train['target']>0]

# df_train = df_train[df_train['部件工作时长']<19000]
# df_train = df_train[df_train['累积量参数2']<390000]
# df_train = df_train[df_train['转速信号2']<43000]

df_train.reset_index(drop=True, inplace=True)

target = df_train.pop('target')
train = df_train
# print(df_train.columns)


# In[9]:


numic_cols = features[:10]
categoricals = features[10:]
# print(numic_cols,categoricals)


# In[10]:


for col in categoricals:
    train[col] = train[col].astype('category')
    test[col] = test[col].astype('category')



def add_features(df):
    # ['部件工作时长mean_开关1信号', '累积量参数1min_开关1信号', '部件工作时长std_告警信号1',
    temp=df.groupby('开关1信号',as_index=False)['部件工作时长'].agg({'部件工作时长'+'mean_'+'开关1信号':'mean'})
    df = df.merge(temp,on='开关1信号',how='left')
    temp=df.groupby('开关1信号',as_index=False)['累积量参数1'].agg({'累积量参数1'+'min_'+'开关1信号':'min'})
    df = df.merge(temp,on='开关1信号',how='left')
    temp=df.groupby('告警信号1',as_index=False)['部件工作时长'].agg({'累积量参数1'+'std_'+'告警信号1':'std'})
    df = df.merge(temp,on='告警信号1',how='left')
    return df


def add_statistics_features(df):
    df['累积量参数1+累积参数2']=df['累积量参数1']+df['累积量参数2']
    df['累积量参数1-累积参数2']=df['累积量参数1']-df['累积量参数2']
    df['累积量参数1*累积参数2']=df['累积量参数1']*df['累积量参数2']
    df['累积量参数1/累积参数2']=df['累积量参数1']/df['累积量参数2']


    df['累积量参数1平方']=df['累积量参数1']*df['累积量参数1']
    df['累积量参数1立方']=df['累积量参数1']*df['累积量参数1']*df['累积量参数1']
    df['累积量参数2平方']=df['累积量参数2']*df['累积量参数2']
    df['累积量参数2立方']=df['累积量参数2']*df['累积量参数2']*df['累积量参数2']


    df['转速信号1+转速信号2']=df['转速信号1']+df['转速信号2']
    df['转速信号1-转速信号2']=df['转速信号1']-df['转速信号2']
    df['转速信号1*转速信号2']=df['转速信号1']*df['转速信号2']
    df['转速信号1/转速信号2']=df['转速信号1']/df['转速信号2']


    df['转速信号1平方']=df['转速信号1']*df['转速信号1']
    df['转速信号1立方']=df['转速信号1']*df['转速信号1']*df['转速信号1']
    df['转速信号2平方']=df['转速信号2']*df['转速信号2']
    df['转速信号2立方']=df['转速信号2']*df['转速信号2']*df['转速信号2']


    df['压力信号1+压力信号2']=df['压力信号1']+df['压力信号2']
    df['压力信号1-压力信号2']=df['压力信号1']-df['压力信号2']
    df['压力信号1*压力信号2']=df['压力信号1']*df['压力信号2']
    df['压力信号1/压力信号2']=df['压力信号1']/df['压力信号2']


    df['压力信号1平方']=df['压力信号1']*df['压力信号1']
    df['压力信号1立方']=df['压力信号1']*df['压力信号1']*df['压力信号1']
    df['压力信号2平方']=df['压力信号2']*df['压力信号2']
    df['压力信号2立方']=df['压力信号2']*df['压力信号2']*df['压力信号2']
    return df



train = add_features(train)
test = add_features(test)

train = add_statistics_features(train)
test = add_statistics_features(test)



# f = [ '转速信号1+转速信号2','流量信号','累积量参数2','压力信号2',
#       '电流信号', '部件工作时长','温度信号',
#        '累积量参数1', '压力信号1', '转速信号1',
#       '开关1信号', '开关2信号', '告警信号1']
f = [ '累积量参数1+累积参数2', '压力信号1+压力信号2','转速信号1+转速信号2',
      '电流信号', '部件工作时长', '温度信号',
       '累积量参数1', '压力信号1', '转速信号1',
      '开关1信号', '开关2信号', '告警信号1']
train  = train[f]
test = test[f]
features = f

# def get_NANorINF(features):
#     nan_cols = []
#     count=0
#     features = list(set(features)-set(categoricals))
#     for i in features:
#         if len(list(set(np.isnan(train[i]).values)))!=1 or len(list(set(np.isinf(train[i]).values)))!=1:
#             count+=1
#             nan_cols.append(i)
#     print('一共{}个含有inf or nan 的列'.format(count))
#     return nan_cols

# nan_cols = get_NANorINF(features)
# train = train.drop(nan_cols,axis=1)
# features = list(set(features)-set(nan_cols))



params = {
        'num_leaves': 100,
        'min_data_in_leaf': 50,
        'min_child_samples':50,
        'objective': 'regression',
        'learning_rate': 0.01,
        "boosting": "gbdt",
        "feature_fraction": 1.0,
        "bagging_freq": 1,
        "bagging_fraction": 0.85,
        "bagging_seed": 23,
        "metric": 'rmse',
        "lambda_l1": 0.1,
        "lambda_l2": 0.2,
        "nthread": -1,

    }


folds = KFold(n_splits=5, shuffle=True, random_state=2333)

oof_lgb = np.zeros(len(train))
predictions_lgb = np.zeros(len(test))
start=time()

for fold_, (trn_idx, val_idx) in enumerate(folds.split(train.values, target.values)):
    print("fold {}".format(fold_))
    trn_data = lgb.Dataset(train.iloc[trn_idx], label=target.iloc[trn_idx],weight=train_weight.iloc[trn_idx][0].values,categorical_feature=categoricals)
    val_data = lgb.Dataset(train.iloc[val_idx], label=target.iloc[val_idx],weight=train_weight.iloc[val_idx][0].values,categorical_feature=categoricals)
    num_round = 20000
    clf = lgb.train(params, trn_data, num_round, valid_sets=[trn_data, val_data], verbose_eval=500,
                    early_stopping_rounds=200)

    oof_lgb[val_idx] = clf.predict(train.iloc[val_idx], num_iteration=clf.best_iteration)
    clf.save_model('./'+str(fold_)+'.txt')
    predictions_lgb += clf.predict(test, num_iteration=clf.best_iteration) / folds.n_splits
print("CV Score: {:<8.5f}".format(r2_score(target, oof_lgb)))
print("CV mean_squared_error: {:<8.5f}".format(mean_squared_error(target, oof_lgb)))
print("CV mean_absolute_error: {:<8.5f}".format(mean_absolute_error(target, oof_lgb)))


prediction = predictions_lgb
end = time()
print("prediction time: " + str(end - start) + " sec")
result = {}
for index, item in enumerate(test_file):
    if item not in result:
        result[item] = prediction[index]
    else:
        result[item] = (result[item] + prediction[index]) / 2

df = pd.DataFrame(list(result.items()), columns=['test_file_name', 'life'])
df.to_csv('submit_hsl.csv', index=False)







