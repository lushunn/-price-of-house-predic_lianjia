# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 11:29:50 2018

@author: Dell
"""
'''
ershoufang
'''
datalist=[]
with open(r'D:\workfile\work2\data\ershou\new.csv') as f:
    for line in f:
        line=line.strip(',,,,\n').split(',')
        line=line[:5]+line[-1].split('/')
        datalist.append(line)
import numpy as np
import pandas as pd
data=pd.DataFrame(datalist)
data=data.iloc[:,1:-1]
data.columns=['区','位置','描述','价格','小区','户型','面积','朝向','类型','电梯']
data.drop_duplicates()
#数据清洗


for indexs in data.index:
    if(data.loc[indexs]['户型'] != None)and( '别墅' in data.loc[indexs]['户型']): 
        data.loc[indexs]['面积']=data.loc[indexs]['朝向']
        data.loc[indexs]['朝向']=data.loc[indexs]['类型']
        data.loc[indexs]['朝向']=data.loc[indexs]['类型']
        data.loc[indexs]['类型']=data.loc[indexs]['电梯']
        data.loc[indexs]['电梯']=None
for indexs in data.index:
    if( data.loc[indexs].values[-1] == None) and (data.loc[indexs].values[2] != None)and ('电梯' in data.loc[indexs].values[2]):
        data.loc[indexs].values[-1]='有电梯' 
data=data.drop(data.index[data['户型']=='车位'])

for indexs in data.index:
    if ( data.loc[indexs]['面积'] != None):      
        data.loc[indexs]['面积']=data.loc[indexs]['面积'][:-2]
for indexs in data.index:       
     if( data.loc[indexs]['价格'] != None):
         data.loc[indexs]['价格']=data.loc[indexs]['价格'][:-1]
         
for indexs in data.index:
    if(data.loc[indexs]['价格']!=None):
        try:
            data.loc[indexs]['价格']==int(data.loc[indexs]['价格'])
        except ValueError:
            data.loc[indexs]['价格']= None
data.info()
data=data.replace({None:-1})
data[['价格']]=data[['价格']].astype(int)
data[['面积']]=data[['面积']].astype(float)
data=data.replace({-1:np.NaN})
#出售量
df_district = data['区'].value_counts()       
failegetdata=df_district[df_district <2]

for indexs in data.index:
    if data.loc[indexs]['区'] in list(failegetdata.index):
        data=data.drop([indexs])
df_district = data['区'].value_counts()
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False
df_district.plot(kind="bar", title="各区在售二手房数量",figsize = (12, 9))
df_xq = data['小区'].value_counts() 
df_xq[:5].plot(kind="bar", title="在售数量最多的五个小区",figsize = (12, 9))
#出售价
data.to_csv('tmp.csv')
data=pd.read_csv('tmp.csv',encoding='gbk')
data['单价']=data['价格']/data['面积']
data['单价'][data['单价']>0.3].plot.hist(bins = 20,title='在售二手房单价直方图',figsize = (12, 9))
a=data[['单价','区']].dropna()

grouped =a['单价'].groupby(a['区']).apply(np.median)
a=list(grouped.index[0:2]+'区')
a.append('亦庄开发区')
a=a+list(grouped.index[3:]+'区')
grouped.index=a
from pyecharts import Map
value =list(grouped)
attr =list(grouped.index)
map=Map("二手房在售单价中位数 单位：万元", width=1200, height=600)
map.add("", attr, value, maptype='北京', is_visualmap=True, visual_text_color='#404a59',visual_range=[1,10],is_map_symbol_show =False)
map.show_config()
map.render()

df=pd.read_csv('ershoufangshuju.csv',encoding='gbk')
del df['Unnamed: 0']
df['电梯'][df['类型']=='有电梯']=df['类型'][df['类型']=='有电梯']
df['电梯'][df['类型']=='无电梯']=df['类型'][df['类型']=='无电梯']
df['类型'][df['类型']=='有电梯']=np.NaN
df['类型'][df['类型']=='无电梯']=np.NaN

tmp=pd.crosstab(df['区'],df['类型'],margins=True)
def plotbar_Stacked(tmp,t=''):
    tmp=tmp.apply(lambda x:x/x[-1],axis=1)
    tmp=tmp.drop('All',1).drop('All',0)
    tmp.plot.bar(stacked=True,figsize = (12, 9),title=str(t))
plotbar_Stacked(tmp,'各区在售二手房装修类型比重')
df['室数']=df['户型']
#for indexs in df.index:
    #if(df.loc[indexs]['户型'] is not  np.NaN):
        #df.loc[indexs]['室数']=df.loc[indexs]['户型'][0]
df['室数']=df['室数'].apply(lambda x: x[0] if not x is np.NaN else x)
df['室数']=df['室数'].apply(lambda x: '别墅' if x in ['双','叠','独','联'] else x)
df['室数']=df['室数'].apply(lambda x: '3居室以下' if x in ['1','2','3'] else x)
df['室数']=df['室数'].apply(lambda x: '4-6居室' if x in ['4','5','6'] else x)
df['室数']=df['室数'].apply(lambda x: '别墅'  if x in ['7','8','9'] else x)
tmp=pd.crosstab(df['区'],df['室数'],margins=True)
df=data.drop(df.index[df['室数']=='车'])
plotbar_Stacked(tmp,'各区在售二手房室数比重')
import seaborn as sns
a=df[['价格','类型','区']].dropna()
a=a.pivot_table(index='区',columns='类型',values='价格', aggfunc=np.median)
sns.factorplot(x='类型', y='价格',data=df, size=4, aspect=2)
a=a.ix[[0,1,9,10,11,12],]
a.plot.line(stacked=False,color=['yellow','red','blue','green'],figsize = (12, 9),title='城六区各装修类型二手房售价中位数',fro)
df1=pd.read_csv(r'D:\workfile\work1\bj_info.csv')
del df1['Unnamed: 0']
df1=df1[df1['参考均价']!='暂无报价']
import json
from urllib.request import urlopen, quote
import requests,csv
import pandas as pd #导入这些库后边都要用到

def getlnglat(address):
    url = 'http://api.map.baidu.com/geocoder/v2/'
    output = 'json'
    ak = 'ry6kSSLorwwNZGUKPe3hZUjSDmyI1maC'
    add = quote(address) #由于本文城市变量为中文，为防止乱码，先用quote进行编码
    uri = url + '?' + 'address=' + add  + '&output=' + output + '&ak=' + ak
    req = urlopen(uri)
    res = req.read().decode() #将其他编码的字符串解码成unicode
    temp = json.loads(res) #对json数据进行解析
    return temp
file = open('point_f.json','w')
faillist=[]
for indexs in df1.index:     
    a=df1.loc[indexs]['地址']
    b=df1.loc[indexs]['参考均价']
    try:
        lng = getlnglat(a)['result']['location']['lng']
        lat = getlnglat(a)['result']['location']['lat'] 
        str_temp = '{"lat":' + str(lat) + ',"lng":' + str(lng) + ',"count":' + str(b) +'},'
    except KeyError as e:
        faillist.append(a)
        print(e)
    file.write(str_temp)
file.close()

#建模
f = open(r'D:\workfile\work1\二手房在售数据整理(纯数字)\二手房在售数据整理(纯数字).csv')
df=pd.read_csv(f)
#数据预处理
from sklearn import preprocessing
df['面积_scale']=preprocessing.scale(df['面积'])
df['卧室'][df['卧室']<=3]=0
df['卧室'][df['卧室']>3]=1

df['客厅'][df['客厅']<2]=0
df['客厅'][df['客厅']>=2]=1
y=df.ix[:,0]
X=df.ix[:,2:]

from sklearn.ensemble import AdaBoostRegressor

#特征选择
# AdaBoost
ada_est =AdaBoostRegressor(random_state=0)
ada_param_grid = {'n_estimators': [500], 'learning_rate': [0.01, 0.1]}
ada_grid = model_selection.GridSearchCV(ada_est, ada_param_grid, n_jobs=25, cv=10, verbose=1)
ada_grid.fit(X, y)
print('Top N Features Best Ada Params:' + str(ada_grid.best_params_))
print('Top N Features Best Ada Score:' + str(ada_grid.best_score_))
print('Top N Features Ada Train Score:' + str(ada_grid.score(X, y)))
feature_imp_sorted_ada = pd.DataFrame({'feature': list(X),
 'importance': ada_grid.best_estimator_.feature_importances_}).sort_values('importance', ascending=False)
features_top_n_ada = feature_imp_sorted_ada.head(7)['feature']
print(str(features_top_n_ada))
del df['朝向']
tmp=pd.get_dummies(df['装修'])
df=pd.concat([df, tmp], axis=1)
del df['装修']
y=df.ix[:,0]
X=df.ix[:,2:]

#朝向重要性 0，舍去

from sklearn import ensemble
from sklearn import model_selection
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
#调参
#gbm
gbm_reg = GradientBoostingRegressor(random_state=42)
gbm_reg_param_grid = {'n_estimators': [2000], 'max_depth': [4], 'learning_rate': [0.01,0.2], 'max_features': [3]}
gbm_reg_grid = model_selection.GridSearchCV(gbm_reg, gbm_reg_param_grid, cv=10, n_jobs=25, verbose=1, scoring='neg_mean_squared_error')
gbm_reg_grid.fit(X_train, y_train)

print(' Best  Params:' + str(gbm_reg_grid.best_params_))
gbm_reg_grid=GradientBoostingRegressor(n_estimators=2000,max_depth=4, learning_rate= 0.01, max_features=3)
gbm_reg_grid.fit(X_train, y_train)
from sklearn.externals import joblib

y_predict_gbm=gbm_reg_grid.predict(X_test)
mae_gbm=(np.abs(y_predict_gbm-y_test)).sum()/9467
joblib.dump(gbm_reg_grid, 'gbm_reg_grid.model')
print(mae_gbm)
#mae=1.1141107597511322
#rf
rf_reg = RandomForestRegressor()
rf_reg_param_grid = {'n_estimators': [2000], 'max_depth': [5,10], 'random_state': [0]}
rf_reg_grid = model_selection.GridSearchCV(rf_reg, rf_reg_param_grid, cv=10, n_jobs=25, verbose=1, scoring='neg_mean_squared_error')
rf_reg_grid.fit(X_train, y_train)
print(' Best  Params:' + str(rf_reg_grid.best_params_))
rf_reg_grid=RandomForestRegressor(n_estimators=2000,max_depth=10, random_state=0)
rf_reg_grid.fit(X_train, y_train)
joblib.dump(rf_reg_grid, 'rf_reg_grid.model')
y_predict_rf=rf_reg_grid.predict(X_test)
mae_rf=(np.abs(y_predict_rf-y_test)).sum()/9467
print(mae_rf)
#mae=1.1062916504558062
#ada (舍去)
ada_est_param_grid = {'n_estimators': [5000], 'learning_rate': [0.02,0.1], 'random_state': [0]}
ada_est =AdaBoostRegressor()
ada_est_grid = model_selection.GridSearchCV(ada_est, ada_est_param_grid, cv=10, n_jobs=25, verbose=1, scoring='neg_mean_squared_error')
ada_est_grid.fit(X_train, y_train)
print(' Best  Params:' + str(ada_est_grid.best_params_))
joblib.dump(rf_reg_grid, 'ada_est.model')
ada_est=AdaBoostRegressor(n_estimators=5000,learning_rate=0.02)
ada_est.fit(X_train, y_train)
y_predict_ada=ada_est.predict(X_test)
mae_ada=(np.abs(y_predict_ada-y_test)).sum()/9467
print(mae_ada)
#1.2866268400664551
#svr
from sklearn.svm import SVR 

svr_param_grid = {'kernel':('rbf','linear')}
svr = SVR(gamma=0.1)
svr_grid = model_selection.GridSearchCV(svr, svr_param_grid, cv=10, n_jobs=25, verbose=1, scoring='neg_mean_squared_error')
svr_grid.fit(X_train, y_train)
print(' Best  Params:' + str(svr_grid.best_params_))
svr = SVR(gamma=0.1,kernel='rbf')
svr.fit(X_train, y_train)
y_predict_svr=svr.predict(X_test)
mae_svr=(np.abs(y_predict_svr-y_test)).sum()/9467
joblib.dump(svr, 'svr.model')
print(mae_svr)
#KNN
from sklearn.neighbors import KNeighborsRegressor
KNN = KNeighborsRegressor()
knn_param_grid = {'n_neighbors':[3,10]}
knn_grid = model_selection.GridSearchCV(KNN, knn_param_grid, cv=10, n_jobs=25, verbose=1, scoring='neg_mean_squared_error')
knn_grid.fit(X_train, y_train)
print(' Best  Params:' + str(knn_grid.best_params_))
KNN = KNeighborsRegressor(n_neighbors=10)
KNN.fit(X_train, y_train)
y_predict_knn=KNN.predict(X_test)
mae_knn=(np.abs(y_predict_knn-y_test)).sum()/9467
joblib.dump(KNN, 'KNN.model')
print(mae_knn)
#mlp
from sklearn.neural_network import MLPRegressor
MLP = MLPRegressor(hidden_layer_sizes=(300, 200,200),max_iter=100,activation='relu')
MLP.fit(X_train, y_train)
y_predict_MLP=MLP.predict(X_test)
mae_MLP=(np.abs(y_predict_MLP-y_test)).sum()/9467
joblib.dump(MLP, 'MLP.model')
print(mae_MLP)
#xgb
import xgboost  as xgb
x_regress = xgb.XGBRegressor(max_depth=20,n_estimators =5000)
x_regress_param_grid = {'max_depth': [5,20]}
x_regress_grid = model_selection.GridSearchCV(x_regress, x_regress_param_grid, cv=10, n_jobs=25, verbose=1, scoring='neg_mean_squared_error')
x_regress.fit(X_train, y_train)
joblib.dump(x_regress, 'x_regress_grid.model')
y_predict_xgb=x_regress.predict(X_test)

mae_xgb=(np.abs(y_predict_xgb-y_test)).sum()/9467
# 模型融合
#简单平均 
pred=pd.DataFrame({'ada':y_predict_ada,'gbm':y_predict_gbm,'rf':y_predict_rf,'svr':y_predict_svr,'knn':y_predict_knn,'mlp':y_predict_MLP,'xgb':y_predict_xgb})
pred['pred']=pred['ada']+pred[ 'knn']+pred[ 'gbm']+pred['mlp']+pred['rf']+pred['svr']+pred['xgb']
pred['pred']=pred['pred']/7
mae_final=(np.abs(pred['pred']-y_test)).sum()/9467
print(mae_final)
#0.8367033561074727
pred_['pred']=pred[ 'gbm']+pred['mlp']+pred['rf']+pred['svr']+pred['xgb']+pred['knn']
pred['pred']=pred['pred']/6
mae_final_1=(np.abs(pred['pred']-y_test)).sum()/9467
print(mae_final_1)

#加权平均
pred['pred']=0.332020866*pred[ 'gbm']+0.334367542*pred['rf']+0.333611592*pred['svr']
pred['pred']=pred['pred']/3
mae_final_1=(np.abs(pred['pred']-y_test)).sum()/9467
print(mae_final_1)
#stacking

from sklearn.model_selection import KFold
ntrain = X_train.shape[0]
ntest = X_test.shape[0]
SEED = 0 # for reproducibility
NFOLDS = 3 # set folds for out-of-fold prediction
kf = KFold(n_splits = NFOLDS, random_state=SEED, shuffle=False)

def get_out_fold(clf, x_train, y_train, x_test):
    oof_train = np.zeros((ntrain,))
    oof_test = np.zeros((ntest,))
    oof_test_skf = np.empty((NFOLDS, ntest))

    for i, (train_index, test_index) in enumerate(kf.split(x_train)):
        x_tr = x_train[train_index]
        y_tr = y_train[train_index]
        x_te = x_train[test_index]

        clf.fit(x_tr, y_tr)

        oof_train[test_index] = clf.predict(x_te)
        oof_test_skf[i, :] = clf.predict(x_test)

    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)

x_train = X_train.values
x_test = X_test.values
y_train_1 = y_train.values
gbm_oof_train, gbm_oof_test = get_out_fold(gbm_reg_grid, x_train, y_train_1, x_test) 
rf_oof_train, rf_oof_test = get_out_fold(rf_reg, x_train, y_train_1, x_test) 
#ada_oof_train, ada_oof_test = get_out_fold(ada_est, x_train, y_train_1, x_test)
svr_oof_train,svr_oof_test = get_out_fold(svr, x_train, y_train_1, x_test) 
#knn_oof_train, bayes_oof_test = get_out_fold(KNN, x_train, y_train_1, x_test) 
#MLP_oof_train, MLP_oof_test = get_out_fold(MLP, x_train, y_train_1, x_test) 
#xbg_oof_train, xbg_oof_test = get_out_fold(x_regress, x_train, y_train_1, x_test) 
knn_oof_test=bayes_oof_test
#gbt LEVEL2

x_train_f = np.concatenate((gbm_oof_train, rf_oof_train, svr_oof_train), axis=1)
x_test_f = np.concatenate((gbm_oof_test, rf_oof_test,  svr_oof_test), axis=1)
gbm_f = GradientBoostingRegressor(random_state=42,n_estimators=2000,max_depth=5,learning_rate=0.01,max_features=3)

gbm_f.fit(x_train_f, y_train)
y_predict_f=gbm_f.predict(x_test_f)
mae_gbm_f=(np.abs(y_predict_f-y_test)).sum()/9467
