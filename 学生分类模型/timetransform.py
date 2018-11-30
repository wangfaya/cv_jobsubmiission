import time
import datetime
import warnings
warnings.filterwarnings('ignore')
import numpy as np
from sklearn.preprocessing import MinMaxScaler,OneHotEncoder,LabelEncoder
import pandas as pd
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report,roc_curve
from sklearn.model_selection import train_test_split,GridSearchCV
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
from sklearn.externals import joblib
import csv
import pymysql
import codecs
#清洗数据   历史数据
def clean_data(data):
    data.dropna(inplace=True)
    #家长最后一次登录app的时间转换
    # data['last_login_at']=data['last_login_at'].apply(lambda x:datetime.datetime.strptime(x,'%Y-%m-%d %H:%M:%S').strftime('%Y-%m-%d'))
    data['sex']=data['sex'].apply(lambda x: 1 if x =='男' else 0)
    data['start_time'] = data['starttime'].apply(lambda x: time.strftime('%Y-%m-%d', time.localtime(x)))
    data['end_time'] = data['endtime'].apply(lambda x: time.strftime('%Y-%m-%d', time.localtime(x)))
    data['days'] = (pd.to_datetime(data['end_time']) - pd.to_datetime(data['start_time'])).apply(lambda x:x.days)
    #家长最后一次登录app离孩子最后一次更新做题时间的天数
    # data['parents_care_days']=(pd.to_datetime(data['last_login_at'])-pd.to_datetime(data['end_time'])).apply(lambda x:x.days)
    data['ave_score']=np.array(data['scores']/data['days'],dtype=int)
    data['ave_dura1']=np.array(data['dura1']/data['days'],dtype=int)
    data['nowtime']=datetime.datetime.today().year
    data['age']=np.array(data['nowtime']-data['birthday'],dtype=int)
    data['age']=pd.cut(data.age,7)
    data['end_time']=pd.to_datetime(data['end_time'])
    data=data.set_index('end_time')
    data=data['2018']
    data = data[['sid','sex','scores', 'count_num_ips','dura1', 'days','age','ips_all','stage','accuracy_all_ips','status','area']]
    data=pd.DataFrame(data)
    data.to_csv('test.csv',encoding='utf-8')
    onehot=OneHotEncoder(sparse=False)
    mscaler=MinMaxScaler()
    stage_one=onehot.fit_transform(data[['stage','age','area']])
    data_min=mscaler.fit_transform(data[['sex','dura1','days','ips_all','scores', 'count_num_ips','accuracy_all_ips']])
    x=np.hstack((data_min,stage_one))
    '''
    print('正常样本和结课样本的比例:',data['status'][data['status'] == 1].count() / data['status'][data['status'] == -2].count())
    print('正常样本和退费样本的比例:',data['status'][data['status'] == 1].count() / data['status'][data['status'] == -3].count())
    正常样本和结课样本的比例: 8.563638746886879
    正常样本和退费样本的比例: 49.26998491704374
    正负样本相差较大 用smote进行过采样   因为过采样完成后 结课和正常数据契合度太高  对最终模型效果有很大的影响 所以放弃
    smot=SMOTE(random_state=1)
    over_x,over_y=smot.fit_sample(x,y)
    '''
    y=data['status']

    return x,y
#交叉验证  选择最优的模型
def cross_score(over_x,over_y):
    # lr=LogisticRegression()
    dtc=DecisionTreeClassifier()
    rfc=RandomForestClassifier()
    # svm=SVC()
    # gs=GaussianNB()
    # mlpc=MLPClassifier()
    xgbc=XGBClassifier()
    gbc=GradientBoostingClassifier()
    # cv1=cross_val_score(lr,over_x,over_y,cv=10)
    # print("逻辑回归的交叉验证为:",cv1.mean())
    cv2 = cross_val_score(dtc, over_x, over_y, cv=10)
    print("决策树的交叉验证为:", cv2.mean())
    cv3 = cross_val_score(rfc, over_x, over_y, cv=10)
    print("随机森林的交叉验证为:", cv3.mean())
    # cv4 = cross_val_score(gs, over_x, over_y, cv=10)
    # print("高斯的交叉验证为:", cv4.mean())
    # cv5 = cross_val_score(mlpc, over_x, over_y, cv=10)
    # print("神经网络的交叉验证为:", cv5.mean())
    cv6 = cross_val_score(xgbc, over_x, over_y, cv=10)
    print("xgbost的交叉验证为:", cv6.mean())
    cv7= cross_val_score(gbc,over_x,over_y,cv=10)
    print("梯度提升决策树的交叉验证为:",cv7.mean())
    # cv8=cross_val_score(svm,over_x,over_y,cv=10)
    # print("SVM的交叉验证为:",cv8.mean())


#开始训练
def train(over_x,over_y):

    train_x, test_x, train_y, test_y = train_test_split(over_x, over_y, train_size=0.6, random_state=123)

    rfc = RandomForestClassifier(
                                 oob_score=True,
                                 n_estimators=61
                                 )
    rfc.fit(train_x, train_y)
    print(rfc.oob_score_)
    joblib.dump(rfc,"./trainingmodel/students_tranmodel.pkl",compress=3)
    predict_y = rfc.predict(test_x)
    # feature_names=['sex', 'dura1', 'days','age','ips_all','count_num_ips','stage']
    # print('特征重要度排序为:',sorted(zip(map(lambda x:round(x,4),rfc.feature_importances_),feature_names),reverse=True))
    # print("袋外分数为:", rfc.oob_score_)
    print("训练集的准确率为:",accuracy_score(train_y,rfc.predict(train_x)))
    # pro_pridict=rfc.predict_proba(test_x)
    print(classification_report(test_y, predict_y))
    print(confusion_matrix(test_y, predict_y))

    # prf,prc,th=roc_curve(test_y,rfc.predict_proba(test_x)[:,1])
    # plt.plot(prf,prc)
    # plt.title("ROC")
    # plt.show()




    xgboost=XGBClassifier()
    gdbt=GradientBoostingClassifier()
    xgboost.fit(train_x,train_y)
    joblib.dump(xgboost,"./training/xgboost.pkl",compress=3)
    gdbt.fit(train_x,train_y)
    joblib.dump(gdbt,"./training/gdbt.pkl",compress=3)
    print("xgboost的准确率为:",accuracy_score(test_y,xgboost.predict(test_x)))
    print('xgboost的分类报告为:',classification_report(test_y,xgboost.predict(test_x)))
    print("gdbt的准确率为:",accuracy_score(test_y,gdbt.predict(test_x)))
    print('gdbt的分类报告为:',classification_report(test_y,gdbt.predict(test_x)))

#网格搜索  模型参数调参
def GridCV(x,y):
    param={
        'max_features' : range(1,10,1)
    }
    grid=GridSearchCV(RandomForestClassifier(oob_score=True,n_estimators=61),param_grid=param,cv=10,scoring='accuracy')
    grid.fit(x,y)
    print(grid.best_params_)
    print(grid.best_score_)


#清洗要预测的数据
def clean_pridict_data(data):
    data.dropna(inplace=True)
    # data['last_login_at']=data['last_login_at'].apply(lambda x:datetime.datetime.strptime(x,'%Y-%m-%d %H:%M:%S').strftime('%Y-%m-%d'))
    data['sex']=data['sex'].apply(lambda x:1 if x =='男' else 0)
    data['start_time'] = data['starttime'].apply(lambda x: time.strftime('%Y-%m-%d', time.localtime(x)))
    data['end_time'] = data['endtime'].apply(lambda x: time.strftime('%Y-%m-%d', time.localtime(x)))
    data['days'] =(pd.to_datetime(data['end_time']) - pd.to_datetime(data['start_time'])).apply(lambda x:x.days)
    data['ave_score']=np.array(data['scores']/data['days'],dtype=int)
    data['ave_dura1'] = np.array(data['dura1'] / data['days'], dtype=int)
    data['nowtime'] = datetime.datetime.today().year
    data['age']=np.array(data['nowtime']-data['birthday'],dtype=int)
    # data=data[data['status'].isin([1,-2])]
    data=data[['sid','sex','scores', 'count_num_ips','dura1','days','age','ips_all','accuracy_all_ips','stage','status','area']]
    data['age']=pd.cut(data.age,7)
    onehot = OneHotEncoder(sparse=False)
    mscaler = MinMaxScaler()
    stage_one = onehot.fit_transform(data[['stage','age','area']])
    data_min = mscaler.fit_transform(
        data[['sex','dura1', 'days', 'ips_all','scores', 'count_num_ips','accuracy_all_ips']])
    x = np.hstack((data_min, stage_one))
    y=data['status']

    return x,y,data

#开始预测测试  并把结果保存到csv的文件中
def application_model(x,y,data):
    aum=0
    bum=0
    cum=0
    result=0
    data=data[['sid','status']]
    rfc=RandomForestClassifier()
    rfc=joblib.load("./trainingmodel/students_tranmodel.pkl")
    predict=rfc.predict(x)
    for i in predict:
        if i==-3 or i==-2:
            aum+=1

    pro_predict=rfc.predict_proba(x)
    print("随机森林的准确率为:",accuracy_score(y,predict))
    print("随机森林分类报告:",classification_report(y,predict))
    xgboost = XGBClassifier()
    gdbt = GradientBoostingClassifier()
    xgboost=joblib.load( "./trainingmodel/xgboost.pkl")
    gdbt=joblib.load( "./trainingmodel/gdbt.pkl")
    xgboost_pre=xgboost.predict(x)
    gdbt_pre=gdbt.predict(x)
    for i in xgboost_pre:
        if i == -2 or i==-3:
            bum+=1
    for i in gdbt_pre:
        if i == -2 or i==-3:
            cum+=1
    for i in y:
        if i ==-2 or i==-3:
            result+=1
    print(len(gdbt_pre))
    print("xgboost的准确率为:", accuracy_score(y, xgboost_pre))
    print('xgboost的分类报告为:', classification_report(y, xgboost_pre))
    print("gdbt的准确率为:", accuracy_score(y, gdbt_pre))
    print('gdbt的分类报告为:', classification_report(y,gdbt_pre ))
    print("真实流失的人数为%s,随机森林预测可能流失的人数为%s,xgboost预测可能流失的人数为%s，gdbt预测可能流失的人数为%s" % (result, aum, bum, cum))

    data['gd'] = gdbt_pre
    data['rtc'] = predict
    data['xg'] = xgboost_pre
    data = data[['sid', 'status', 'rtc','xg','gd']]
    data.to_csv('data.csv', encoding='utf-8', index=None)

# 连接数据库
def getconn():
    conn=pymysql.connect(host='localhost',port=3306,user='root',passwd='root',db='腾讯招聘',charset='utf8')
    return conn
def insert(cur,sql,arg):
    cur.execute(sql,arg)

def csv_to_database(filename):
    with codecs.open(filename,mode='r',encoding='utf-8') as f:
        reads=csv.reader(f)
        head=next(reads)
        conn=getconn()
        cur=conn.cursor()
        sql='insert into students_pridict(sid,normal,abnormal,result_predict) values (%s,%s,%s,%s)'
        for item in reads:
            if item[1] is None or item[1] == '':
                continue
            arg=tuple(item)
            print(arg)
            insert(cur,sql,arg)
        conn.commit()
        cur.close()
        conn.close()








#主函数运行
if __name__ == '__main__':

    data = pd.read_csv("./data/test_datas_pure", encoding='utf-8', na_values=r'\N',
                       names=['class_id','sid','sex','scores', 'dura1', 'ips_all', 'count_num_ips', 'accuracy_all_ips',
                              'starttime', 'endtime','birthday','stage','status','area'])


    commit=int(input("请输入是训练(1)还是测试(2):"))
    if commit == 1:
        over_x,over_y=clean_data(data)
        # GridCV(over_x,over_y)
        # cross_score(over_x,over_y)
        train(over_x,over_y)
    elif commit == 2:
        x, y, data = clean_pridict_data(data)
        application_model(x, y, data)
    else:print("输入的指令错误")






