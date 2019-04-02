


import datetime
import warnings
warnings.filterwarnings('ignore')
import numpy as np
#smote解决正负样本比例不平衡问题
from imblearn.over_sampling import SMOTE
#数据预处理   标准化 跟 独热编码转换
from sklearn.preprocessing import OneHotEncoder,StandardScaler
import pandas as pd
#支持向量机
from sklearn.svm import SVC
#随机森林跟gdbt
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
#评估方法，准确率，混淆矩阵，分类报告，精确率，召回率
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report,precision_score,recall_score
#划分数据集，网格搜索 找最有参数
from sklearn.model_selection import train_test_split,GridSearchCV
#逻辑回归
from sklearn.linear_model import LogisticRegression
#knn
from sklearn.neighbors import KNeighborsClassifier
#xgboost以及打印xgboost重要度
from xgboost import XGBClassifier,plot_importance
#决策树
from sklearn.tree import DecisionTreeClassifier
#神经网络
from sklearn.neural_network import MLPClassifier
#朴素贝叶斯
from sklearn.naive_bayes import GaussianNB
#交叉验证
from sklearn.model_selection import cross_val_score
#保存模型
from sklearn.externals import joblib
#把预测结果往数据库里保存
import csv
import pymysql
import codecs


#清洗数据
def clean_data(data):

    data=data[data['status'].isin((1,-2))]
    # data['status']=data['status'].replace(-3,-2)
    print(data['status'].value_counts())
    for colum in list(data.columns[data.isnull().sum()>0]):
        if colum != 'read_minday'and colum!='read_maxday':
            data[colum].fillna(value=0,inplace=True)
        else:
            data[colum].fillna(data['read_minday'].mode()[0],inplace=True)


    data['study_days'] = (pd.to_datetime(data['read_maxday'], format='%Y%m%d') - pd.to_datetime(data['read_minday'],format='%Y%m%d')).apply(lambda x: x.days)
    data['sex']=data['sex'].apply(lambda x: 1 if x =='男' else 0)
    data['nowtime'] = datetime.datetime.today().year
    data['age'] = np.array(data['nowtime'] - data['birthday'], dtype=int)
    data['age'] = pd.cut(data.age, 7)
    onehot=OneHotEncoder(sparse=False)
    mscaler=StandardScaler()
    stage_one=onehot.fit_transform(data[['sex','stage','age']])
    data_min = mscaler.fit_transform(data[['scores','ips_nums',
                                   'ips_complete_topicnums',
                                  'ips_finish_sum', 'read_numss','read_day','read_study_times',
                                  'read_mession_firsts','read_book_nums',
                                  'read_mession_complate_nums','read_finish_sum']])


    x=np.hstack((data_min,stage_one))
    y=data['status']
    smot = SMOTE(random_state=1)
    over_x, over_y = smot.fit_sample(x, y)
    return over_x,over_y


#交叉验证  选择最优的模型
def cross_score(over_x,over_y):
    lr=LogisticRegression()
    dtc=DecisionTreeClassifier()
    rfc=RandomForestClassifier()
    gs=GaussianNB()
    svm=SVC(C=0.8)
    mlpc=MLPClassifier()
    xgbc=XGBClassifier()
    gbc=GradientBoostingClassifier()
    knn=KNeighborsClassifier()
    cv1=cross_val_score(knn,over_x,over_y,cv=10,scoring='accuracy')
    print(" KNN的交叉验证为:",cv1.mean())
    cv9=cross_val_score(lr,over_x,over_y,cv=10)
    print("逻辑回归的交叉验证为:",cv9.mean())
    cv2 = cross_val_score(dtc, over_x, over_y, cv=10,scoring='accuracy')
    print("决策树的交叉验证为:", cv2.mean())
    cv3 = cross_val_score(rfc, over_x, over_y, cv=10,scoring='accuracy')
    print("随机森林的交叉验证为:", cv3.mean())
    cv4 = cross_val_score(gs, over_x, over_y, cv=10)
    print("高斯的交叉验证为:", cv4.mean())
    cv5 = cross_val_score(mlpc, over_x, over_y, cv=10)
    print("神经网络的交叉验证为:", cv5.mean())
    cv6 = cross_val_score(xgbc, over_x, over_y, cv=10,scoring='accuracy')
    print("xgbost的交叉验证为:", cv6.mean())
    cv7= cross_val_score(gbc,over_x,over_y,cv=10,scoring='accuracy')
    print("梯度提升决策树的交叉验证为:",cv7.mean())
    cv8=cross_val_score(svm,over_x,over_y,cv=10,scoring='accuracy')
    print("SVM的交叉验证为:",cv8.mean())


#开始训练
def train(over_x,over_y):
    print('训练开始')
    rfc = RandomForestClassifier(
                                 oob_score=True,
                                 n_estimators=90,
                                 max_depth=75,
                                 max_features=3,
                                 random_state=10

                                 )
    rfc.fit(over_x, over_y)
    joblib.dump(rfc,"./trainingmodel/students_tranmodel.pkl",compress=3)

    #随机森林打印特征重要度
    # feature_names=['stage','sex','area_name','scores','ips_nums','ips_times',
    #                               'ips_complete_topicnums','ips_should_topicnums','first_accuracy_nums','ips_finish_sum',
    #                               'read_numss','bookromm_times','read_mission_times','read_study_times','read_day','read_book_nums',
    #                               'read_mession_firsts','read_mession_frists_nums','read_mession_complate_nums',
    #                               'read_mession_student_all','read_finish_sum']
    # print('特征重要度排序为:',sorted(zip(map(lambda x:round(x,4),rfc.feature_importances_),feature_names),reverse=True))

    svm = SVC(C=0.8,kernel='rbf')
    svm.fit(over_x, over_y)
    joblib.dump(svm, "./trainingmodel/svm.pkl", compress=3)
    print('训练结束')



#网格搜索  模型参数调参
def GridCV(x,y):
    print('网格搜索最优模型参数中........')
    # param={
    #     'max_depth' : range(10,100,5),
    #     'n_estimators':range(10,100,10),
    #     'max_features':range(3,15,2)
    # }
    # grid=GridSearchCV(RandomForestClassifier(oob_score=True),param_grid=param,scoring='accuracy')
    # grid.fit(x,y)
    # print(grid.best_params_)
    # print(grid.best_score_)
    param = {
        # 'gamma':[1,0.1,0.01],
        'C':[0.1,1,0.01],
        'kernel':['rbf','linear','poly','sigmoid']
    }
    grid=GridSearchCV(SVC(kernel='linear'),param_grid=param,scoring='precision',cv=4)
    grid.fit(x, y)
    print(grid.best_params_)
    print(grid.best_score_)



#清洗要预测的数据
def clean_pridict_data(data):

    data['status'] = data['status'].replace(-3, -2)
    for colum in list(data.columns[data.isnull().sum() > 0]):
        if colum != 'read_minday' and colum != 'read_maxday':
            data[colum].fillna(value=0, inplace=True)
        elif colum=='trend':
            data[colum].fillna(value="无变化",inplace=True)
        else:
            data[colum].fillna(data['read_minday'].mode()[0], inplace=True)

    data['study_days'] = (pd.to_datetime(data['read_maxday'], format='%Y%m%d') - pd.to_datetime(data['read_minday'],format='%Y%m%d')).apply(lambda x: x.days)
    data['sex'] = data['sex'].apply(lambda x: 1 if x == '男' else 0)

    data['nowtime'] = datetime.datetime.today().year
    data['age'] = np.array(data['nowtime'] - data['birthday'], dtype=int)
    data['age'] = pd.cut(data.age, 7)
    onehot = OneHotEncoder(sparse=False)
    mscaler = StandardScaler()
    stage_one = onehot.fit_transform(data[['sex','stage','age']])
    data_min = mscaler.fit_transform(data[['scores','ips_nums',
                                   'ips_complete_topicnums',
                                  'ips_finish_sum', 'read_numss','read_day','read_study_times',
                                  'read_mession_firsts','read_book_nums',
                                  'read_mession_complate_nums','read_finish_sum'
                                           ]])
    x = np.hstack((data_min, stage_one))
    y=data['status']
    return x,y,data

#开始预测测试  并把结果保存到csv的文件中
def application_model(x,y,data):
    aum=0
    bum=0
    result=0
    data=data[['sid','status']]
    svm=SVC()
    svm=joblib.load("./trainingmodel/svm.pkl")
    svmpre=svm.predict(x)
    print("SVM的准确率为:",accuracy_score(y,svm.predict(x)))
    print("SVM的精确率为:",precision_score(y,svm.predict(x),average='macro'))
    print("SVM的召回率为:",recall_score(y,svm.predict(x),average='macro'))
    print("SVM的分类报告为:",classification_report(y,svm.predict(x)))
    print("SVM的混淆矩阵为:",confusion_matrix(y,svm.predict(x)))
    rfc=RandomForestClassifier()
    rfc=joblib.load("./trainingmodel/students_tranmodel.pkl")
    predict=rfc.predict(x)
    for i in predict:
        if i==-3 or i==-2:
            aum+=1
    for i in svmpre:
        if i==-3 or i==-2:
            bum+=1
    pro_predict=rfc.predict_proba(x)
    normal=[]
    abnormal=[]
    for i in pro_predict:
        abnormal.append(i[0])
        normal.append(i[1])
    data['normal']=normal
    data['abnormal']=abnormal

    print("随机森林的准确率为:",accuracy_score(y,predict))
    print("随机森林的精确率为:",precision_score(y,predict,average='macro'))
    print('随机森林的召回率为:',recall_score(y,predict,average='macro'))
    print("随机森林的分类报告为:",classification_report(y,predict))
    print("随机森林的混淆矩阵为:",confusion_matrix(y,predict))
    print('*'*100)

    for i in y:
        if i ==-2 or i==-3:
            result+=1

    print("真实流失的人数为%s,随机森林预测可能流失的人数为%s,SVM预测可能流失的人数为%s" % (result, aum,bum))

    data['rtc'] = predict
    data['svm']=svmpre
    data=data[['sid','rtc','svm','status']]
    data1 = data[['sid', 'svm']]
    data1=data1[(data1['svm']==-2)]
    data1.to_csv('./data/newgksvmdata.csv', encoding='utf-8', index=None)
    data2=data[['sid','rtc']]
    data2=data2[(data['rtc']==-2)]
    data2.to_csv('./data/newgkrtcdata.csv',encoding='utf-8',index=None)
    data=data[(data['rtc']==-2)&(data['svm']==-2)]
    data.to_csv('./data/newgkmerge.csv',encoding='utf-8',index=None)
    print('预测流失为',len(data[(data['rtc']==-2)&(data['svm']==-2)]))
    print('预测真实流失为',len(data[(data['rtc']==-2)&(data['svm']==-2)&(data['status']==-2)]))

    print('存储完成')

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
    commit=int(input("请输入是训练(1)还是测试(2):"))
    if commit == 1:
        #执行训练数据
        data = pd.DataFrame(pd.read_csv("./newdata/8-11traindatas",
                           na_values=[r'\N'],
                           names=['sid','sex','scores','birthday','area_name','status','ips_nums','ips_times',
                                  'ips_complete_topicnums','ips_should_topicnums','first_accuracy_nums','ips_finish_sum',
                                  'read_numss','bookromm_times','read_mission_times','read_study_times','read_day','read_book_nums',
                                  'read_mession_firsts','read_mession_frists_nums','read_mession_complate_nums',
                                  'read_mession_student_all','read_finish_sum','stage','read_minday','read_maxday'
                                   ]))



        over_x,over_y=clean_data(data)
        # cross_score(over_x,over_y)
        # GridCV(over_x,over_y)
        train(over_x,over_y)
    elif commit == 2:
        #执行预测数据
        data = pd.read_csv("./newdata/12-3testdatas",
                           na_values=[r'\N'],
                           names=['sid','sex','scores','birthday','area_name','status','ips_nums','ips_times',
                                  'ips_complete_topicnums','ips_should_topicnums','first_accuracy_nums','ips_finish_sum',
                                  'read_numss','bookromm_times','read_mission_times','read_study_times','read_day','read_book_nums',
                                  'read_mession_firsts','read_mession_frists_nums','read_mession_complate_nums',
                                  'read_mession_student_all','read_finish_sum','stage','read_minday','read_maxday'

                                  ])


        x, y, data = clean_pridict_data(data)
        application_model(x, y, data)


    else:print("输入的指令错误")






