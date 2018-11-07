import pandas as pd
import time
import warnings
warnings.filterwarnings('ignore')
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report,roc_curve
from sklearn.model_selection import cross_val_score,GridSearchCV,train_test_split

#清洗数据
def clean_data(data):
    # print(data.head())
    data.dropna(inplace=True)
    data['start_time']=data['activated_at'].apply(lambda x:time.strftime('%Y-%m-%d',time.localtime(x)))
    data['end_time']=data['updated_at'].apply(lambda x:time.strftime('%Y-%m-%d',time.localtime(x)))
    data['days']=pd.to_datetime(data['end_time'])-pd.to_datetime(data['start_time'])
    data['days']=data['days'].apply(lambda x:x.days)
    data['ave_time']=np.array(data['dura1']/data['days'],dtype=int)
    data['status']=data['status'].replace([-1,-2,-3,-4],[0,0,0,0])
    # print(data['status'])
    data=data[['stars','scores','dura1','start_time','end_time','days','ave_time','status']]
    minmax_x=MinMaxScaler()
    x=data[['stars','scores','dura1','days','ave_time']]
    x=minmax_x.fit_transform(x)
    y=data['status']
    # print(len(data['status'][data['status']==0]))
    # print(data['status'][data['status']==1].count()/data['status'][data['status']==0].count())
    '''
      比例：5 ： 1   总数据82438
    
    '''
    smot=SMOTE(random_state=42)
    over_x,over_y=smot.fit_resample(x,y)
    return over_x,over_y

#交叉验证  选择最优的模型
def cross_score(over_x,over_y):
    lr=LogisticRegression()
    dtc=DecisionTreeClassifier()
    rfc=RandomForestClassifier()
    # svm=SVC()
    gs=GaussianNB()
    mlpc=MLPClassifier()
    xgbc=XGBClassifier()
    gbc=GradientBoostingClassifier
    cv1=cross_val_score(lr,over_x,over_y,scoring='roc_auc',cv=5)
    print("逻辑回归的交叉验证为:",cv1.mean())
    cv2 = cross_val_score(dtc, over_x, over_y, scoring='roc_auc', cv=5)
    print("决策树的交叉验证为:", cv2.mean())
    cv3 = cross_val_score(rfc, over_x, over_y, scoring='roc_auc', cv=5)
    print("随机森林的交叉验证为:", cv3.mean())
    cv4 = cross_val_score(gs, over_x, over_y, scoring='roc_auc', cv=5)
    print("高斯的交叉验证为:", cv4.mean())
    cv5 = cross_val_score(mlpc, over_x, over_y, scoring='roc_auc', cv=5)
    print("神经网络的交叉验证为:", cv5.mean())
    cv6 = cross_val_score(xgbc, over_x, over_y, scoring='roc_auc', cv=5)
    print("xgbost的交叉验证为:", cv6.mean())
    cv7= cross_val_score(gbc,over_x,over_y,scoring='roc_auc',cv=5)
    print("梯度提升决策树的交叉验证为:",cv7.mean())

#训练数据
def train(over_x,over_y):
    train_x,test_x,train_y,test_y=train_test_split(over_x,over_y,train_size=0.6,random_state=123)
    rfc=RandomForestClassifier(oob_score=True,
                               random_state=1,
                               n_estimators=70,
                               )
    rfc.fit(train_x,train_y)
    predict_y=rfc.predict(test_x)
    print("袋外分数为:",rfc.oob_score_)
    print(accuracy_score(test_y,predict_y))
    print(classification_report(test_y,predict_y))
    print(confusion_matrix(test_y,predict_y))
    fpr,fpc,th=roc_curve(test_y,rfc.predict_proba(test_x)[:,1])
    plt.plot(fpr,fpc)
    plt.title("ROC")
    plt.show()
#网格搜索
def gridcv(over_x,over_y):
    param_grid={
        'max_depth':range(3,14,2),
        'min_samples_split':range(50,201,20)
    }
    grid=GridSearchCV(RandomForestClassifier(n_estimators=70,oob_score=True),param_grid=param_grid,scoring='roc_auc',cv=5)
    grid.fit(over_x,over_y)
    print(grid.best_score_)
    print(grid.best_params_)
    '''
      默认参数  袋外分数  即模型的泛化能力 0.7607646310127039
      0.8427520867822661
      {'max_depth': 13, 'min_samples_split': 50}
      
    
    '''


if __name__ == '__main__':
    data=pd.read_csv('./students_original_all.csv',encoding='utf-8',names=['sid','stars','scores','dura1','ips_all','count_num_ips','accuracy_all_ips','class_id','activated_at','updated_at','stage','status'],na_values=r'\N')
    over_x,over_y=clean_data(data)
    # cross_score(over_x,over_y)
    # gridcv(over_x,over_y)
    train(over_x,over_y)
