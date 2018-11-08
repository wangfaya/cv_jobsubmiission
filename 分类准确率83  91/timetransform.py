import time
import warnings
warnings.filterwarnings('ignore')
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report,roc_curve
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score

def clean_data(data):
    data.dropna(inplace=True)
    data['start_time'] = data['activated_at'].apply(lambda x: time.strftime('%Y-%m-%d', time.localtime(x)))
    data['end_time'] = data['updated_at'].apply(lambda x: time.strftime('%Y-%m-%d', time.localtime(x)))
    data['days'] = pd.to_datetime(data['end_time']) - pd.to_datetime(data['start_time'])
    data['days'] = data['days'].apply(lambda x: x.days)
    data['ave_time'] = np.array(data['dura1'] / data['days'], dtype=int)
    data['status'] = data['status'].replace([-1, -2, -3, -4], [0, 0, 0, 0])
    # print(data['status'])
    data = data[['stars', 'scores', 'dura1', 'start_time', 'end_time', 'days', 'ave_time', 'status']]
    data['start_time']=pd.to_datetime(data['start_time'])
    data=data.set_index('start_time')
    data=data['2018']
    # print(data.head())
    data = data[['stars', 'scores', 'dura1', 'days', 'ave_time', 'status']]
    minmax_x = MinMaxScaler()
    x = data[['stars', 'scores', 'dura1', 'days', 'ave_time']]
    x = minmax_x.fit_transform(x)
    y = data['status']
    # print(len(data['status'][data['status']==0]))

    '''
      print(data['status'][data['status']==1].count()/data['status'][data['status']==0].count())
      print(len(data))
      比例：29 ： 1   总数据29078
    
    '''
    smot = SMOTE(random_state=42)
    over_x, over_y = smot.fit_resample(x, y)
    return over_x,over_y
def cross_score(over_x,over_y):
    lr=LogisticRegression()
    dtc=DecisionTreeClassifier()
    rfc=RandomForestClassifier()
    # svm=SVC()
    gs=GaussianNB()
    mlpc=MLPClassifier()
    xgbc=XGBClassifier()
    gbc=GradientBoostingClassifier()
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

def train(x,y):
    train_x, test_x, train_y, test_y = train_test_split(over_x, over_y, train_size=0.6, random_state=123)
    rfc = RandomForestClassifier(oob_score=True,
                                 random_state=1,
                                 n_estimators=70,
                                 )
    rfc.fit(train_x, train_y)
    predict_y = rfc.predict(test_x)
    print(predict_y)
    feature_names=['stars', 'scores', 'dura1', 'days', 'ave_time']
    print('特征重要度排序为:',sorted(zip(map(lambda x:round(x,4),rfc.feature_importances_),feature_names),reverse=True))
    print("袋外分数为:", rfc.oob_score_)
    pro_pridict=rfc.predict_proba(test_x)
    print(pro_pridict)
    print(accuracy_score(test_y, predict_y))
    print(classification_report(test_y, predict_y))
    print(confusion_matrix(test_y, predict_y))
    prf,prc,th=roc_curve(test_y,rfc.predict_proba(test_x)[:,1])
    plt.plot(prf,prc)
    plt.title("ROC")
    plt.show()



if __name__ == '__main__':
    data = pd.read_csv("./students_original_all.csv", encoding='utf-8', na_values=r'\N',
                       names=['sid', 'stars', 'scores', 'dura1', 'ips_all', 'count_num_ips', 'accuracy_all_ips',
                              'class_id', 'activated_at', 'updated_at', 'stage', 'status'])
    over_x,over_y=clean_data(data)
    # cross_score(over_x,over_y)
    train(over_x,over_y)



