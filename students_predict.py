import warnings
warnings.filterwarnings("ignore")
import pandas as pd#数据清洗库
from sklearn.preprocessing import MinMaxScaler#归一化
from sklearn.linear_model import LogisticRegression#逻辑回归
from sklearn.tree import DecisionTreeClassifier#决策树
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,classification_report#评估  准确率  和 分类报告 包括精确率  f1 值
from sklearn.externals import joblib #保存训练模型用的
from imblearn.over_sampling import SMOTE  # 处理不平衡数据
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, train_test_split,GridSearchCV #交叉验证  切分数据集  网格搜索
import time
import numpy as np

#数据清洗
data=pd.read_csv("./工作簿2.csv",names=["sid","durations","stars","scores","activated_at","updated_at",'status'],encoding='utf-8',dtype={'activated_at':str},na_values=r"\N")
data=data[['durations','stars','scores','activated_at','updated_at','status']].dropna()
data['start_time']=data['activated_at'].astype("int").apply(lambda x:time.strftime("%Y/%m/%d",time.localtime(x)))
data['end_time']=data['updated_at'].apply(lambda x:time.strftime("%Y/%m/%d",time.localtime(x)))
del data['activated_at']
del data['updated_at']
data["days"]=pd.to_datetime(data['end_time'])-pd.to_datetime(data['start_time'])
data['days']=data['days'].map(lambda x:x.days)
data['ave_time']=(data['durations']/data['days'])
data=data[['stars','scores','durations','status']]
# print(data.head())
min_scaler=MinMaxScaler()
x=data[['stars','scores','durations']]
x=min_scaler.fit_transform(x)
# # # print(x)
y=data['status']
# # print(data['status'].value_counts())
# #
# # #
smote=SMOTE(random_state=12)
over_samples_x,over_samples_y=smote.fit_sample(x,y)
x_train,x_test,y_train,y_test=train_test_split(over_samples_x,over_samples_y,train_size=0.7,random_state=1)

#逻辑回归
# log=LogisticRegression()
# log_score=cross_val_score(log,over_samples_x,over_samples_y,cv=5)
# print("逻辑回归的交叉验证为:%s"%log_score.mean())
#决策树
tree=DecisionTreeClassifier()
tree.fit(x_train,y_train)
predict_tree=tree.predict(x_test)
print("决策树的准确率为：",accuracy_score(y_test,predict_tree))
print("决策树的分类报告:",classification_report(y_test,predict_tree))
# tree_score=cross_val_score(tree,over_samples_x,over_samples_y,cv=5)
# print("决策树的交叉验证为：%s"%tree_score.mean())
#KNN
knn=KNeighborsClassifier(n_neighbors=4)
knn.fit(x_train,y_train)
predict_knn=knn.predict(x_test)
print("KNN的准确率为:",accuracy_score(y_test,predict_knn))
print("KNN的分类报告:",classification_report(y_test,predict_knn))

# knn_score=cross_val_score(knn,over_samples_x,over_samples_y,cv=5)
# print("knn的交叉验证为:%s"%knn_score.mean())

#随机森林
randomtree=RandomForestClassifier(oob_score=True,n_estimators=70)
randomtree.fit(x_train,y_train)
randomtree_predict=randomtree.predict(x_test)
print("随机森林的准确率为:",accuracy_score(y_test,randomtree_predict))
print("随机森林的分类报告:",classification_report(y_test,randomtree_predict))

# randomtree_score=cross_val_score(randomtree,over_samples_x,over_samples_y,cv=5)
# print("随机森林的交叉验证为%s:"%randomtree_score.mean())
#
#支持向量机
# svc=SVC()
# svc_score=cross_val_score(svc,over_samples_x,over_samples_y,cv=5)
# print("支持向量机的交叉验证为%s"%svc_score.mean())


#网格搜索找出决策树最优的参数
# paramter_tree={'max_depth':range(1,21),'min_samples_split':range(2,50,2)}
# clf=GridSearchCV(randomtree,paramter_tree,scoring='roc_auc',cv=5)
# grid=clf.fit(over_samples_x,over_samples_y)
# print(grid.best_params_)
# 逻辑回归的交叉验证为:0.5390298936475999
# 决策树的交叉验证为：0.7193201243586312
# knn的交叉验证为:0.7191466202799722
# 随机森林的交叉验证为0.7496123208875585:
# 支持向量机的交叉验证为0.5147727828382447
# 决策树网格搜索结果:{'max_depth': 20, 'min_samples_split': 4}










