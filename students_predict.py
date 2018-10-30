import pandas as pd#数据清洗库
from sklearn.preprocessing import MinMaxScaler#归一化
from sklearn.linear_model import LogisticRegression#逻辑回归
from sklearn.tree import DecisionTreeClassifier#决策树
from sklearn.metrics import accuracy_score,classification_report#评估  准确率  和 分类报告 包括精确率  f1 值
from sklearn.externals import joblib #保存训练模型用的
from imblearn.over_sampling import SMOTE  # 处理不平衡数据
from sklearn.model_selection import cross_val_score, train_test_split,GridSearchCV #交叉验证  切分数据集  网格搜索
data=pd.read_csv("./lb_students_screen.csv",encoding='utf-8')
min_scaler=MinMaxScaler()
x=data[['stars','scores']]
x=min_scaler.fit_transform(x)
y=data['status']

x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.7)
# unclassifieldScale=data['status'][data['status']==-2].count()/data['status'][data['status']==-3].count()
# print("未重采样前的比例为:%s"%unclassifieldScale)

#逻辑回归
log=LogisticRegression()
log_score=cross_val_score(log,x,y,cv=5)
print("逻辑回归的交叉验证为:%s"%log_score.mean())
#决策树
tree=DecisionTreeClassifier(max_depth=1,min_samples_split=2)
tree_score=cross_val_score(tree,x,y,cv=5)
print("决策树的交叉验证为：%s"%tree_score.mean())
# 逻辑回归的交叉验证为:0.8372093109897711
# 决策树的交叉验证为：0.7271552443496654

#网格搜索找出决策树最优的参数
# paramter_tree={'max_depth':range(1,21),'min_samples_split':range(2,30,2)}
# clf=GridSearchCV(tree,paramter_tree,scoring='accuracy',cv=5)
# grid=clf.fit(x,y)
# print(grid.best_params_)
# {'max_depth': 1, 'min_samples_split': 2}






over_samples=SMOTE(random_state=1234)
over_samples_x,over_samples_y=over_samples.fit_sample(x_train,y_train)
log2=log.fit(over_samples_x,over_samples_y)
# joblib.dump(log,'./log.pkl'){'max_depth': 1, 'min_samples_split': 2}
# log2=joblib.load('log.pkl')
log_predict=log2.predict(x_test)
tree2=tree.fit(over_samples_x,over_samples_y)
pro_tree=tree2.predict(x_test)


print("逻辑回归准确率为:%s"%accuracy_score(y_test,log_predict))
print("逻辑回归分类报告为:%s"%classification_report(y_test,log_predict))
print("决策树准确率为:%s"%accuracy_score(y_test,pro_tree))
print("决策树分类报告为:%s"%classification_report(y_test,pro_tree))

