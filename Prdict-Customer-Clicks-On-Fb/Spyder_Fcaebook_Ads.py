# importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# import data_set
data=pd.read_csv('Facebook_Ads_2.csv',encoding='ISO-8859-1')
click=data[data['Clicked']==1]
non_click=data[data['Clicked']==0]

data.drop(['Names','emails','Country'],axis=1,inplace=True)


X=data.drop('Clicked',axis=1)
y=data['Clicked']

# feature scaling

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X=sc.fit_transform(X)

# splitting data_set

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y)

# train model

from sklearn.linear_model import LogisticRegression
clf=LogisticRegression(random_state=0)
clf.fit(X_train,y_train)


# predicting the result
y_predict_test=clf.predict(X_test)
from sklearn.metrics import confusion_matrix,classification_report
cm1=confusion_matrix(y_test,y_predict_test)  

y_predict_train=clf.predict(X_train)

cm2=confusion_matrix(y_train,y_predict_train)
report2=classification_report(y_train, y_predict_train)
