 # importing libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as mlt
import seaborn as sns

data=pd.read_csv('creditcard.csv')
fraud=data.loc[data['Class']==1]
normal=data.loc[data['Class']==0]

from sklearn.model_selection import train_test_split

X=data.iloc[:,:-1]
y=data['Class']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

from sklearn.linear_model import LogisticRegression
clf=LogisticRegression()
clf.fit(X_train,y_train)

y_predict=np.array(clf.predict(X_test))
Y=np.array(y_test)

from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
print(confusion_matrix(y_test, y_predict))
print(accuracy_score(y_test, y_predict))
print(classification_report(y_test, y_predict))


