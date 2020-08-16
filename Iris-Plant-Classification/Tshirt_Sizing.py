# IMPORT LIBRARIES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# IMPORT DATA_SET

data=pd.read_csv('Tshirt_Sizing_Dataset.csv')
#sns.pairplot(data,hue='T Shirt Size')

# SPLITIING DATA_SET INTO TARIN_SET AND TEST_SET
X=data.drop('T Shirt Size',axis=1)
y=data['T Shirt Size'].values

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)


# TRAIN MODEL
from sklearn.neighbors import KNeighborsClassifier
clf=KNeighborsClassifier(n_neighbors=5,metric='minkowski',p=2)
clf.fit(X_train,y_train)

y_predict=clf.predict([[163,64]])

"""# EVALUATING THE RESULTS
y_predict=clf.predict(X_test)
from sklearn.metrics import confusion_matrix,classification_report
cm=confusion_matrix(y_test,y_predict)

print(classification_report(y_test,y_predict))"""
