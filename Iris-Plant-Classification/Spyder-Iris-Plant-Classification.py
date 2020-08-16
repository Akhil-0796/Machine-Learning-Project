# IMPORTING LIBRARIES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# IMPORTING DATA_SET
data=pd.read_csv('Iris.csv')
#sns.pairplot(data,hue='Species')

# SPLITTING DATA_SET INTO TRAIN_SET TEST_SET
X=data.drop('Species',axis=1)
y=data['Species'].values

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.35)


#   TRAIN MODEL 
from sklearn.neighbors import KNeighborsClassifier
clf=KNeighborsClassifier(n_neighbors=5,metric='minkowski',p=2)
clf.fit(X_train,y_train)


# EVALUATING THE RESULTS
#y_predict=clf.predict(X_test)
y_predict=clf.predict([[5.7,3.8,1.7,0.3]])

"""from sklearn.metrics import classification_report,confusion_matrix
cm=confusion_matrix(y_test,y_predict)
print(classification_report(y_test,y_predict))"""