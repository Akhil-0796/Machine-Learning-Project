# importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as srn

# importing  data set
data_set=pd.read_csv('emails.csv')
ham=data_set[data_set['spam']==0]
spam=data_set[data_set['spam']==1]


# applying vectorizer

from sklearn.feature_extraction.text import CountVectorizer
vectorizer=CountVectorizer()
spamham_countvectorizer=vectorizer.fit_transform(data_set['text'])
#print(spamham_countvectorizer.toarray())

label=data_set['spam'].values

from sklearn.naive_bayes import MultinomialNB
NB_classifier=MultinomialNB()
NB_classifier.fit(spamham_countvectorizer,label)

test=['	kartikmamgain2@gmail.com.Hassle-free online claiming process','	Kartik Mamgain,Only for serious riskTakers Rs.1600Crore Lottery ..']

test_vectorizer=vectorizer.transform(test)

y_predict=NB_classifier.predict(test_vectorizer)

"""# splitting dataset

X=spamham_countvectorizer
y=label

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

from sklearn.naive_bayes import MultinomialNB
NB_classifier=MultinomialNB()

NB_classifier.fit(X_train,y_train)


y_predict=NB_classifier.predict(X_test)

from sklearn.metrics import confusion_matrix,classification_report
cm=confusion_matrix(y_test,y_predict)

print(classification_report(y_test, y_predict))"""



