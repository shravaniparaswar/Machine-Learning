from sklearn.datasets import load_breast_cancer
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

#212 malignant
#357 begins


Data=load_breast_cancer()
#print(Data.feature_names)
#print(Data.target_names)

x=Data.data
y=Data.target

#print(x)
#print(y)

print('No of records' , len(y))
n=np.count_nonzero(y)

print('Number of Begine' ,n)

#print(np.where(y==0))



print('class labels ' ,np.unique(y))

X_Train ,X_Test,Y_Train,Y_Test=train_test_split(x,y,random_state=50 ,test_size=0.25)

cls=svm.SVC(kernel='linear')
cls.fit(X_Train ,Y_Train)

Y_Predict_Train=cls.predict(X_Train)
print('accuracy score on training data with SVM',\
      accuracy_score(y_true=Y_Train ,y_pred=Y_Predict_Train))


Y_Predict_Test=cls.predict(X_Test)
print('accuracy score on training data with SVM',\
      accuracy_score(y_true=Y_Test,y_pred=Y_Predict_Test))


'''
accuracy score 
'''