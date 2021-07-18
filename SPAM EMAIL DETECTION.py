
import urllib.request

import  numpy as np

import pandas as pd

from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import train_test_split

raw_data=urllib.request.urlopen(\
    'https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data')

Dataset1=np.loadtxt(raw_data ,delimiter=',')

x=Dataset1[: ,0:48]

y=Dataset1[:,-1]

#print(x)
print(y)

X_Train ,X_Test,Y_Train,Y_Test=train_test_split(x,y,test_size=0.2 ,random_state=10)


#bernoulai is a classifier for naive bayes
#when we say binarize =True value between 0.0 to 0,1 mapped and it fits in quadrant

BerNb=BernoulliNB(binarize=True)

BerNb.fit(X_Train , Y_Train)

Score=BerNb.score(X_Train ,Y_Train)

print('Spam / not spam Train Score ' ,Score)

Score=BerNb.score(X_Test ,Y_Test)

print('Spam / not spam Test Score ' ,Score)







#finally obtained train score is 0.8540760869565217

#finally obtained test score is  0.8599348534201955


