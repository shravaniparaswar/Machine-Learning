from sklearn.datasets import make_classification

import pandas as pd

import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn import svm

from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt




X,Y=make_classification(100 ,2,2,0,weights=[0.5,0.5] ,random_state=0)

#X IS n samples n features
#y is class of each sample

#print(X)
#print(Y)

#print(np.where(Y==0))
#print(np.where(Y==1))

#Linear SVM used for large datsets
#SVC IS A CLASS IN SVM !


clf=svm.SVC(kernel='linear' , random_state=0)
clf.fit(X,Y)
print('support vectors :' , clf.support_vectors_)
print('numbers :' ,len(clf.support_vectors_))

'''
scikit learn documentation states that coef is an array of shape =[n_class-1/2 ,n_features]
Assuming 4 classes 9 features coef is shape of 6x9=54

SVM will assign weights to the feature they are given in coef 
this is avaliable in linear kernal !

'''
print(clf.coef_)

w=clf.coef_[0]

#print(w.shape)
#print(w)

a=-w[0]/w[1]
#print(a)

'''
Clf.intercept 
'''

xx=np.linspace(-5 ,5)

#print(xx)

#print(clf.intercept_[0])
#print(clf.intercept_)

yy=a*xx-(clf.intercept_[0])/w[1]

#print(yy)

#plt.plot(xx,yy)
#plt.show()
'''
support vector -array-like shape[nn_sv ,n_features]
it returns support vector

'''

#print('support vectors ' ,clf.support_vectors_)
#print('hello there')
deciFunc=clf.decision_function(clf.support_vectors_)
#print(deciFunc)

b=clf.support_vectors_[0]
print(clf.support_vectors_.shape)
print(b)
yy_down=a*xx-(b[1] - a*b[0])
print(yy_down)
plt.plot(xx , yy_down)

#plt.show()


b=clf.support_vectors_[1]
yy_up=a*xx-(b[1]-a*b[0])
print(yy_up)
plt.plot(xx,yy_up)

plt.show()

from mlxtend.plotting import plot_decision_regions


plot_decision_regions(X,Y,clf=clf)

plt.scatter(clf.support_vectors_[: ,0] , clf.support_vectors_[:,1],\
            s=80,facecolor='none')

#plt.plot(xx,y_down,'bo')

plt.plot(xx ,yy_down ,'k--')
plt.plot(xx,yy_up,'k')

plt.xlabel('X1')
plt.ylabel('X2')

plt.show()
