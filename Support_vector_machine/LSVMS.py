
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import make_classification

x=np.array([[1,2] ,[5,8] ,[1.5,1.8] , [8,8] ,[1,0.6] ,[9,11]])

y=[0,1,0,1,0,1]
'''
variableas that are measured at different scales do not contribute to model fitting
'''
from sklearn.preprocessing import StandardScaler

sc=StandardScaler()
x=sc.fit_transform(x)
print(x)

from sklearn.svm import SVC
classifier=SVC(kernel='linear',random_state=0)
classifier.fit(x,y)

x_set , y_set=x,y
x1,x2=np.meshgrid(np.arange(start=x_set[:,0].min()-1 ,\
                            stop=x_set[:,0].max()+1, step=0.01), \
                  np.arange(start=x_set[:, 1].min() - 1, \
                            stop=x_set[:, 1].max() +1 ,step=0.01),
                  )


#print(np.ravel(x1))
#print((np.ravel(x1).shape))
#print((np.array([x1.ravel(),  x2.ravel()]).T).reshape(x1.shape))

from matplotlib.colors import ListedColormap





plt.contourf(x1, x2, \
             classifier.predict(np.array([x1.ravel() , x2.ravel() ]).T).reshape(x1.shape), \
             alpha=0.75, cmap=ListedColormap(('blue', 'green')))

plt.xlim(x1.min() , x1.max())
plt.ylim(x2.min() , x2.max())

'''
print(y_set)
print(np.unique(y_set))
print(enumerate(np.unique(y_set)))
'''
print(type(x_set))
print(x_set)



for i ,j in enumerate(np.unique(y_set)):
    print(i ,j)
    plt.scatter(x_set[y_set==j,0], x_set[y_set==j,1],
                c=ListedColormap(('blue', 'green')) (i), label=1)

plt.title('SVM')
plt.legend()
plt.show()
