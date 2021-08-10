from mlxtend.plotting import plot_decision_regions
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.svm import SVC

iris=datasets.load_iris()

x=iris.data[: ,[0,2]]

#print(x)

y=iris.target
#print(y)

#training a classifier ,linear support vector classifier

svm=SVC(C=0.5 ,kernel='linear')
svm.fit(x,y)

plot_decision_regions(x,y,clf=svm , legend=2)

plt.xlabel('sepal length in cm [cm]')
plt.ylabel('petal length in [cm]')
plt.title('SVM ON IRIS ')
plt.show()


