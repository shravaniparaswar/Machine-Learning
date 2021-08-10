

# importing libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from mpl_toolkits.mplot3d import Axes3D

# generating data
#Make a large circle containing a smaller circle in 2d.
#X is 2D Array of outer and inner circles
#We are adding some noise to the data
X, Y = make_circles(n_samples = 500, noise = 0.02)

# visualizing data
plt.scatter(X[:, 0], X[:, 1], c = Y, marker = '.')
plt.show()

# adding a new dimension to X

#Take x coordinate in X1 , y coordinate in X2
X1 = X[:, 0].reshape((-1, 1))
X2 = X[:, 1].reshape((-1, 1))
X3 = (X1**2 + X2**2)

X = np.hstack((X, X3))

# visualizing data in higher dimension
#returns object of type figure
fig = plt.figure()

#This adds an axis to existing figure
axes = fig.add_subplot(111, projection = '3d')
axes.scatter(X1, X2, X1**2 + X2**2, c = Y, depthshade = True)
plt.show()

# create support vector classifier using a linear kernel
from sklearn import svm

svc = svm.SVC(kernel = 'linear')
svc.fit(X, Y)
#print (X)
#coef_ndarray of shape (n_classes * (n_classes - 1) / 2, n_features)
w = svc.coef_
print (w)

#intercept_ndarray of shape (n_classes * (n_classes - 1) / 2,)
#Constants in decision function.
print (svc.decision_function)
b = svc.intercept_
print (b)

# plotting the separating hyperplane
x1 = X[:, 0].reshape((-1, 1))
x2 = X[:, 1].reshape((-1, 1))
#this will give coordinates of x3 (coordinates of datapoints in terms of x1 and x2)
x1, x2 = np.meshgrid(x1, x2)
x3 = -(w[0][0]*x1 + w[0][1]*x2 + b) / w[0][2]

fig = plt.figure()
axes2 = fig.add_subplot(111, projection = '3d')
axes2.scatter(X1, X2, X1**2 + X2**2, c = Y, depthshade = True)

#get current axis
axes1 = fig.gca(projection = '3d')
#plot a 3D surface
axes1.plot_surface(x1, x2, x3, alpha = 0.01)

plt.show()

