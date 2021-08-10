import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

x,y=make_circles(n_samples=500 ,noise=0.02)

plt.scatter(x[:,0],x[:,1] , c=y,marker='.')
plt.show()


x1=x[:,0].reshape((-1 ,1))
x2=x[:,1].reshape((-1 ,1))
x3=(x1**2+x2**2)

x=np.hstack((x,x3))

fig=plt.figure()
axes=fig.add_subplot(111 , projection='3d')
axes.scatter(x1,x2,x1**2 + x2**2 ,c=y ,depthshade=True)
plt.show()
