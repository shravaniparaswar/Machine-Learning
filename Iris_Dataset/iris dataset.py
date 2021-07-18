

from sklearn import datasets

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt


Iris1 = datasets.load_iris()

print(Iris1)

print(Iris1['data'])

print(Iris1['target'])

print(Iris1['feature_names'])


Df1= pd.DataFrame(data = np.c_[Iris1['data'] ,Iris1['target'] ],  \
                  columns = Iris1['feature_names']+ ['species'])

#print(Df1.head())

Df1.hist()

plt.show()


Df1.plot()

plt.show()


Df1.groupby(by = 'species').mean().plot(kind='bar')

plt.grid(True)

plt.xticks(rotation = 30)
plt.legend(loc ='upper left' , bbox_to_anchor =(1,1))

plt.show()




