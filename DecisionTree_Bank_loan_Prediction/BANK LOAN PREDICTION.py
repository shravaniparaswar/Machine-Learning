
import pandas as pd

from matplotlib import pyplot as plt

from sklearn import tree

import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split



df=pd.read_csv("D:\OOPS\machine learning\Bank Loan.csv")

#print(df.target_names)



df.dropna(inplace=True)

df=df[['Gender','Married' ,'Self_Employed','Credit_History','Loan_Status']]

df['Gender']=df['Gender'].replace(to_replace='Male',value=1)
df['Gender']=df['Gender'].replace(to_replace='Female',value=0)

df['Married']=df['Married'].replace(to_replace='Yes',value=1)
df['Married']=df['Married'].replace(to_replace='No',value=0)


df['Self_Employed']=df['Self_Employed'].replace(to_replace='Yes',value=1)
df['Self_Employed']=df['Self_Employed'].replace(to_replace='No',value=0)

#df['Education']=df['Education'].replace(to_replace='Yes',value=1)

df.to_csv("D:\OOPS\machine learning\Bank Loan1.csv" , index=False)

X=df.drop(columns=['Loan_Status'])

Y=df['Loan_Status']

#print(X)

#print(Y)


X_train ,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3 ,random_state=42)

clf=DecisionTreeClassifier(criterion='entropy')

model=clf.fit(X_train , Y_train)

fig=tree.plot_tree(clf , feature_names=\
                   ['Gender' , 'Married' ,'Education' ,'Self_Employed','Credit_History'] ,\
                   class_names=['Yes' ,'NO'],\
                   filled=True)

plt.savefig('D:\OOPS\machine learning\Example0.png')












