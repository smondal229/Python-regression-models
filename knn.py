import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets

iris=datasets.load_iris()
x=iris.data
y=iris.target
knn=KNeighborsClassifier(n_neighbors=1)
knn.fit(x,y)
pred=knn.predict(np.array([6,6.7,3.5,1.2]).reshape(1,-1))
print(pred)
