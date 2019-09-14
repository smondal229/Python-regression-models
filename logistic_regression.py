from sklearn import datasets
import numpy as np
from sklearn.linear_model import LogisticRegression

iris=datasets.load_iris()
x=iris.data
y=iris.target
logreg=LogisticRegression(solver='liblinear',multi_class='auto')
logreg.fit(x,y)
pred=logreg.predict(np.array([6,6.7,0,0]).reshape(1,-1))
print(pred)
